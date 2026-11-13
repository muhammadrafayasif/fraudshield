"""FraudShield API: two specialist fraud models behind one platform.

Channels
    paysim  - mobile-money transfers   (models/fraudshield_paysim.joblib)
    sparkov - card payments            (models/fraudshield_sparkov.joblib)

Every incoming transaction names its channel; the matching specialist scores it,
using the decision threshold selected on the validation split during training
(models/fraudshield_<channel>_metadata.json).
"""
import asyncio
import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

log = logging.getLogger("fraudshield")

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
DATABASE_PATH = DATA_DIR / "transactions.db"
MAX_BATCH = 5000


# --------------------------------------------------------------------------- #
# Request schemas (one per channel, selected by the `channel` field)
# --------------------------------------------------------------------------- #
class PaySimTransaction(BaseModel):
    channel: Literal["paysim"]
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


class SparkovTransaction(BaseModel):
    channel: Literal["sparkov"]
    category: Literal[
        "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
        "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
        "personal_care", "shopping_net", "shopping_pos", "travel",
    ]
    amt: float = Field(ge=0)
    city_pop: int = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)  # 0 = Monday
    age: float = Field(ge=0, le=120)
    distance_km: float = Field(ge=0)  # customer home -> merchant


AnyTransaction = Annotated[
    Union[PaySimTransaction, SparkovTransaction], Field(discriminator="channel")
]

SEGMENT_FIELD = {"paysim": "type", "sparkov": "category"}
AMOUNT_FIELD = {"paysim": "amount", "sparkov": "amt"}
# every column a channel's model could be asked for (guards against a stale model/metadata pair)
AVAILABLE_FEATURES = {
    "paysim": {"type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
               "newbalanceDest", "errorBalanceOrig", "errorBalanceDest"},
    "sparkov": {"category", "amt", "city_pop", "hour", "day_of_week", "age", "distance_km"},
}


# --------------------------------------------------------------------------- #
# Model loading: a channel that fails to load is disabled, the other keeps working
# --------------------------------------------------------------------------- #
class ChannelModel:
    def __init__(self, key: str) -> None:
        self.key = key
        self.pipeline: Any = None
        self.meta: dict[str, Any] = {}
        self.threshold = 0.5
        self.features: list[str] = []
        self.error: str | None = None
        try:
            self.meta = json.loads((MODELS_DIR / f"fraudshield_{key}_metadata.json").read_text())
            self.threshold = float(self.meta["threshold"])
            self.features = list(self.meta["categorical_features"]) + list(self.meta["numeric_features"])
            unknown = set(self.features) - AVAILABLE_FEATURES[key]
            if unknown:
                raise ValueError(f"model expects features the API cannot build: {sorted(unknown)}")
            self.pipeline = joblib.load(MODELS_DIR / f"fraudshield_{key}.joblib")
        except Exception as exc:  # noqa: BLE001 - report any load problem on the channel
            self.pipeline = None
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("Could not load the %s model: %s", key, self.error)

    @property
    def available(self) -> bool:
        return self.pipeline is not None


MODELS = {key: ChannelModel(key) for key in ("paysim", "sparkov")}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def build_frame(key: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if key == "paysim":
        frame["errorBalanceOrig"] = frame["newbalanceOrig"] + frame["amount"] - frame["oldbalanceOrg"]
        frame["errorBalanceDest"] = frame["oldbalanceDest"] + frame["amount"] - frame["newbalanceDest"]
    return frame[MODELS[key].features]


def score_transactions(items: list[BaseModel]) -> list[dict[str, Any]]:
    """Score a mixed batch: one vectorised model call per channel, original order kept."""
    positions: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        positions.setdefault(item.channel, []).append(index)  # type: ignore[attr-defined]

    results: list[dict[str, Any] | None] = [None] * len(items)
    for key, indexes in positions.items():
        model = MODELS[key]
        if not model.available:
            raise HTTPException(
                status_code=503,
                detail=f"The '{key}' model is not available: {model.error}",
            )
        inputs = [items[i].model_dump() for i in indexes]
        probabilities = model.pipeline.predict_proba(build_frame(key, inputs))[:, 1]
        for i, row, probability in zip(indexes, inputs, probabilities):
            results[i] = {
                "id": None,
                "channel": key,
                "segment": str(row[SEGMENT_FIELD[key]]),
                "amount": float(row[AMOUNT_FIELD[key]]),
                "inputs": row,
                "fraud_probability": float(probability),
                "is_fraud": bool(probability >= model.threshold),
                "threshold": model.threshold,
                "model": str(model.meta.get("model", "unknown")),
                "created_at": None,
            }
    return results  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Storage (SQLite). Scored results are stored, so analytics never re-run the models.
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS transaction_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT    NOT NULL,
    channel          TEXT    NOT NULL,
    segment          TEXT    NOT NULL,
    amount           REAL    NOT NULL,
    payload          TEXT    NOT NULL,
    fraud_probability REAL   NOT NULL,
    is_fraud         INTEGER NOT NULL,
    threshold        REAL    NOT NULL,
    model            TEXT    NOT NULL
)
"""


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with closing(connect()) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        with connection:
            connection.execute(SCHEMA)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_log_channel ON transaction_log (channel, segment)"
            )


init_db()


def row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "channel": row["channel"],
        "segment": row["segment"],
        "amount": row["amount"],
        "inputs": json.loads(row["payload"]),
        "fraud_probability": row["fraud_probability"],
        "is_fraud": bool(row["is_fraud"]),
        "threshold": row["threshold"],
        "model": row["model"],
        "created_at": row["created_at"],
    }


def score_and_store(transaction: BaseModel) -> dict[str, Any]:
    record = score_transactions([transaction])[0]
    record["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with closing(connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO transaction_log
                        (created_at, channel, segment, amount, payload,
                         fraud_probability, is_fraud, threshold, model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["created_at"], record["channel"], record["segment"],
                        record["amount"], json.dumps(record["inputs"]),
                        record["fraud_probability"], int(record["is_fraud"]),
                        record["threshold"], record["model"],
                    ),
                )
                record["id"] = cursor.lastrowid
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=500, detail=f"Could not store transaction: {error}"
        ) from error
    return record


# --------------------------------------------------------------------------- #
# Model information for the dashboard (metrics, importances, comparison, fairness)
# --------------------------------------------------------------------------- #
def read_csv_records(path: Path, limit: int | None = None) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if limit:
        frame = frame.head(limit)
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def build_model_info() -> dict[str, Any]:
    channels: dict[str, Any] = {}
    for key, model in MODELS.items():
        channels[key] = {
            "available": model.available,
            "error": model.error,
            "model": model.meta.get("model"),
            "threshold": model.meta.get("threshold"),
            "metrics": model.meta.get("test_metrics_at_threshold"),
            "confusion_matrix": model.meta.get("confusion_matrix"),
            "rows": model.meta.get("rows"),
            "top_features": read_csv_records(REPORTS_DIR / f"feature_importance_{key}.csv", 8) or [],
            "fairness": read_csv_records(REPORTS_DIR / "fairness_audit_sparkov_gender.csv")
            if key == "sparkov" else None,
        }
    return {
        "channels": channels,
        "comparison": read_csv_records(MODELS_DIR / "model_comparison.csv") or [],
    }


MODEL_INFO = build_model_info()


# --------------------------------------------------------------------------- #
# Live events (Server-Sent Events)
# --------------------------------------------------------------------------- #
event_subscribers: set[asyncio.Queue[str]] = set()


def publish_event(record: dict[str, Any]) -> None:
    event = f"data: {json.dumps(record)}\n\n"
    for subscriber in tuple(event_subscribers):
        try:
            subscriber.put_nowait(event)
        except asyncio.QueueFull:  # a stalled client must not grow memory without bound
            pass


async def event_stream():
    subscriber: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
    event_subscribers.add(subscriber)
    try:
        yield ": connected\n\n"
        while True:
            try:
                yield await asyncio.wait_for(subscriber.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"  # stops proxies from closing an idle stream
    finally:
        event_subscribers.discard(subscriber)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="FraudShield API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "FRAUDSHIELD_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "channels": {
            key: {"available": m.available, "model": m.meta.get("model"), "error": m.error}
            for key, m in MODELS.items()
        },
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return MODEL_INFO


@app.post("/predict")
def predict(
    transactions: Annotated[list[AnyTransaction], Body(max_length=MAX_BATCH)],
) -> list[dict[str, Any]]:
    """Score transactions of either channel without storing them."""
    return score_transactions(transactions) if transactions else []


@app.post("/transaction", status_code=201)
async def commit_transaction(transaction: AnyTransaction) -> dict[str, Any]:
    """Score one transaction, store it, and push it to every live dashboard."""
    record = await run_in_threadpool(score_and_store, transaction)
    publish_event(record)
    return record


@app.get("/events")
async def events() -> StreamingResponse:
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def read_analytics(limit: int) -> dict[str, Any]:
    channels: dict[str, Any] = {
        key: {"total": 0, "flagged": 0, "volume": 0.0, "avg_risk": 0.0, "segments": {}}
        for key in MODELS
    }
    try:
        with closing(connect()) as connection:
            groups = connection.execute(
                """
                SELECT channel, segment, COUNT(*) AS n, SUM(is_fraud) AS flagged,
                       SUM(amount) AS volume, SUM(fraud_probability) AS risk
                FROM transaction_log GROUP BY channel, segment
                """
            ).fetchall()
            latest = connection.execute(
                """
                SELECT * FROM (SELECT * FROM transaction_log ORDER BY id DESC LIMIT ?)
                ORDER BY id ASC
                """,
                (limit,),
            ).fetchall()
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=500, detail=f"Could not read transactions: {error}"
        ) from error

    risk_sums = {key: 0.0 for key in channels}
    for group in groups:
        summary = channels.setdefault(
            group["channel"],
            {"total": 0, "flagged": 0, "volume": 0.0, "avg_risk": 0.0, "segments": {}},
        )
        summary["total"] += group["n"]
        summary["flagged"] += group["flagged"] or 0
        summary["volume"] += group["volume"] or 0.0
        summary["segments"][group["segment"]] = {"total": group["n"], "flagged": group["flagged"] or 0}
        risk_sums[group["channel"]] = risk_sums.get(group["channel"], 0.0) + (group["risk"] or 0.0)
    for key, summary in channels.items():
        summary["avg_risk"] = risk_sums.get(key, 0.0) / summary["total"] if summary["total"] else 0.0

    return {
        "total_transactions": sum(s["total"] for s in channels.values()),
        "channels": channels,
        "transactions": [row_to_record(row) for row in latest],
    }


@app.get("/analytics")
def analytics(limit: Annotated[int, Query(ge=1, le=2000)] = 300) -> dict[str, Any]:
    return read_analytics(limit)