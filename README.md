<p align="center">
  <img src="https://img.shields.io/badge/Complex_Computing_Project-4B5563?style=for-the-badge" alt="Complex Computing Project" />
</p>

<h1 align="center">FraudShield: An ML-Based Fraud Detection System for Digital Payment</h1>

<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/8428/8428639.png" alt="FraudShield icon" width="160" />
</p>

FraudShield is an end-to-end fraud detection platform for digital payment transactions. It combines specialist machine-learning models for mobile-money and card-payment data with a web dashboard for scoring transactions, monitoring risk, and exploring model performance.

## Overview

FraudShield supports two payment channels:

- **PaySim**: mobile-money transactions such as transfers, payments, cash-outs, debits, and cash-ins.
- **Sparkov**: card payments with merchant categories, transaction amounts, timing, customer age, city population, and distance features.

Each channel has its own trained model, validation-selected decision threshold, feature metadata, and evaluation results. The API stores scored transactions in SQLite so analytics can be viewed without re-running the models.

## Features

- Score individual transactions or mixed batches of transactions.
- Use separate specialist models for PaySim and Sparkov data.
- View fraud probability, classification threshold, model name, and transaction details.
- Monitor live prediction events through server-sent events.
- Explore channel summaries, flagged transactions, risk trends, and model metrics.
- Compare model performance and inspect top features.
- Run demo traffic from the dashboard using representative transaction templates.

## Tech Stack

### Backend

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" title="Python" alt="Python" width="42" height="42" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg" title="FastAPI" alt="FastAPI" width="42" height="42" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" title="Pandas" alt="Pandas" width="42" height="42" />
  <img src="https://cdn.simpleicons.org/scikitlearn/F7931E" title="scikit-learn" alt="scikit-learn" width="42" height="42" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/sqlite/sqlite-original.svg" title="SQLite" alt="SQLite" width="42" height="42" />
</p>

### Frontend

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/react/react-original.svg" title="React" alt="React" width="42" height="42" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/typescript/typescript-original.svg" title="TypeScript" alt="TypeScript" width="42" height="42" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vitejs/vitejs-original.svg" title="Vite" alt="Vite" width="42" height="42" />
</p>

## Project Structure

```text
fraudshield/
├── api/                 # FastAPI service
├── data/                # Datasets and transaction database
├── models/              # Trained models and model metadata
├── notebooks/           # Training and visualization notebooks
├── reports/             # Evaluation tables, plots, and audit outputs
├── docs/                # Technical report sources and PDF
└── src/                 # React/Vite dashboard
    ├── public/          # Static assets, including icon.png
    └── src/             # Frontend application source
```

## Getting Started

### 1. Start the API

From the project root, install the Python packages used by the API and run Uvicorn:

```bash
python -m pip install fastapi "uvicorn[standard]" joblib pandas scikit-learn
python -m uvicorn api.index:app --reload
```

The API is available at `http://localhost:8000`. Interactive API documentation is available at `http://localhost:8000/docs`.

### 2. Start the dashboard

```bash
cd src
npm install
npm run dev
```

Open the local Vite URL shown in the terminal. The dashboard expects the API at `http://localhost:8000`.

### Production build

```bash
cd src
npm run build
npm run preview
```

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | API status |
| `GET` | `/model-info` | Model metadata, metrics, and feature importance |
| `POST` | `/predict` | Score one or more transactions without storing them |
| `POST` | `/transaction` | Score and store a transaction |
| `GET` | `/events` | Stream newly scored transactions |
| `GET` | `/analytics` | Retrieve stored transaction analytics |

## Research Materials

- [`docs/report.pdf`](docs/report.pdf) contains the project report.
- [`notebooks/train.ipynb`](notebooks/train.ipynb) contains the model-training workflow.
- [`notebooks/plots.ipynb`](notebooks/plots.ipynb) contains visualization work.
- `reports/` contains model comparisons, feature-importance tables, fairness outputs, and figures.

## Development

Frontend checks can be run from `src/`:

```bash
npm run lint
npm run build
```

FraudShield is an academic and research project. Model predictions should be validated against operational data and reviewed with appropriate human, privacy, and regulatory safeguards before production use.