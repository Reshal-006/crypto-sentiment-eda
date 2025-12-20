# Crypto Sentiment EDA Dashboard

Multi-asset crypto analytics dashboard (Flask) powered by a notebook pipeline.

This project combines:
- A notebook-based data pipeline (collection + processing)
- A web dashboard to visualize prices, returns/volatility, and the Crypto Fear & Greed Index

## Key Features

- Select multiple cryptocurrencies in the UI (e.g., BTC, ETH, BNB, XRP, SOL)
- Choose a date range and click **Fetch Data**
- The dashboard triggers the notebook pipeline headlessly (01 → 02) and reloads the processed dataset
- Charts and tables render dynamically for all selected assets

## How The Data Works

All data collection happens in the notebooks.

- `notebooks/01_data_collection.ipynb`
  - Downloads price data for the selected tickers
  - Fetches the Fear & Greed Index
  - Writes raw files to `data/raw/`
- `notebooks/02_data_cleaning.ipynb`
  - Loads raw `*_prices.csv` files
  - Engineers features (returns, moving averages, rolling volatility)
  - Writes the processed dataset to `data/processed/merged_clean.csv`

The dashboard reads the processed CSV and serves charts via API endpoints.

## Outputs

After a successful fetch/pipeline run:
- `data/raw/<TICKER>_prices.csv` (one per selected asset)
- `data/raw/fear_greed_index.csv` (and/or trimmed variant depending on the pipeline)
- `data/processed/merged_clean.csv`

The processed dataset typically includes columns like:
- `Close_<T>`, `Volume_<T>`
- `<T>_Return`, `<T>_MA7`, `<T>_MA30`, `<T>_Vol30`
- `FG_Value`, `value_classification`

## Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the dashboard

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

## Using The Dashboard

1) Select one or more cryptocurrencies
2) Select a start and end date
3) Click **Fetch Data**

The dashboard will run the notebook pipeline and refresh:
- Summary cards
- Price & Sentiment time series
- Analysis charts
- Data table

## Notebooks (Optional)

You can also run the notebooks manually (in order) for exploration:
1) `notebooks/01_data_collection.ipynb`
2) `notebooks/02_data_cleaning.ipynb`
3) `notebooks/03_eda.ipynb`

## Notes

- Internet access is required (Yahoo Finance + Fear & Greed API)
- If you select assets that are not present in `merged_clean.csv`, re-run the pipeline with those assets selected
