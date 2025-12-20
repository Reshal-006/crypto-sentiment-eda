***

# Cryptocurrency Price and Sentiment EDA

Exploratory data analysis (EDA) of Bitcoin and Ethereum daily price data together with the Crypto Fear & Greed Index. The project focuses on how prices, returns, volatility, and market sentiment behave over time and how they relate to each other.

## Features

- Downloads daily OHLCV data for BTC and ETH from Yahoo Finance  
- Fetches the Crypto Fear & Greed Index and aligns it with price data  
- Cleans and merges all sources into a single time-series dataset  
- Engineers returns, moving averages, and rolling volatility features  
- Performs EDA on:
  - Price and return distributions  
  - Sentiment distribution and regimes (Fear vs Greed)  
  - Correlations between returns, volatility, and sentiment  
  - Simple lag analysis (does sentiment lead returns?)  

## Tech Stack

- Python
- pandas, NumPy  
- yfinance, requests  
- matplotlib, seaborn  
- Jupyter Notebooks  

## Project Structure

```text
.
├── data
│   ├── raw
│   │   ├── btc_prices.csv          # Raw BTC OHLCV
│   │   ├── eth_prices.csv          # Raw ETH OHLCV
│   │   └── fear_greed_index.csv    # Raw sentiment data
│   └── processed
│       └── merged_clean.csv        # Cleaned, feature-rich dataset
├── notebooks
│   ├── 01_data_collection.ipynb    # Download price + sentiment data
│   ├── 02_data_cleaning.ipynb      # Cleaning + feature engineering
│   └── 03_eda.ipynb                # Exploratory data analysis
└── README.md
```

## Dataset

After processing, the main dataset `merged_clean.csv` contains one row per trading day with:

- `Close_BTC`, `Close_ETH` – daily closing prices (USD)  
- `Volume_BTC`, `Volume_ETH` – daily traded volume  
- `BTC_Return`, `ETH_Return` – daily returns  
- `BTC_MA7`, `BTC_MA30`, `ETH_MA7`, `ETH_MA30` – 7-day & 30-day moving averages  
- `BTC_Vol30`, `ETH_Vol30` – 30-day rolling volatility of returns  
- `FG_Value` – Fear & Greed Index score (0–100)  
- `value_classification` – sentiment label (Extreme Fear, Fear, Neutral, Greed, Extreme Greed)  

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and activate an environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don’t use a `requirements.txt`, install manually:

```bash
pip install pandas numpy yfinance requests matplotlib seaborn
```

### 4. Run the notebooks

Run the notebooks in order inside Jupyter or VS Code:

1. `01_data_collection.ipynb` – downloads and saves raw data into `data/raw/`  
2. `02_data_cleaning.ipynb` – loads raw CSVs, cleans and merges into `data/processed/merged_clean.csv`  
3. `03_eda.ipynb` – performs EDA and generates plots and tables  

## Main Analyses

- Time-series plots of BTC/ETH prices vs Fear & Greed Index  
- Distribution plots for returns and sentiment  
- Correlation heatmaps for returns, volatility, and sentiment  
- Scatter plots of sentiment vs returns  
- Boxplots of returns across sentiment regimes  
- Simple lag correlation checks using 1-day and 3-day lagged sentiment values  

## Possible Extensions

- Add more cryptocurrencies or alternative sentiment indices  
- Try different rolling windows for volatility  
- Add basic forecasting models (ARIMA, GARCH, simple ML) on top of this EDA  

***
