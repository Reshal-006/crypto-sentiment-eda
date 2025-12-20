"""Crypto Sentiment EDA Dashboard (Flask).

Serves a small dashboard UI + JSON APIs backed by a processed CSV produced by
the notebook pipeline.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend (safe for server use)
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from flask import Flask, render_template, jsonify, request
import numpy as np
from datetime import datetime
from pathlib import Path
import os
import threading
import math
import time

app = Flask(__name__)

# Dev-friendly caching behavior: avoid stale templates/static files so the browser
# doesn't keep running an older JS bundle.
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.after_request
def _no_store_cache_headers(resp):
    resp.headers['Cache-Control'] = 'no-store'
    return resp

# Default locations + initial UI selections
DATA_PATH = "data/processed/merged_clean.csv"
DEFAULT_START_DATE = "2023-01-02"
DEFAULT_END_DATE = "2025-12-16"


# In-memory app state (filtered views of the processed CSV)
class DataHolder:
    def __init__(self):
        self.df = None
        self.start_date = DEFAULT_START_DATE
        self.end_date = DEFAULT_END_DATE
        self.selected_cryptos = ["BTC-USD", "ETH-USD"]
    
    def load_default(self):
        """Load the processed dataset from disk (if available)."""
        try:
            self.df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Error loading default data: {e}")
            self.df = pd.DataFrame()
    
    def apply_selection(self, start_date: str, end_date: str, crypto_symbols=None):
        """Apply UI selections without downloading data (data collection lives in the notebook)."""
        if crypto_symbols:
            self.selected_cryptos = crypto_symbols
        self.start_date = start_date
        self.end_date = end_date
        return True, "Selections applied"
    
    def get_filtered_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Return the processed dataset filtered to a date window."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        df = self.df.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        return df

# Bootstrap state
data_holder = DataHolder()
data_holder.load_default()
df = data_holder.df

# Notebook pipeline execution (papermill)
PIPELINE_LOCK = threading.Lock()

# Matplotlib -> base64 PNG (for <img> tags)
def fig_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return plot_url


def _tickers_from_symbols(symbols):
    if not symbols:
        return []
    out = []
    for sym in symbols:
        if not sym:
            continue
        out.append(str(sym).split('-')[0].upper())
    return out


def _available_tickers(df_: pd.DataFrame) -> list[str]:
    if df_ is None or df_.empty:
        return []
    tickers = set()
    for col in df_.columns:
        if col.startswith('Close_'):
            tickers.add(col.split('_', 1)[1].upper())
    return sorted(tickers)


def _selected_tickers(df_: pd.DataFrame) -> list[str]:
    requested = _tickers_from_symbols(getattr(data_holder, 'selected_cryptos', None))
    available = set(_available_tickers(df_))
    selected = [t for t in requested if t in available]
    return selected or sorted(available)


def _require_ticker_column(df_: pd.DataFrame, col: str, ticker: str) -> None:
    if df_ is None or df_.empty or col not in df_.columns:
        raise KeyError(
            f"{ticker} data is not available for the current processed dataset. "
            f"Include {ticker} in your Fetch selection or re-run the pipeline with {ticker}-USD."
        )

# Dashboard page
@app.route('/')
def index():
    return render_template('index.html', cache_bust=int(time.time()))

# Date picker metadata (UI guidance, not a hard constraint)
@app.route('/api/date-range')
def get_date_range():
    # Keep the picker flexible; we allow any date from 2015 onward.
    from datetime import datetime
    return jsonify({
        'min_date': '2015-01-01',  # Earliest available data
        'max_date': datetime.now().strftime('%Y-%m-%d'),  # Today's date
        'current_start': data_holder.start_date,
        'current_end': data_holder.end_date,
        'default_start': DEFAULT_START_DATE,
        'default_end': DEFAULT_END_DATE
    })

# Current UI selection (used by notebooks / dashboard)
@app.route('/api/selected-cryptos')
def get_selected_cryptos():
    return jsonify({
        'selected_cryptos': data_holder.selected_cryptos
    })

# Apply selections (validation + filtering). Data collection is notebook-only.
@app.route('/api/fetch-data', methods=['POST'])
def fetch_data():
    try:
        data = request.json
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        crypto_symbols = data.get('crypto_symbols', ["BTC-USD", "ETH-USD"])
        
        if not start_date or not end_date:
            return jsonify({'success': False, 'error': 'Both start_date and end_date are required'}), 400
        
        if not crypto_symbols or len(crypto_symbols) == 0:
            return jsonify({'success': False, 'error': 'At least one cryptocurrency must be selected'}), 400
        
        # Validate that the processed dataset already contains the requested assets.
        # This app does not download data directly; the notebooks handle collection.
        requested_tickers = [sym.split('-')[0] for sym in crypto_symbols]
        missing = [t for t in requested_tickers if f"Close_{t}" not in data_holder.df.columns]
        if missing:
            return (
                jsonify(
                    {
                        'success': False,
                        'error': (
                            "Selected assets not available in the current processed dataset: "
                            + ", ".join(missing)
                            + ". Run notebooks/01_data_collection.ipynb (and your processing steps) "
                            "to include them."
                        ),
                    }
                ),
                400,
            )

        success, message = data_holder.apply_selection(start_date, end_date, crypto_symbols)
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'error': message}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _run_notebook_pipeline(start_date: str, end_date: str, crypto_symbols: list[str]) -> None:
    """Run notebooks 01 -> 02 headlessly using papermill.

    Runs with CWD set to notebooks/ so the existing relative paths (../data/...) keep working.
    """
    try:
        import papermill as pm
    except Exception as e:
        raise RuntimeError(
            "papermill is not installed. Install it with `pip install papermill nbformat`."
        ) from e

    base_dir = Path(__file__).resolve().parent
    notebooks_dir = base_dir / "notebooks"
    results_dir = base_dir / "results" / "papermill"
    results_dir.mkdir(parents=True, exist_ok=True)

    input_01 = notebooks_dir / "01_data_collection.ipynb"
    input_02 = notebooks_dir / "02_data_cleaning.ipynb"
    out_01 = results_dir / "01_data_collection_run.ipynb"
    out_02 = results_dir / "02_data_cleaning_run.ipynb"

    if not input_01.exists() or not input_02.exists():
        raise FileNotFoundError("Expected notebooks/01_data_collection.ipynb and notebooks/02_data_cleaning.ipynb")

    prev_cwd = os.getcwd()
    os.chdir(str(notebooks_dir))
    try:
        # Execute 01 with parameters (papermill injects a parameters cell)
        pm.execute_notebook(
            str(input_01.name),
            str(out_01),
            parameters={
                "start_date": start_date,
                "end_date": end_date,
                "selected_cryptos": crypto_symbols,
            },
        )

        # Execute 02 (it reads whatever raw files 01 produced)
        pm.execute_notebook(
            str(input_02.name),
            str(out_02),
        )
    finally:
        os.chdir(prev_cwd)


@app.route('/api/run-pipeline', methods=['POST'])
def run_pipeline():
    """Trigger 01_data_collection + 02_data_cleaning from the dashboard."""
    if PIPELINE_LOCK.locked():
        return jsonify({'success': False, 'error': 'Pipeline already running. Please wait and try again.'}), 409

    data = request.json or {}
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    crypto_symbols = data.get('crypto_symbols')

    if not start_date or not end_date:
        return jsonify({'success': False, 'error': 'Both start_date and end_date are required'}), 400
    if not crypto_symbols or not isinstance(crypto_symbols, list):
        return jsonify({'success': False, 'error': 'crypto_symbols must be a non-empty list'}), 400

    with PIPELINE_LOCK:
        try:
            _run_notebook_pipeline(start_date, end_date, crypto_symbols)
            # Reload the processed dataset into memory for the running app
            data_holder.load_default()
            return jsonify({'success': True, 'message': 'Pipeline completed'}), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

# Summary cards
@app.route('/api/summary')
def get_summary():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df = data_holder.get_filtered_data(start_date, end_date)
    
    if df.empty:
        return jsonify({'error': 'No data available'}), 400
    
    tickers = _selected_tickers(df)

    assets = {}
    for ticker in tickers:
        close_col = f'Close_{ticker}'
        if close_col not in df.columns:
            continue

        return_col = f'{ticker}_Return'
        if return_col in df.columns:
            avg_ret = float(df[return_col].mean())
        else:
            avg_ret = float(df[close_col].pct_change().mean())

        assets[ticker] = {
            'current': f"${df[close_col].iloc[-1]:,.2f}",
            'high': f"${df[close_col].max():,.2f}",
            'low': f"${df[close_col].min():,.2f}",
            'avg_return': f"{avg_ret * 100:.2f}%",
        }

    return jsonify(
        {
            'total_days': len(df),
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            'sentiment_avg': f"{df['FG_Value'].mean():.1f}",
            'sentiment_current': f"{df['FG_Value'].iloc[-1]:.0f}",
            'tickers': tickers,
            'assets': assets,
        }
    )

# Main time-series payload (prices + sentiment)
@app.route('/api/timeseries')
def get_timeseries():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df = data_holder.get_filtered_data(start_date, end_date)
    
    if df.empty:
        return jsonify({'error': 'No data available'}), 400
    
    tickers = _selected_tickers(df)
    prices = {}
    for ticker in tickers:
        close_col = f'Close_{ticker}'
        if close_col in df.columns:
            prices[ticker] = df[close_col].tolist()

    return jsonify(
        {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'sentiment': df['FG_Value'].tolist(),
            'tickers': tickers,
            'prices': prices,
        }
    )

# Charts below return PNGs (base64) so the frontend can use <img> tags
@app.route('/api/chart/sentiment-distribution')
def sentiment_distribution():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df = data_holder.get_filtered_data(start_date, end_date)
    
    if df.empty:
        return jsonify({'error': 'No data available'}), 400
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
    ax.patch.set_alpha(0)
    sns.histplot(df['FG_Value'], bins=20, kde=True, color='#00c48c', ax=ax, edgecolor='#00c48c', alpha=0.7)
    ax.set_title('Distribution of Fear & Greed Index', fontsize=14, fontweight='bold', color='#e6edf5')
    ax.set_xlabel('Index Value (0-100)', color='#8ea2c2', fontsize=11)
    ax.set_ylabel('Frequency', color='#8ea2c2', fontsize=11)
    ax.tick_params(colors='#8ea2c2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#8ea2c2')
    ax.spines['bottom'].set_color('#8ea2c2')
    ax.grid(True, alpha=0.15, color='#8ea2c2')
    plt.tight_layout()
    plot_url = fig_to_base64(fig)
    return jsonify({'image': plot_url})


def _price_distribution_image(df_: pd.DataFrame, ticker: str) -> str:
    close_col = f'Close_{ticker}'
    _require_ticker_column(df_, close_col, ticker)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
    ax.patch.set_alpha(0)
    sns.histplot(df_[close_col], bins=30, kde=True, color='#00b7ff', ax=ax, edgecolor='#00b7ff', alpha=0.7)
    ax.set_title(f'{ticker} Price Distribution', fontsize=14, fontweight='bold', color='#e6edf5')
    ax.set_xlabel('Price (USD)', color='#8ea2c2', fontsize=11)
    ax.set_ylabel('Frequency', color='#8ea2c2', fontsize=11)
    ax.tick_params(colors='#8ea2c2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#8ea2c2')
    ax.spines['bottom'].set_color('#8ea2c2')
    ax.grid(True, alpha=0.15, color='#8ea2c2')
    plt.tight_layout()
    return fig_to_base64(fig)


@app.route('/api/chart/price-distribution')
def price_distribution():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    ticker = (request.args.get('ticker') or '').upper().strip()

    if not ticker:
        return jsonify({'error': 'ticker query parameter is required'}), 400

    df = data_holder.get_filtered_data(start_date, end_date)

    if df.empty:
        return jsonify({'error': 'No data available'}), 400

    try:
        plot_url = _price_distribution_image(df, ticker)
        return jsonify({'image': plot_url})
    except KeyError as e:
        return jsonify({'error': str(e)}), 400

# Legacy single-asset endpoints (kept for compatibility)
@app.route('/api/chart/btc-distribution')
def btc_distribution():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df = data_holder.get_filtered_data(start_date, end_date)
    
    if df.empty:
        return jsonify({'error': 'No data available'}), 400

    if 'Close_BTC' not in df.columns:
        return jsonify({'error': 'BTC data is not available for the current processed dataset. Include BTC in your Fetch selection or re-run the pipeline with BTC-USD.'}), 400
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
    ax.patch.set_alpha(0)
    sns.histplot(df['Close_BTC'], bins=30, kde=True, color='#00b7ff', ax=ax, edgecolor='#00b7ff', alpha=0.7)
    ax.set_title('Bitcoin Price Distribution', fontsize=14, fontweight='bold', color='#e6edf5')
    ax.set_xlabel('Price (USD)', color='#8ea2c2', fontsize=11)
    ax.set_ylabel('Frequency', color='#8ea2c2', fontsize=11)
    ax.tick_params(colors='#8ea2c2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#8ea2c2')
    ax.spines['bottom'].set_color('#8ea2c2')
    ax.grid(True, alpha=0.15, color='#8ea2c2')
    plt.tight_layout()
    plot_url = fig_to_base64(fig)
    return jsonify({'image': plot_url})

@app.route('/api/chart/eth-distribution')
def eth_distribution():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df = data_holder.get_filtered_data(start_date, end_date)
    
    if df.empty:
        return jsonify({'error': 'No data available'}), 400

    if 'Close_ETH' not in df.columns:
        return jsonify({'error': 'ETH data is not available for the current processed dataset. Include ETH in your Fetch selection or re-run the pipeline with ETH-USD.'}), 400
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
    ax.patch.set_alpha(0)
    sns.histplot(df['Close_ETH'], bins=30, kde=True, color='#00c48c', ax=ax, edgecolor='#00c48c', alpha=0.7)
    ax.set_title('Ethereum Price Distribution', fontsize=14, fontweight='bold', color='#e6edf5')
    ax.set_xlabel('Price (USD)', color='#8ea2c2', fontsize=11)
    ax.set_ylabel('Frequency', color='#8ea2c2', fontsize=11)
    ax.tick_params(colors='#8ea2c2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#8ea2c2')
    ax.spines['bottom'].set_color('#8ea2c2')
    ax.grid(True, alpha=0.15, color='#8ea2c2')
    plt.tight_layout()
    plot_url = fig_to_base64(fig)
    return jsonify({'image': plot_url})

# Returns vs sentiment scatter
@app.route('/api/chart/returns-vs-sentiment')
def returns_vs_sentiment():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df = data_holder.get_filtered_data(start_date, end_date)
    
    if df.empty:
        return jsonify({'error': 'No data available'}), 400

    if 'FG_Value' not in df.columns:
        return jsonify({'error': 'FG_Value column is missing from the current dataset.'}), 400

    tickers = _selected_tickers(df)
    if len(tickers) == 0:
        return jsonify({'error': 'No assets available'}), 400

    n = len(tickers)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), facecolor='none')
    axes_flat = np.array(axes).reshape(-1)

    colors = ['#00b7ff', '#00c48c', '#ff7f0e', '#f45b69', '#8ea2c2', '#7b61ff', '#00c48c', '#00b7ff']

    for i, ticker in enumerate(tickers):
        ax = axes_flat[i]
        ax.patch.set_alpha(0)

        return_col = f'{ticker}_Return'
        close_col = f'Close_{ticker}'
        if return_col in df.columns:
            r = df[return_col] * 100
        elif close_col in df.columns:
            r = df[close_col].pct_change() * 100
        else:
            ax.axis('off')
            continue

        ax.scatter(df['FG_Value'], r, alpha=0.6, s=25, color=colors[i % len(colors)], edgecolors=colors[i % len(colors)], linewidth=0.4)
        ax.set_title(f'{ticker} Returns vs Sentiment', fontsize=12, fontweight='bold', color='#e6edf5')
        ax.set_xlabel('Fear & Greed Index', color='#8ea2c2', fontsize=11)
        ax.set_ylabel('Daily Return (%)', color='#8ea2c2', fontsize=11)
        ax.tick_params(colors='#8ea2c2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#8ea2c2')
        ax.spines['bottom'].set_color('#8ea2c2')
        ax.grid(True, alpha=0.15, color='#8ea2c2')

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plot_url = fig_to_base64(fig)
    return jsonify({'image': plot_url})

# Rolling volatility
@app.route('/api/chart/volatility')
def volatility():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df = data_holder.get_filtered_data(start_date, end_date)
    
    if df.empty:
        return jsonify({'error': 'No data available'}), 400

    tickers = _selected_tickers(df)
    if len(tickers) == 0:
        return jsonify({'error': 'No assets available'}), 400
    
    fig, ax = plt.subplots(figsize=(14, 5), facecolor='none')
    ax.patch.set_alpha(0)
    colors = ['#00b7ff', '#00c48c', '#ff7f0e', '#f45b69', '#8ea2c2', '#7b61ff', '#00c48c', '#00b7ff']
    for i, ticker in enumerate(tickers):
        vol_col = f'{ticker}_Vol30'
        close_col = f'Close_{ticker}'
        if vol_col in df.columns:
            vol = df[vol_col] * 100
        elif close_col in df.columns:
            vol = df[close_col].pct_change().rolling(30).std() * 100
        else:
            continue
        ax.plot(df.index, vol, label=f'{ticker} 30-Day Volatility', color=colors[i % len(colors)], linewidth=2.5, alpha=0.9)
    ax.set_title('Rolling 30-Day Volatility', fontsize=14, fontweight='bold', color='#e6edf5')
    ax.set_xlabel('Date', color='#8ea2c2', fontsize=11)
    ax.set_ylabel('Volatility (%)', color='#8ea2c2', fontsize=11)
    ax.tick_params(colors='#8ea2c2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#8ea2c2')
    ax.spines['bottom'].set_color('#8ea2c2')
    legend = ax.legend(facecolor='#161f30', edgecolor='#8ea2c2', framealpha=0.9)
    for text in legend.get_texts():
        text.set_color('#e6edf5')
    ax.grid(True, alpha=0.15, color='#8ea2c2')
    plt.tight_layout()
    plot_url = fig_to_base64(fig)
    return jsonify({'image': plot_url})

# Correlation matrix
@app.route('/api/chart/correlation')
def correlation():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df = data_holder.get_filtered_data(start_date, end_date)
    
    if df.empty:
        return jsonify({'error': 'No data available'}), 400
    
    tickers = _selected_tickers(df)
    corr_cols = []
    for t in tickers:
        for c in [f'Close_{t}', f'{t}_Return', f'{t}_Vol30']:
            if c in df.columns:
                corr_cols.append(c)
    if 'FG_Value' in df.columns:
        corr_cols.append('FG_Value')

    if len(corr_cols) < 2:
        return jsonify({'error': 'Not enough numeric columns to compute correlation'}), 400

    corr_matrix = df[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='none')
    ax.patch.set_alpha(0)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax, 
                cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1, linewidths=0.5, linecolor='#8ea2c2',
                annot_kws={'color': '#0c0f16', 'fontsize': 10, 'fontweight': 'bold'})
    ax.set_title('Correlation Matrix: Prices, Returns, Volatility & Sentiment', fontsize=12, fontweight='bold', color='#e6edf5')
    ax.tick_params(colors='#8ea2c2')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='#8ea2c2')
    cbar.set_label('Correlation', color='#8ea2c2')
    plt.tight_layout()
    plot_url = fig_to_base64(fig)
    return jsonify({'image': plot_url})

# Sentiment regime breakdown
@app.route('/api/chart/sentiment-regimes')
def sentiment_regimes():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df = data_holder.get_filtered_data(start_date, end_date)
    
    if df.empty:
        return jsonify({'error': 'No data available'}), 400
    
    # Count sentiment classifications
    value_counts = df['value_classification'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='none')
    ax.patch.set_alpha(0)
    colors = {'Fear': '#f45b69', 'Greed': '#00c48c'}
    color_list = [colors.get(label, '#8ea2c2') for label in value_counts.index]
    value_counts.plot(kind='bar', ax=ax, color=color_list, alpha=0.85, edgecolor='#8ea2c2', linewidth=1.5)
    ax.set_title('Market Sentiment Regime Distribution', fontsize=14, fontweight='bold', color='#e6edf5')
    ax.set_xlabel('Sentiment Classification', color='#8ea2c2', fontsize=11)
    ax.set_ylabel('Days Count', color='#8ea2c2', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.tick_params(colors='#8ea2c2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#8ea2c2')
    ax.spines['bottom'].set_color('#8ea2c2')
    ax.grid(True, alpha=0.15, color='#8ea2c2', axis='y')
    plt.tight_layout()
    plot_url = fig_to_base64(fig)
    return jsonify({'image': plot_url})

# Data table payload
@app.route('/api/table/latest')
def latest_data():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    limit = request.args.get('limit', '10')
    
    df = data_holder.get_filtered_data(start_date, end_date)
    
    if df.empty:
        return jsonify({'tickers': [], 'rows': []})
    
    # Limit controls how many rows we return to the table
    if limit == 'all':
        latest = df.copy()
    else:
        try:
            limit_int = int(limit)
            latest = df.tail(limit_int).copy()
        except ValueError:
            latest = df.tail(10).copy()
    
    tickers = _selected_tickers(latest)

    latest = latest.copy()
    latest['Date'] = latest.index.strftime('%Y-%m-%d')

    rows = []
    for _, row in latest.iterrows():
        prices_out = {}
        returns_out = {}

        for t in tickers:
            close_col = f'Close_{t}'
            if close_col in latest.columns:
                prices_out[t] = f"${float(row[close_col]):,.2f}"

            ret_col = f'{t}_Return'
            if ret_col in latest.columns:
                returns_out[t] = f"{float(row[ret_col]) * 100:.2f}%"

        rows.append(
            {
                'date': row['Date'],
                'prices': prices_out,
                'returns': returns_out,
                'sentiment': f"{float(row['FG_Value']):.0f}" if 'FG_Value' in latest.columns else '',
                'classification': row['value_classification'] if 'value_classification' in latest.columns else '',
            }
        )

    return jsonify({'tickers': tickers, 'rows': rows})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
