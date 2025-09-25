## OilForecast: Quantitative Oil Price Forecasting Platform

This repository contains a full pipeline for ingesting market, macro, geopolitical, and weather data to forecast oil price returns, generate trading signals, size positions, backtest performance, and visualize results via Streamlit.

### Key Components
- Data ingestion: Yahoo Finance market data, enhanced GPR features, EPU, BDI, and weather (Open-Meteo)
- Feature engineering: Stationary macro features including the target (`wti_price_logret`)
- Modeling: Ensemble ML models and GARCH-based volatility overlay
- Signals and sizing: Threshold-based signal generation and volatility-scaled position sizing
- Backtesting: Portfolio equity curve and metrics
- Visualization: Streamlit dashboard showing price, signals, equity, P&L, and trade stats

### Project Layout
- `quant_oil_forecast/` core library (data_ingestion, features, models, signals, backtest, utils)
- `main.py` production pipeline (optional weather via `--with-weather`)
- `test_2024_2025_performance.py` fixed 2024–2025 evaluation (weather included)
- `outputs/` generated artifacts and reports
- `datasets/` required input files (GPR, EPU, BDI)

### Requirements
Install with your preferred virtual environment.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Environment variables (in `.env`) must define API keys used by the pipeline:
- `FRED_API_KEY`
- `EIA_API_KEY`

### Datasets
Place these files in `datasets/` (already referenced by absolute paths in `quant_oil_forecast/config/settings.py`):
- `data_gpr_daily_recent.xls`
- `data_gpr_export.xls`
- `USEPUINDXD.csv`
- `BalticDryIndexHistoricalData.csv`

### Run the Production Pipeline
```bash
python main.py --with-weather
```
Artifacts saved under `outputs/`:
- `model_metrics.csv`, `ensemble_predictions.csv`, `signals.csv`, `positions.csv`
- `equity_curve.png`, `pred_vs_actual.png`, `temporal_validation_report.txt`

### Run the 2024–2025 Evaluation (Weather Included)
```bash
python test_2024_2025_performance.py
```
Saves to `outputs/2024_2025_test/`:
- `predictions_2024_2025.csv`, `signals_2024_2025.csv`, `equity_curve_2024_2025.csv`
- `test_results_summary.json`, `temporal_validation_report.txt`, `2024_2025_performance_analysis.png`

### Launch the Streamlit Dashboard
Generate test outputs first (above), then launch:

```bash
python run_dashboard.py
# or directly
streamlit run simple_dashboard.py
```

The dashboard displays:
- Oil price with buy/sell markers (always renders, even if no signals in range)
- Equity curve and daily P&L
- Trade analysis table, win rate, and quick stats

### Weather Features
Weather is integrated using Open-Meteo for cities defined in `settings.CITIES_TO_FETCH`. The production pipeline enables weather via `--with-weather`. The 2024–2025 evaluation includes weather by default with robust fallbacks if the API is unavailable.

### Troubleshooting
- Missing outputs: run the evaluation first (`python test_2024_2025_performance.py`)
- Streamlit not installed: `pip install streamlit plotly`
- Port in use: `streamlit run simple_dashboard.py --server.port 8502`

### License
Proprietary – internal research use only.


