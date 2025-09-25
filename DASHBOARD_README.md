# 🛢️ Oil Trading Dashboard Guide

## 📋 **Differences Between main.py and test_2024_2025_performance.py**

### **main.py** (Production Pipeline)
- **Purpose**: Full end-to-end production pipeline for oil price forecasting
- **Train/Test Split**: Uses default 80/20 split on entire dataset
- **Time Period**: Trains on historical data, tests on most recent period
- **Features**:
  - Complete data pipeline with weather data integration
  - Configurable model hyperparameters
  - Grid search optimization
  - Comprehensive backtesting
  - Production-ready output files

### **test_2024_2025_performance.py** (Specific Evaluation)
- **Purpose**: Specialized script to evaluate model performance on 2024-2025 data specifically
- **Train/Test Split**: Custom split - trains on pre-2024 data, tests on 2024-2025
- **Time Period**: Specifically designed to test recent performance (out-of-sample)
- **Features**:
  - Fixed train/test dates for consistent evaluation
  - Feature selection optimization
  - Focused on recent market conditions
  - Detailed temporal validation
  - 2024-2025 specific performance analysis

| Aspect | main.py | test_2024_2025_performance.py |
|--------|---------|--------------------------------|
| **Use Case** | Production trading | Recent performance evaluation |
| **Data Split** | 80/20 rolling | Fixed 2024+ test period |
| **Optimization** | Full grid search | Fast evaluation |
| **Weather Data** | Optional inclusion | Excluded for speed |
| **Output Focus** | General performance | 2024-2025 specific metrics |

---

## 🚀 **How to Use the Interactive Dashboard**

### **Step 1: Install Required Dependencies**

```bash
# Install Streamlit and Plotly
pip install streamlit plotly

# Or add to requirements.txt:
echo "streamlit>=1.28.0" >> requirements.txt
echo "plotly>=5.17.0" >> requirements.txt
pip install -r requirements.txt
```

### **Step 2: Generate Trading Data**

First, run the test script to generate the trading results:

```bash
# Activate your virtual environment
source venv/bin/activate

# Run the 2024-2025 performance test
python test_2024_2025_performance.py
```

This creates the following files in `outputs/2024_2025_test/`:
- `predictions_2024_2025.csv` - Model predictions vs actual prices
- `signals_2024_2025.csv` - Buy/sell signals and equity curve
- `test_results_summary.json` - Performance metrics

### **Step 3: Launch the Dashboard**

```bash
# Launch Streamlit dashboard
streamlit run streamlit_trading_dashboard.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

---

## 📊 **Dashboard Features**

### **1. Performance Summary**
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Worst peak-to-trough decline
- **Directional Accuracy**: % of correct predictions
- **Trade Count**: Total number of trades executed

### **2. Interactive Price Chart**
- **Oil price line chart** with full historical data
- **🟢 Green triangles**: Buy signals with exact entry prices
- **🔴 Red triangles**: Sell signals with exact exit prices
- **Hover details**: Exact dates and prices for each trade
- **Zoom/Pan**: Interactive exploration of any time period

### **3. Equity Curve**
- **Portfolio value progression** over time
- **Visual drawdown periods** 
- **Performance tracking** relative to starting capital

### **4. Daily P&L Chart**
- **Green bars**: Profitable days
- **Red bars**: Loss days
- **Daily profit/loss** breakdown

### **5. Trade Analysis Table**
- **Individual trade details**:
  - Entry/Exit dates and prices
  - Trade duration
  - Profit/Loss per trade
  - Return percentage
- **Color-coded profitability**:
  - 🟢 Green background: Profitable trades
  - 🔴 Red background: Losing trades

### **6. Trading Statistics**
- **Win Rate**: Percentage of profitable trades
- **Average P&L**: Mean profit/loss per trade
- **Current Position**: Current market exposure
- **Trading Frequency**: Activity metrics

---

## 🔧 **Customization Options**

### **Date Range Filtering**
Use the sidebar date pickers to focus on specific periods:
- **Start Date**: Beginning of analysis period
- **End Date**: End of analysis period
- **Real-time updates**: All charts and metrics update automatically

### **Performance Period Analysis**
```bash
# Generate data for different periods
python test_2024_2025_performance.py --train-end 2022-12-31 --test-start 2023-01-01

# Or custom date ranges
python test_2024_2025_performance.py --train-end 2023-06-30 --test-start 2023-07-01
```

---

## 📈 **Understanding the Metrics**

### **Trading Performance**
- **Total Return**: `(Final Value - Initial Value) / Initial Value`
- **Sharpe Ratio**: `(Return - Risk-free Rate) / Volatility`
- **Calmar Ratio**: `Annualized Return / Max Drawdown`
- **Max Drawdown**: Maximum peak-to-trough portfolio decline

### **Prediction Accuracy**
- **Directional Accuracy**: % of days where predicted direction matched actual
- **Correlation**: Linear relationship between predictions and actual returns
- **RMSE**: Root Mean Square Error of return predictions

### **Trade Analysis**
- **Win Rate**: `Winning Trades / Total Trades`
- **Average Trade P&L**: Mean profit/loss per completed trade
- **Trade Duration**: Average holding period per position

---

## 🚨 **Important Notes**

### **Data Requirements**
- Dashboard loads cached results from `outputs/2024_2025_test/`
- **Must run test script first** before launching dashboard
- Data covers the period specified in the test run

### **Performance Interpretation**
- **Backtest Results**: Historical simulation, not live trading
- **Transaction Costs**: Included in P&L calculations
- **Slippage**: Not explicitly modeled (uses close prices)
- **Realistic Expectations**: Live trading results may differ

### **Technical Considerations**
- **Data Freshness**: Ensure GPR and EPU data are current for live trading
- **Model Retraining**: Consider periodic model updates
- **Risk Management**: Implement proper position sizing and stop-losses

---

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **"No cached results found"**
   ```bash
   # Solution: Run the test script first
   python test_2024_2025_performance.py
   ```

2. **Streamlit not found**
   ```bash
   # Solution: Install Streamlit
   pip install streamlit
   ```

3. **Empty dashboard**
   ```bash
   # Check if output files exist
   ls -la outputs/2024_2025_test/
   ```

4. **Port already in use**
   ```bash
   # Use different port
   streamlit run streamlit_trading_dashboard.py --server.port 8502
   ```

---

## 🎯 **Next Steps for Live Trading**

1. **Set up real-time data feeds** for GPR and EPU indices
2. **Implement broker API integration** for automated trading
3. **Add risk management rules** (stop-losses, position limits)
4. **Monitor model performance** and implement retraining schedule
5. **Paper trade first** to validate execution logic

The dashboard provides the foundation for understanding your strategy's performance and can be extended for live trading monitoring once you implement real-time data feeds.
