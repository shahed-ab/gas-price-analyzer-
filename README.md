# Natural Gas Price Analyzer & Forecasting Tool

A Python-based financial data analysis tool designed to examine historical natural gas price trends and forecast future prices using linear regression and seasonal pattern decomposition.

---

## Features

- **Historical Trend Analysis**: Evaluates long-term price direction using linear regression modeling.
- **Seasonality Decomposition**: Identifies recurring cyclic and seasonal behavior in natural gas prices.
- **Data Visualization**: Generates analytical plots using Matplotlib to visualize historical data and predicted trajectories.
- **CSV Data Integration**: Directly ingests tabular price and date records from CSV format.

---

## Project Structure

```text
gas-price-analyzer/
│
├── gas_price_analyzer.py   # Core analysis, forecasting logic, and visualization
├── Nat_Gas.csv             # Historical natural gas prices dataset
├── README.md               # Project documentation
└── .gitignore              # Files and directories ignored by Git
