# PV EDA & Prediction

Solar panel data analysis and time series forecasting using LSTM and Prophet models.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Python 3.12+

## Setup

```bash
# Install dependencies
uv sync

# Start Jupyter Lab
uv run jupyter lab
```

## Project Structure

```
├── data/
│   ├── pv_data.csv          # Solar panel production data
│   └── weather.csv          # Weather data
├── eda.ipynb                # Exploratory data analysis
├── forecasting.ipynb        # Time series forecasting models
├── lstm_model.py            # LSTM model implementation
├── exports/
│   ├── eda.html             # Exported EDA notebook
│   └── forecasting.html     # Exported forecasting notebook
└── pyproject.toml           # Project dependencies
```

## Usage

Open and run the notebooks in order:
1. `eda.ipynb` - Data exploration and visualization
2. `forecasting.ipynb` - LSTM and Prophet forecasting models

Exported HTML versions are available in `exports/` for quick viewing without running the notebooks.
