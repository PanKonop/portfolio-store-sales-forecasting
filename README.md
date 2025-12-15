# Store Sales - Time Series Forecasting 📈

## 📌 Project Overview
This project aims to predict daily sales for thousands of products across varying stores in Ecuador. The solution handles complex time-series challenges such as seasonality, holidays, and external economic factors (e.g., oil prices).

**Metric:** Root Mean Squared Logarithmic Error (RMSLE)
**Result:** ~0.46 (Top tier for non-ensemble models)

## 🔧 Technologies Used
* **Python 3.10+**
* **pandas & numpy:** Data manipulation and time-series grid generation.
* **XGBoost:** Gradient Boosting for regression tasks.
* **scikit-learn:** Feature encoding and metrics.

## 🧠 Methodology (CRISP-DM)
1.  **Data Cleaning:** Handled missing dates (e.g., Christmas) using Cartesian Product to ensure continuous time series.
2.  **Feature Engineering:**
    * **Time Features:** Day of week, Month, Wage Day (15th/End of month).
    * **Lag Features:** Generated safe lags (Lag 16, Lag 21) to avoid data leakage during the 16-day test horizon.
    * **Rolling Statistics:** Moving averages and standard deviation to capture local trends.
3.  **Modeling:** Used XGBoost Regressor with a log-transformed target (`log1p`) to optimize for RMSLE.

## 📂 Project Structure
```text
Store_Sales/
├── data/                  # Raw CSV files (not included in repo)
├── notebooks/             # EDA and experiments
├── src/                   # Source code
│   ├── data_loader.py     # Data ingestion & grid creation
│   ├── features.py        # Feature engineering logic
│   └── model.py           # Model definition (optional)
├── main.py                # Executive script
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

🚀 How to Run
Clone the repository.

Install dependencies:

Bash

pip install -r requirements.txt
Place Kaggle data files (train.csv, test.csv) in the data/ folder.

Run the pipeline:

Bash

python main.py
The output file submission_pipeline.csv will be generated in the data/ folder.
test