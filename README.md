# House Price Prediction System

**AIML Summer Internship 2026 — Capstone Project**
IIHMF, MNNIT Allahabad, Prayagraj

## Objective
Predict residential house prices based on area, location, number of rooms, age of property, and available amenities, using a complete machine learning lifecycle from data preprocessing to deployment.

## Folder Structure
```
HousePricePrediction/
│
├── Dataset/
│   ├── generate_dataset.py        # Synthetic, realistic dataset generator
│   ├── house_prices_raw.csv       # Raw dataset (3000+ records)
│   └── house_prices_cleaned.csv   # Cleaned dataset after preprocessing
│
├── Notebook/
│   ├── house_price_pipeline.py    # Full pipeline script (preprocessing -> evaluation)
│   └── house_price_pipeline.ipynb # Same pipeline as a Jupyter notebook
│
├── Model/
│   ├── best_house_price_model.pkl # Best performing trained model (deployed)
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── model_metadata.pkl
│
├── Streamlit_App/
│   └── app.py                     # Deployment app (Streamlit)
│
├── Documentation/
│   ├── eda_plots/                 # All EDA and evaluation charts
│   ├── model_comparison.csv       # Model performance comparison table
│   └── Project_Report.docx        # Full project report (15-25 pages)
│
└── README.md
```

## Dataset
A realistic synthetic dataset (3,000+ records) was generated to model Indian residential real estate pricing patterns (city tier, locality tier, BHK, area, age, amenities, furnishing, etc.), since this best supports the offline/sandboxed environment used to build this submission. The generation logic and pricing relationships are documented in `Dataset/generate_dataset.py` and explained in Chapter 3 of the project report. The dataset deliberately contains missing values, duplicate rows, and outliers so that every preprocessing step in the methodology is genuinely exercised.

**To use a real Kaggle dataset instead** (e.g. "House Price Prediction Dataset" or "Bengaluru House Price Data" on Kaggle), simply replace `Dataset/house_prices_raw.csv` with the downloaded CSV (matching column names, or adjusting the column list at the top of `house_price_pipeline.py`) and re-run the pipeline — the rest of the project works unchanged.

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit joblib
```

### 2. Generate the dataset
```bash
cd Dataset
python generate_dataset.py
```

### 3. Run the full ML pipeline (preprocessing, EDA, training, evaluation)
```bash
cd ../Notebook
python house_price_pipeline.py
```
This produces:
- Cleaned dataset in `Dataset/house_prices_cleaned.csv`
- EDA plots and model comparison charts in `Documentation/eda_plots/`
- Trained model files in `Model/`

### 4. Launch the Streamlit app
```bash
cd ../Streamlit_App
streamlit run app.py
```

## Models Trained
| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | ~13.9 Lakh | ~19.9 Lakh | 0.826 |
| Random Forest | ~7.1 Lakh | ~13.1 Lakh | 0.924 |
| XGBoost (Best) | ~7.2 Lakh | ~13.1 Lakh | **0.924** |

The best model (XGBoost) is automatically selected and saved as `best_house_price_model.pkl`, which powers the Streamlit app.

## Academic Integrity
This project was built following the AIML Summer Internship guidelines. AI tools were used to scaffold the pipeline and documentation structure; students using this as a reference must understand, adapt, and be able to explain every line of code before submission, per the program's academic integrity policy.
