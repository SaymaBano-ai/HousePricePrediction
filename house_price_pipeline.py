"""
House Price Prediction - Full ML Pipeline
===========================================
Implements Phases 2-7 of the project methodology:
 Phase 2: Data Collection
 Phase 3: Data Preprocessing
 Phase 4: Exploratory Data Analysis (EDA)
 Phase 5: Feature Engineering
 Phase 6: Model Building (Linear Regression, Random Forest, XGBoost)
 Phase 7: Model Evaluation (MAE, MSE, RMSE, R2)

Run from the Notebook/ folder: python house_price_pipeline.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sns.set_style("whitegrid")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "..", "Dataset", "house_prices_raw.csv")
EDA_DIR = os.path.join(BASE, "..", "Documentation", "eda_plots")
MODEL_DIR = os.path.join(BASE, "..", "Model")
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# PHASE 2: DATA COLLECTION
# ============================================================
df = pd.read_csv(DATA_PATH)
print("Raw dataset shape:", df.shape)
print(df.info())

# ============================================================
# PHASE 3: DATA PREPROCESSING
# ============================================================

# 3.1 Remove duplicates
before = df.shape[0]
df = df.drop_duplicates()
print(f"Removed {before - df.shape[0]} duplicate rows")

# 3.2 Handle missing values
print("\nMissing values before imputation:\n", df.isnull().sum()[df.isnull().sum() > 0])

num_cols_missing = ["Area_SqFt", "Bathrooms", "Distance_to_Metro_KM", "Total_Floors"]
for col in num_cols_missing:
    df[col] = df[col].fillna(df[col].median())

df["Furnishing"] = df["Furnishing"].fillna(df["Furnishing"].mode()[0])

print("\nMissing values after imputation:", df.isnull().sum().sum())

# 3.3 Outlier treatment (IQR capping) on Area_SqFt and Price
def cap_outliers_iqr(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower, upper)

for col in ["Area_SqFt", "Price"]:
    df[col] = cap_outliers_iqr(df[col])

# 3.4 Encode categorical features (Phase 5 will do model-ready encoding via pipeline,
# but we create a couple of explicit engineered/encoded columns here for EDA clarity)
df["Locality_Tier"] = pd.Categorical(df["Locality_Tier"], categories=["Affordable", "Mid-range", "Premium"], ordered=True)
df["Tier_Rank"] = df["Locality_Tier"].cat.codes

print("\nCleaned dataset shape:", df.shape)
df.to_csv(os.path.join(BASE, "..", "Dataset", "house_prices_cleaned.csv"), index=False)

# ============================================================
# PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Univariate: Price distribution
plt.figure(figsize=(7, 5))
sns.histplot(df["Price"], bins=40, kde=True, color="#4C72B0")
plt.title("Price Distribution (Univariate Analysis)")
plt.xlabel("Price (INR)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "01_price_distribution.png"), dpi=120)
plt.close()

# Univariate: Area distribution
plt.figure(figsize=(7, 5))
sns.histplot(df["Area_SqFt"], bins=40, kde=True, color="#55A868")
plt.title("Area (SqFt) Distribution")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "02_area_distribution.png"), dpi=120)
plt.close()

# Bivariate: Area vs Price
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="Area_SqFt", y="Price", hue="Locality_Tier", alpha=0.5)
plt.title("Area vs Price by Locality Tier (Bivariate Analysis)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "03_area_vs_price.png"), dpi=120)
plt.close()

# Bivariate: BHK vs Price boxplot
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="BHK", y="Price")
plt.title("Price by Number of BHK (Boxplot)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "04_bhk_vs_price_boxplot.png"), dpi=120)
plt.close()

# City-wise boxplot
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="City", y="Price")
plt.xticks(rotation=30)
plt.title("Price Distribution by City")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "05_city_vs_price_boxplot.png"), dpi=120)
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "06_correlation_heatmap.png"), dpi=120)
plt.close()

print("\nEDA plots saved to:", EDA_DIR)
print("\nTop correlations with Price:\n", corr["Price"].sort_values(ascending=False))

# ============================================================
# PHASE 5: FEATURE ENGINEERING
# ============================================================

# New meaningful features
df["Price_per_SqFt"] = df["Price"] / df["Area_SqFt"]  # used only for EDA reference, dropped before modeling
df["Floor_Ratio"] = df["Floor_No"] / df["Total_Floors"].replace(0, 1)
df["Rooms_per_Bathroom"] = df["BHK"] / df["Bathrooms"].replace(0, 1)
df["Is_New_Property"] = (df["Age_Years"] <= 2).astype(int)
df["Amenity_Score"] = df["School_Nearby"] + df["Hospital_Nearby"] + df["Has_Lift"] + (df["Parking_Spaces"] > 0).astype(int)

# Feature selection: drop leakage / redundant columns before modeling
model_df = df.drop(columns=["Price_per_SqFt", "Locality_Tier"])  # Locality_Tier kept as Tier_Rank (already encoded)

target = "Price"
X = model_df.drop(columns=[target])
y = model_df[target]

categorical_features = ["City", "Property_Type", "Furnishing"]
numeric_features = [c for c in X.columns if c not in categorical_features]

print("\nFinal feature set:", list(X.columns))

# ============================================================
# PHASE 6: MODEL BUILDING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.85,
                             colsample_bytree=0.85, random_state=42, n_jobs=-1)
}

results = []
trained_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2_Score": r2})
    trained_pipelines[name] = pipe
    print(f"\n{name}: MAE={mae:,.0f} | RMSE={rmse:,.0f} | R2={r2:.4f}")

# ============================================================
# PHASE 7: MODEL EVALUATION
# ============================================================

results_df = pd.DataFrame(results).sort_values("R2_Score", ascending=False).reset_index(drop=True)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(BASE, "..", "Documentation", "model_comparison.csv"), index=False)

# Comparison bar chart
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Model", y="R2_Score", palette="viridis")
plt.title("Model Comparison - R2 Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "07_model_comparison_r2.png"), dpi=120)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Model", y="RMSE", palette="magma")
plt.title("Model Comparison - RMSE")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "08_model_comparison_rmse.png"), dpi=120)
plt.close()

best_model_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]
print(f"\nBest model: {best_model_name}")

# Predicted vs Actual plot for best model
preds_best = best_pipeline.predict(X_test)
plt.figure(figsize=(7, 7))
plt.scatter(y_test, preds_best, alpha=0.4, color="#C44E52")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted Price - {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "09_actual_vs_predicted.png"), dpi=120)
plt.close()

# Feature importance (for tree-based best model)
if best_model_name in ["Random Forest", "XGBoost"]:
    ohe_cols = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(ohe_cols)
    importances = best_pipeline.named_steps["regressor"].feature_importances_
    fi_df = pd.DataFrame({"Feature": all_feature_names, "Importance": importances}).sort_values("Importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="crest")
    plt.title(f"Top 15 Feature Importances - {best_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "10_feature_importance.png"), dpi=120)
    plt.close()

# Save best model + metadata
joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_house_price_model.pkl"))
joblib.dump({"numeric_features": numeric_features, "categorical_features": categorical_features,
             "best_model_name": best_model_name}, os.path.join(MODEL_DIR, "model_metadata.pkl"))

# also save all models for reference
for name, pipe in trained_pipelines.items():
    fname = name.lower().replace(" ", "_") + "_model.pkl"
    joblib.dump(pipe, os.path.join(MODEL_DIR, fname))

print("\nAll models saved to Model/ directory.")
print("Pipeline complete.")
