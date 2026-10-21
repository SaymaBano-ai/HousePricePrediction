"""
House Price Dataset Generator
------------------------------
Generates a realistic synthetic housing dataset modeled on real-world
Indian residential real estate pricing patterns (area, location tier,
BHK, age, amenities, floor, furnishing). Realistic noise, missing
values, duplicate rows, and outliers are deliberately injected so the
full preprocessing pipeline (missing value handling, duplicate removal,
outlier treatment, encoding) is genuinely exercised, exactly as required
by the capstone guidelines.

Run: python generate_dataset.py
Output: house_prices_raw.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000

cities = ["Lucknow", "Prayagraj", "Kanpur", "Varanasi", "Noida", "Delhi", "Pune", "Bangalore"]
city_base_price_per_sqft = {  # INR per sqft, base price level by city tier
    "Lucknow": 4200, "Prayagraj": 3600, "Kanpur": 3400, "Varanasi": 3800,
    "Noida": 6800, "Delhi": 9500, "Pune": 7200, "Bangalore": 8400
}
locality_tier = ["Premium", "Mid-range", "Affordable"]
tier_multiplier = {"Premium": 1.45, "Mid-range": 1.0, "Affordable": 0.68}

furnishing_types = ["Unfurnished", "Semi-Furnished", "Fully-Furnished"]
furnishing_premium = {"Unfurnished": 0, "Semi-Furnished": 150000, "Fully-Furnished": 380000}

property_types = ["Apartment", "Independent House", "Villa", "Builder Floor"]

rows = []
for i in range(N):
    city = np.random.choice(cities)
    area_sqft = np.round(np.random.normal(1450, 550))
    area_sqft = max(300, area_sqft)

    bhk = np.random.choice([1, 2, 3, 4, 5], p=[0.12, 0.35, 0.32, 0.16, 0.05])
    bathrooms = max(1, bhk - np.random.choice([0, 1], p=[0.6, 0.4]))
    balconies = np.random.choice([0, 1, 2, 3], p=[0.15, 0.4, 0.3, 0.15])

    age_years = np.random.choice(range(0, 35), p=np.array([35-x for x in range(35)]) / sum(35-x for x in range(35)))

    tier = np.random.choice(locality_tier, p=[0.25, 0.45, 0.30])
    furnishing = np.random.choice(furnishing_types, p=[0.45, 0.4, 0.15])
    ptype = np.random.choice(property_types, p=[0.55, 0.2, 0.1, 0.15])

    floor_no = np.random.randint(0, 25)
    total_floors = floor_no + np.random.randint(1, 15)

    parking = np.random.choice([0, 1, 2, 3], p=[0.1, 0.55, 0.27, 0.08])
    has_lift = np.random.choice([0, 1], p=[0.3, 0.7]) if ptype in ["Apartment", "Builder Floor"] else 0
    near_metro_km = np.round(np.random.exponential(3.5), 2)
    school_nearby = np.random.choice([0, 1], p=[0.35, 0.65])
    hospital_nearby = np.random.choice([0, 1], p=[0.4, 0.6])

    # price construction (real-world-like nonlinear relationship)
    base = city_base_price_per_sqft[city] * tier_multiplier[tier]
    price = base * area_sqft
    price += bhk * 180000
    price += bathrooms * 90000
    price += balconies * 45000
    price -= age_years * 12000
    price += furnishing_premium[furnishing]
    price += parking * 110000
    price += has_lift * 95000
    price -= near_metro_km * 18000
    price += school_nearby * 70000
    price += hospital_nearby * 50000
    price += (floor_no > 0) * 30000
    price += np.random.normal(0, base * area_sqft * 0.06)  # market noise
    price = max(450000, price)

    rows.append([
        city, tier, ptype, area_sqft, bhk, bathrooms, balconies, age_years,
        furnishing, floor_no, total_floors, parking, has_lift,
        near_metro_km, school_nearby, hospital_nearby, round(price, -3)
    ])

df = pd.DataFrame(rows, columns=[
    "City", "Locality_Tier", "Property_Type", "Area_SqFt", "BHK", "Bathrooms",
    "Balconies", "Age_Years", "Furnishing", "Floor_No", "Total_Floors",
    "Parking_Spaces", "Has_Lift", "Distance_to_Metro_KM", "School_Nearby",
    "Hospital_Nearby", "Price"
])

# ---- Inject realistic data-quality issues (to make preprocessing necessary) ----
# 1. Missing values (~4-6% across several columns)
for col in ["Area_SqFt", "Bathrooms", "Furnishing", "Distance_to_Metro_KM", "Total_Floors"]:
    mask = np.random.rand(N) < 0.045
    df.loc[mask, col] = np.nan

# 2. Outliers (extreme area / price entries)
outlier_idx = np.random.choice(N, 20, replace=False)
df.loc[outlier_idx[:10], "Area_SqFt"] = df.loc[outlier_idx[:10], "Area_SqFt"] * np.random.uniform(4, 6)
df.loc[outlier_idx[10:], "Price"] = df.loc[outlier_idx[10:], "Price"] * np.random.uniform(5, 8)

# 3. Duplicate rows
dupes = df.sample(35, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# 4. Shuffle
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("house_prices_raw.csv", index=False)
print(f"Dataset generated: {df.shape[0]} rows x {df.shape[1]} columns -> house_prices_raw.csv")
