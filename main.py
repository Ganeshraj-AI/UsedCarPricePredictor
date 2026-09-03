# ==============================================================================
# USED CAR PRICE PREDICTOR — LEARNING-FIRST ML PROJECT
# Algorithm: Random Forest Regressor
# Goal: Understand tabular regression, categorical encoding, decision tree ensembles,
#       and evaluation metrics step-by-step without unnecessary abstractions.
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------------------
# STEP 1: LOAD THE DATASET
# ------------------------------------------------------------------------------
# WHAT: Read tabular data from a CSV file into a pandas DataFrame.
# WHY: Machine learning models require structured numeric/tabular inputs to operate on.
# CONCEPTUALLY: CSV data is loaded into memory as a table of rows (samples) and columns (features/target).

print("=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

df = pd.read_csv("data.csv")
print(f"Dataset Loaded Successfully! Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")
print("First 5 rows of raw dataset:")
print(df.head())
print("\nDataset Info / Summary:")
print(df.info())

# ------------------------------------------------------------------------------
# STEP 2: DATA CLEANING & FEATURE ENGINEERING
# ------------------------------------------------------------------------------
# WHAT: Clean raw data and engineer new domain-specific features (e.g. Car Age).
# WHY: 'Year' as a raw calendar year (e.g., 2018) is less direct than 'Car_Age' (e.g., 6 years old).
# CONCEPTUALLY: Price depreciates with age, not calendar year directly. Converting Year to Age
#               helps decision trees split effectively on how old the vehicle is.

print("\n" + "=" * 70)
print("STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 70)

# Check for missing values
print("Missing values per column before cleaning:")
print(df.isnull().sum())

# Feature engineering: Derive Car_Age from Year
CURRENT_YEAR = 2024
df['Car_Age'] = CURRENT_YEAR - df['Year']

# Drop original 'Year' column as 'Car_Age' replaces it
df = df.drop(columns=['Year'])

print("\nEngineered 'Car_Age' feature (Current Year 2024 - Manufacturing Year).")
print(df[['Brand', 'Car_Age', 'Kms_Driven', 'Selling_Price']].head())

# ------------------------------------------------------------------------------
# STEP 3: SEPARATE FEATURES (X) AND TARGET (y)
# ------------------------------------------------------------------------------
# WHAT: Split the dataset into predictor variables (X) and target variable (y).
# WHY: Machine learning algorithms learn a function mapping f(X) -> y.
# CONCEPTUALLY: X is the input matrix containing car attributes; y is the output vector containing prices.

print("\n" + "=" * 70)
print("STEP 3: SEPARATING FEATURES (X) AND TARGET (y)")
print("=" * 70)

X_raw = df.drop(columns=['Selling_Price'])
y = df['Selling_Price']

print(f"Features (X_raw) shape: {X_raw.shape}")
print(f"Target (y) shape: {y.shape}")

# ------------------------------------------------------------------------------
# STEP 4: CATEGORICAL ENCODING (ONE-HOT ENCODING)
# ------------------------------------------------------------------------------
# WHAT: Convert categorical text features ('Brand', 'Fuel_Type', 'Transmission', 'Owner_Type')
#       into binary indicator columns (0s and 1s).
# WHY: Mathematical models cannot directly process text strings like "Maruti" or "Diesel".
# CONCEPTUALLY: Creates a new column for each category (e.g., 'Fuel_Type_Diesel' = 1 or 0).
#               We drop the first category (`drop_first=True`) to avoid multi-collinearity / redundant features.

print("\n" + "=" * 70)
print("STEP 4: CATEGORICAL ENCODING (ONE-HOT ENCODING)")
print("=" * 70)

categorical_cols = ['Brand', 'Fuel_Type', 'Transmission', 'Owner_Type']
print(f"Categorical columns to encode: {categorical_cols}")

# One-Hot Encoding using pandas get_dummies
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int)

print(f"\nFeatures shape after One-Hot Encoding: {X.shape}")
print("Encoded feature names (first 10 columns):")
print(list(X.columns[:10]))

# ------------------------------------------------------------------------------
# STEP 5: TRAIN / TEST SPLIT
# ------------------------------------------------------------------------------
# WHAT: Divide data into a Training set (80%) and a Testing set (20%).
# WHY: We must evaluate the model on unseen data to test for generalization and check for overfitting.
# CONCEPTUALLY: The model learns patterns only from the training set; the testing set acts as an unseen exam.

print("\n" + "=" * 70)
print("STEP 5: TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training features (X_train) shape: {X_train.shape}")
print(f"Testing features (X_test) shape:  {X_test.shape}")
print(f"Training target (y_train) shape:   {y_train.shape}")
print(f"Testing target (y_test) shape:     {y_test.shape}")

# ------------------------------------------------------------------------------
# STEP 6: MODEL INSTANTIATION
# ------------------------------------------------------------------------------
# WHAT: Instantiate the Random Forest Regressor algorithm.
# WHY: Random Forest is an ensemble method combining multiple decision trees (Bagging).
# CONCEPTUALLY:
# - n_estimators=100: Builds 100 individual decision trees.
# - max_depth=10: Prevents trees from growing infinitely deep (controls overfitting).
# - random_state=42: Ensures reproducible, identical tree splits across runs.

print("\n" + "=" * 70)
print("STEP 6: INITIALIZING RANDOM FOREST REGRESSOR")
print("=" * 70)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

print(f"Model Configuration: {model}")

# ------------------------------------------------------------------------------
# STEP 7: MODEL TRAINING (FIT)
# ------------------------------------------------------------------------------
# WHAT: Train the Random Forest on X_train and y_train using .fit().
# WHY: The model learns how feature combinations map to car prices.
# CONCEPTUALLY:
# 1. Random Forest creates 100 random bootstrap samples (subsets with replacement) of the training data.
# 2. At each split point in each tree, it chooses from a random subset of features to find the best split point (minimizing MSE).
# 3. Each tree learns to partition feature space into leaf nodes containing average prices.

print("\n" + "=" * 70)
print("STEP 7: TRAINING THE MODEL")
print("=" * 70)

model.fit(X_train, y_train)
print("Model training complete! 100 decision trees built.")

# ------------------------------------------------------------------------------
# STEP 8: PREDICTION & EVALUATION
# ------------------------------------------------------------------------------
# WHAT: Predict prices on X_test and calculate evaluation metrics against y_test.
# WHY: Quantitative metrics show how accurate predictions are in actual monetary units.
# CONCEPTUALLY:
# - MAE (Mean Absolute Error): Average absolute difference in Lakhs between real & predicted prices.
# - MSE (Mean Squared Error): Average squared difference (penalizes large errors).
# - RMSE (Root Mean Squared Error): Square root of MSE (in original unit: Lakhs).
# - R² Score (Coefficient of Determination): Proportion of variance in target explained by model (1.0 = perfect).

print("\n" + "=" * 70)
print("STEP 8: PREDICTION & EVALUATION ON TEST DATA")
print("=" * 70)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE):     {mae:.3f} Lakhs (Average off by Rs. {mae*100000:.0f})")
print(f"Mean Squared Error (MSE):      {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f} Lakhs")
print(f"R^2 Score (Accuracy Metric):    {r2:.4f} ({r2*100:.2f}% of price variance explained)")

# Display sample side-by-side comparison
comparison_df = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': np.round(y_pred, 2),
    'Absolute Difference': np.round(np.abs(y_test.values - y_pred), 2)
}).head(7)

print("\nSample Predictions vs Actual Values:")
print(comparison_df)

# ------------------------------------------------------------------------------
# STEP 9: MODEL INSPECTION (FEATURE IMPORTANCES)
# ------------------------------------------------------------------------------
# WHAT: Inspect feature importance scores learned by Random Forest.
# WHY: Understand WHICH features contributed most to predicting car price.
# CONCEPTUALLY: Measures how much each feature reduces impurity/variance across all 100 trees.

print("\n" + "=" * 70)
print("STEP 9: FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Top 10 Most Influential Features:")
print(importance_df.head(10).to_string(index=False))

# ------------------------------------------------------------------------------
# STEP 10: PREDICTING ON UNSEEN NEW CAR DATA
# ------------------------------------------------------------------------------
# WHAT: Pass a brand-new car record through the trained pipeline.
# WHY: Demonstrate how trained models deploy to forecast real-world cases.
# CONCEPTUALLY: Format input to match identical encoded features, run through all 100 trees, and take average prediction.

print("\n" + "=" * 70)
print("STEP 10: PREDICTING PRICE FOR A NEW/UNSEEN CAR")
print("=" * 70)

new_car_raw = {
    'Kms_Driven': 45000,
    'Engine_CC': 1496,
    'Max_Power_BHP': 118.0,
    'Car_Age': 4,
    'Brand': 'Honda',
    'Fuel_Type': 'Petrol',
    'Transmission': 'Automatic',
    'Owner_Type': 'First'
}

print("New Car Specification:")
for k, v in new_car_raw.items():
    print(f"  {k}: {v}")

# Create DataFrame for single sample
new_car_df = pd.DataFrame([new_car_raw])

# Encode single sample using pandas get_dummies
new_car_encoded = pd.get_dummies(new_car_df, columns=categorical_cols, drop_first=True, dtype=int)

# Align columns with X_train (ensure missing dummy columns are filled with 0)
new_car_encoded = new_car_encoded.reindex(columns=X.columns, fill_value=0)

# Generate prediction
predicted_price = model.predict(new_car_encoded)[0]

print("\n" + "-" * 45)
print(f" PREDICTED SELLING PRICE: {predicted_price:.2f} Lakhs (Rs. {predicted_price*100000:,.0f})")
print("-" * 45)
print("\n" + "=" * 70)
print("END OF PIPELINE")
print("=" * 70)
