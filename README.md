# Used Car Price Predictor — Learning-First ML Project

An academic, beginner-friendly Python Machine Learning project designed to demonstrate **Tabular Regression** using the **Random Forest Regressor** algorithm.

---

## What Problem We Are Solving

We want to predict the **Selling Price (in Lakhs)** of a used car based on its attributes:
- **Brand / Make** (e.g., Maruti, Hyundai, Honda, Toyota, BMW, Ford)
- **Age of the Car** (Derived from manufacturing year)
- **Kilometers Driven**
- **Fuel Type** (Petrol, Diesel, CNG)
- **Transmission Type** (Manual, Automatic)
- **Ownership Level** (First, Second, Third)
- **Engine Capacity** (in CC)
- **Maximum Power** (in BHP)

Predicting used car prices is significantly harder than basic linear regression because car values suffer from **non-linear depreciation** (cars lose value faster in the first 3 years), complex feature interactions (e.g., luxury brands depreciate differently), and categorical inputs (Brand, Fuel Type, Transmission).

---

## 🚀 Interactive Streamlit Web App

The project includes an interactive web dashboard built with **Streamlit** (`app.py`), allowing real-time price predictions and data exploration.

### How to Run the App:
```bash
py -m streamlit run app.py
```

### Features:
- **🔮 Real-Time Price Estimation**: Select vehicle brand, manufacturing year, fuel type, transmission, mileage, engine CC, max power, and ownership history to get an instant resale valuation in **Lakhs** and **Rupees (₹)**.
- **📊 Model Performance & Feature Importance**: View live $R^2$, MAE, RMSE metrics and interactive Plotly bar charts of top predictive features.
- **📁 Dataset Explorer & Visualizations**: Interactive scatter plots (Price vs Power) and boxplots (Brand price distribution).

---

## 1. Concepts Learned

1. **Tabular Regression**: Predicting a continuous numerical outcome ($y$) based on multiple predictor features ($X$).
2. **Feature Engineering**: Transforming raw columns like `Year` into domain-meaningful variables like `Car_Age = Current_Year - Year`.
3. **Categorical One-Hot Encoding**: Converting textual categorical attributes into numeric $0$ or $1$ binary indicator vectors.
4. **Train/Test Splitting**: Partitioning data into 80% training set (for learning parameters) and 20% testing set (for unbiased evaluation).
5. **Ensemble Learning (Bagging)**: Combining predictions from multiple individual decision trees to produce lower variance and higher accuracy.
6. **Feature Importance Analysis**: Extracting mathematical impurity reduction values to determine which features drive price predictions.
7. **Regression Evaluation Metrics**: MAE, MSE, RMSE, and $R^2$ Score.

---

## 2. Algorithm Used: Random Forest Regressor

### Simple Explanation
A **Decision Tree** asks a sequence of yes/no questions to split data into price estimates (e.g., *"Is Max_Power > 120 BHP?"* -> *"Is Car_Age < 4 years?"*). However, a single decision tree is prone to **overfitting** (memorizing the training data).

A **Random Forest** solves this by building a collection ("forest") of hundreds of diverse decision trees:
- **Bootstrap Aggregation (Bagging)**: Each tree receives a random subset of rows sampled with replacement.
- **Random Feature Selection**: At every split point, each tree only looks at a random subset of available features.
- **Aggregation / Averaging**: When predicting a new car's price, all trees make an independent prediction, and the forest outputs the **average price** across all trees.

$$\hat{y}_{final} = \frac{1}{N} \sum_{i=1}^{N} \text{Tree}_i(X)$$

---

## 3. Machine Learning Pipeline Flow

```
[ Raw CSV Dataset ]
        │
        ▼
[ Step 1: Load Data & Inspect ]
        │
        ▼
[ Step 2: Feature Engineering ] ──► (Car_Age = 2024 - Year)
        │
        ▼
[ Step 3: X / y Separation ]
   ├── Features X (Brand, Age, Kms, Fuel, etc.)
   └── Target y (Selling_Price)
        │
        ▼
[ Step 4: One-Hot Encoding ] ──► (pd.get_dummies for text categories)
        │
        ▼
[ Step 5: Train / Test Split ]
   ├── X_train, y_train (80%)
   └── X_test, y_test (20%)
        │
        ▼
[ Step 6 & 7: Random Forest Model Training ] ──► (n_estimators=100, max_depth=10)
        │
        ▼
[ Step 8: Evaluation & Predictions ]
   ├── MAE, MSE, RMSE, R² Score
   └── Actual vs Predicted Comparison
        │
        ▼
[ Step 9: Feature Importance Analysis ]
        │
        ▼
[ Step 10: Unseen Sample Prediction ] ──► (Input single car data ──► Predicted Price)
```

---

## 4. Things You Should Experiment With

Try editing `main.py` directly to see how the model behaves under different configurations:

1. **Change the Number of Trees (`n_estimators`)**:
   - Change `n_estimators=10` or `n_estimators=300` in Step 6.
   - *Observe*: Does accuracy ($R^2$) improve with more trees? Does execution time increase?
2. **Limit Tree Depth (`max_depth`)**:
   - Try setting `max_depth=2` (underfitting) vs `max_depth=None` (overfitting).
   - *Observe*: Notice how restrictive depth hurts accuracy, while unbounded depth can overfit small datasets.
3. **Compare with Linear Regression**:
   - Import `from sklearn.linear_model import LinearRegression` and fit it on the same training data.
   - *Observe*: Compare the $R^2$ score of Linear Regression vs Random Forest. Why does Random Forest perform better on non-linear data?
4. **Test Different Train/Test Ratios**:
   - Change `test_size=0.10` or `test_size=0.40` in Step 5.
   - *Observe*: How does reducing training data affect test set metrics?
5. **Engineer a New Ratio Feature**:
   - Try adding a new feature: `df['Power_to_CC'] = df['Max_Power_BHP'] / df['Engine_CC']`.
   - *Observe*: Check Step 9 (Feature Importance) to see if `Power_to_CC` ranks high!

---

## 5. Next Recommended Step

Now that you have mastered non-linear regression using an ensemble method (Random Forest), your next step should be:

### **Next Project: Customer Churn Prediction (Classification using XGBoost / Gradient Boosting)**
- **Why**: Moving from **Regression** (predicting continuous numbers) to **Classification** (predicting binary outcomes: Will a customer stay or leave?).
- **What You Will Learn**:
  - Binary Cross-Entropy Loss
  - Boosting (building trees sequentially to fix previous errors) vs Bagging
  - Classification Metrics: Precision, Recall, F1-Score, ROC-AUC curve.
