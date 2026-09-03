import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

CURRENT_YEAR = 2024

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .price-display {
        font-size: 2.5rem;
        font-weight: 800;
        color: #059669;
    }
    .price-subtext {
        font-size: 1.2rem;
        font-weight: 600;
        color: #10B981;
    }
    </style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# DATA LOADING & MODEL TRAINING (CACHED)
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df['Car_Age'] = CURRENT_YEAR - df['Year']
    return df

@st.cache_resource
def train_model():
    df = load_data()
    
    # Feature & Target separation
    X_raw = df.drop(columns=['Selling_Price', 'Year'])
    y = df['Selling_Price']
    
    # One-hot encoding
    categorical_cols = ['Brand', 'Fuel_Type', 'Transmission', 'Owner_Type']
    X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    # Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions & Metrics
    y_pred = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Feature importances dataframe
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return model, X.columns, metrics, importance_df, categorical_cols

df_raw = load_data()
model, encoded_columns, metrics, importance_df, categorical_cols = train_model()


# ------------------------------------------------------------------------------
# HEADER SECTION
# ------------------------------------------------------------------------------
st.markdown("<div class='main-header'>🚗 Used Car Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predict estimated resale market price powered by a Random Forest Machine Learning Model</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# NAVIGATION TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Price Predictor", "📊 Model Performance & Insights", "📁 Dataset Explorer"])


# ==============================================================================
# TAB 1: PRICE PREDICTOR
# ==============================================================================
with tab1:
    col_input, col_result = st.columns([1.1, 1], gap="large")
    
    with col_input:
        st.subheader("📋 Enter Vehicle Details")
        
        col_a, col_b = st.columns(2)
        with col_a:
            brand = st.selectbox(
                "Brand / Manufacturer",
                options=sorted(df_raw['Brand'].unique()),
                index=2  # Default: Honda
            )
            year = st.slider(
                "Manufacturing Year",
                min_value=2010,
                max_value=2024,
                value=2020,
                step=1
            )
            fuel_type = st.selectbox(
                "Fuel Type",
                options=sorted(df_raw['Fuel_Type'].unique()),
                index=2  # Default: Petrol
            )
            transmission = st.radio(
                "Transmission Type",
                options=sorted(df_raw['Transmission'].unique()),
                index=1,
                horizontal=True
            )

        with col_b:
            kms_driven = st.number_input(
                "Kilometers Driven",
                min_value=1000,
                max_value=250000,
                value=45000,
                step=5000
            )
            engine_cc = st.slider(
                "Engine Displacement (CC)",
                min_value=700,
                max_value=3000,
                value=1496,
                step=50
            )
            max_power = st.slider(
                "Max Power (BHP)",
                min_value=40.0,
                max_value=260.0,
                value=118.0,
                step=1.0
            )
            owner_type = st.selectbox(
                "Owner History",
                options=sorted(df_raw['Owner_Type'].unique()),
                index=0  # Default: First
            )

        car_age = CURRENT_YEAR - year

    with col_result:
        st.subheader("💡 Estimated Valuation")
        
        # Prepare input dictionary
        input_data = {
            'Kms_Driven': kms_driven,
            'Engine_CC': engine_cc,
            'Max_Power_BHP': max_power,
            'Car_Age': car_age,
            'Brand': brand,
            'Fuel_Type': fuel_type,
            'Transmission': transmission,
            'Owner_Type': owner_type
        }
        
        # Process sample for prediction
        sample_df = pd.DataFrame([input_data])
        sample_encoded = pd.get_dummies(sample_df, columns=categorical_cols, drop_first=True, dtype=int)
        sample_encoded = sample_encoded.reindex(columns=encoded_columns, fill_value=0)
        
        predicted_price = model.predict(sample_encoded)[0]
        predicted_price_inr = max(0, predicted_price * 100000)
        
        st.markdown(
            f"""
            <div class='metric-card'>
                <p style='margin:0; font-size:1.1rem; color:#4B5563; font-weight:600;'>Estimated Market Value</p>
                <div class='price-display'>₹ {predicted_price:.2f} Lakhs</div>
                <div class='price-subtext'>≈ ₹ {predicted_price_inr:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("#### Vehicle Configuration Summary")
        summary_df = pd.DataFrame({
            "Attribute": ["Brand", "Manufacturing Year", "Car Age", "Kms Driven", "Fuel Type", "Transmission", "Owner Type", "Engine CC", "Max Power"],
            "Value": [brand, f"{year}", f"{car_age} Year(s)", f"{kms_driven:,} km", fuel_type, transmission, owner_type, f"{engine_cc} CC", f"{max_power} BHP"]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 2: MODEL PERFORMANCE & FEATURE IMPORTANCES
# ==============================================================================
with tab2:
    st.subheader("📈 Machine Learning Performance Metrics")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(label="R² Accuracy Score", value=f"{metrics['r2']*100:.2f}%", help="Proportion of variance explained by model")
    col_m2.metric(label="Mean Absolute Error (MAE)", value=f"₹ {metrics['mae']*100000:,.0f}", help="Average absolute prediction error")
    col_m3.metric(label="Root Mean Sq Error (RMSE)", value=f"{metrics['rmse']:.2f} Lakhs", help="Root mean squared error in Lakhs")
    
    st.markdown("---")
    st.subheader("🔍 Top Feature Importance Analysis")
    st.write("Random Forest calculates which features reduce tree impurity the most across all 100 decision trees:")
    
    top_10 = importance_df.head(10).copy()
    top_10['Feature_Clean'] = top_10['Feature'].str.replace('Brand_', 'Brand: ').str.replace('Fuel_Type_', 'Fuel: ').str.replace('Transmission_', 'Trans: ').str.replace('Owner_Type_', 'Owner: ')
    
    fig_importance = px.bar(
        top_10,
        x='Importance',
        y='Feature_Clean',
        orientation='h',
        title="Top 10 Most Influential Features in Price Prediction",
        labels={'Importance': 'Relative Importance', 'Feature_Clean': 'Feature Name'},
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig_importance.update_layout(yaxis=dict(autorange="reversed"), height=400)
    st.plotly_chart(fig_importance, use_container_width=True)


# ==============================================================================
# TAB 3: DATASET EXPLORER
# ==============================================================================
with tab3:
    st.subheader("📁 Used Car Dataset Overview")
    
    col_d1, col_d2 = st.columns([1, 2], gap="large")
    with col_d1:
        st.write("#### Dataset Quick Stats")
        st.write(f"- **Total Records**: {len(df_raw)} rows")
        st.write(f"- **Brands Included**: {', '.join(sorted(df_raw['Brand'].unique()))}")
        st.write(f"- **Year Range**: {df_raw['Year'].min()} - {df_raw['Year'].max()}")
        st.write(f"- **Price Range**: ₹ {df_raw['Selling_Price'].min():.2f} L to ₹ {df_raw['Selling_Price'].max():.2f} L")
        
    with col_d2:
        st.write("#### Raw Data Preview")
        st.dataframe(df_raw.head(10), use_container_width=True, hide_index=True)
        
    st.markdown("---")
    st.subheader("📊 Interactive Exploratory Data Analysis")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig_scatter = px.scatter(
            df_raw,
            x='Max_Power_BHP',
            y='Selling_Price',
            color='Brand',
            size='Engine_CC',
            hover_data=['Year', 'Kms_Driven', 'Fuel_Type'],
            title="Selling Price vs Max Power (BHP) by Brand",
            labels={'Selling_Price': 'Selling Price (Lakhs)', 'Max_Power_BHP': 'Max Power (BHP)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_e2:
        fig_box = px.box(
            df_raw,
            x='Brand',
            y='Selling_Price',
            color='Brand',
            title="Price Distribution Across Brands",
            labels={'Selling_Price': 'Selling Price (Lakhs)'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
