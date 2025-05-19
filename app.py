import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---- SETUP ----
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.title("🌍 Air Pollution Analysis & Prediction")

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    df = pd.read_csv("Pollutant_Radar.csv")
    df.dropna(inplace=True)
    return df

df = load_data()
st.success("✅ Dataset loaded successfully!")

# ---- ENCODE DATA ----
def encode_data(df):
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

encoded_df, label_encoders = encode_data(df.copy())

# ---- MODEL TRAINING ----
def train_models(df, target_column='pollutant_id'):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)

    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='mlogloss')
    xgb_model.fit(X_train_scaled, y_train)
    xgb_preds = xgb_model.predict(X_test_scaled)

    return {
        "rf": (rf_model, rf_preds),
        "xgb": (xgb_model, xgb_preds),
        "y_test": y_test,
        "X_columns": X.columns
    }

with st.spinner("Training models..."):
    models = train_models(encoded_df)

# ---- EVALUATION ----
st.header("🔍 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Random Forest Classification Report")
    st.text(classification_report(models["y_test"], models["rf"][1]))

with col2:
    st.subheader("XGBoost Classification Report")
    st.text(classification_report(models["y_test"], models["xgb"][1]))

# ---- CONFUSION MATRICES ----
st.subheader("📊 Confusion Matrices")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(models["y_test"], models["rf"][1]), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Random Forest")

sns.heatmap(confusion_matrix(models["y_test"], models["xgb"][1]), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("XGBoost")

st.pyplot(fig)

# ---- FEATURE IMPORTANCE ----
st.subheader("🔥 Feature Importance (XGBoost)")
xgb_model = models["xgb"][0]
importance_df = pd.DataFrame({
    "Feature": models["X_columns"],
    "Importance": xgb_model.feature_importances_
}).sort_values("Importance", ascending=True)

fig_imp, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df["Feature"], importance_df["Importance"], color='darkorange')
ax.set_title("XGBoost Feature Importance")
st.pyplot(fig_imp)

# ---- VISUALIZATION: CITY VS POLLUTANTS ----
st.header("🏙️ City-wise Pollutant Levels")

fig_city, ax_city = plt.subplots(figsize=(14, 6))
sns.scatterplot(data=df, x="city", y="pollutant_avg", hue="pollutant_id", s=100, alpha=0.7, ax=ax_city)
ax_city.set_xticklabels(ax_city.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig_city)

# ---- AQI CALCULATIONS ----
def calculate_aqi_pm10(pm10):
    if pm10 <= 50: return (pm10 * 50) / 50
    elif pm10 <= 100: return 50 + ((pm10 - 50) * 50) / 50
    elif pm10 <= 250: return 100 + ((pm10 - 100) * 100) / 150
    elif pm10 <= 350: return 200 + ((pm10 - 250) * 100) / 100
    elif pm10 <= 430: return 300 + ((pm10 - 350) * 100) / 80
    else: return 400 + ((pm10 - 430) * 100) / 80

def calculate_aqi_pm25(pm25):
    if pm25 <= 30: return (pm25 * 50) / 30
    elif pm25 <= 60: return 50 + ((pm25 - 30) * 50) / 30
    elif pm25 <= 90: return 100 + ((pm25 - 60) * 100) / 30
    elif pm25 <= 120: return 200 + ((pm25 - 90) * 100) / 30
    elif pm25 <= 250: return 300 + ((pm25 - 120) * 100) / 130
    else: return 400 + ((pm25 - 250) * 100) / 130

st.header("🧮 AQI Analysis")

pm10_data = df[df['pollutant_id'] == 'PM10'].copy()
pm10_data['AQI'] = pm10_data['pollutant_avg'].apply(calculate_aqi_pm10)

pm25_data = df[df['pollutant_id'] == 'PM2.5'].copy()
pm25_data['AQI'] = pm25_data['pollutant_avg'].apply(calculate_aqi_pm25)

for data, label in zip([pm10_data, pm25_data], ['PM10', 'PM2.5']):
    if not data.empty:
        fig_aqi, ax_aqi = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x='pollutant_avg', y='AQI', ax=ax_aqi)
        sns.regplot(data=data, x='pollutant_avg', y='AQI', scatter=False, color='red', ax=ax_aqi)
        ax_aqi.set_title(f'{label} Concentration vs AQI')
        st.pyplot(fig_aqi)

# ---- USER INPUT (Optional: Add prediction form here) ----
st.header("🧾 Predict Pollutant Type (Coming soon...)")
st.info("Interactive input for custom predictions can be added. Let me know if you'd like that now.")

