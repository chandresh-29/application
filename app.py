import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_and_clean_data():
    df = pd.read_csv("Pollutant_Radar.csv")  # ✅ Directly loading your dataset
    df.dropna(inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'pollutant_id':  # Keep pollutant_id original for plotting later
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Encode target column separately
    le_target = LabelEncoder()
    df['pollutant_id'] = le_target.fit_transform(df['pollutant_id'])
    label_encoders['pollutant_id'] = le_target

    return df, label_encoders

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

    print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_preds))
    print("XGBoost Classification Report:\n", classification_report(y_test, xgb_preds))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues')
    plt.title("Random Forest Confusion Matrix")

    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(y_test, xgb_preds), annot=True, fmt='d', cmap='Greens')
    plt.title("XGBoost Confusion Matrix")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=True)

    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

def plot_city_pollutants():
    df = pd.read_csv("Pollutant_Radar.csv")
    df = df.dropna(subset=["city", "pollutant_id", "pollutant_avg"])

    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    sns.scatterplot(data=df, x="city", y="pollutant_avg", hue="pollutant_id", alpha=0.7, s=100)
    plt.xlabel("City", fontsize=12)
    plt.ylabel("Pollutant Concentration (µg/m³)", fontsize=12)
    plt.title("Air Quality Levels Across Cities", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def calculate_aqi_pm10(pm10):
    if pm10 <= 50:
        return (pm10 * 50) / 50
    elif pm10 <= 100:
        return 50 + ((pm10 - 50) * 50) / 50
    elif pm10 <= 250:
        return 100 + ((pm10 - 100) * 100) / 150
    elif pm10 <= 350:
        return 200 + ((pm10 - 250) * 100) / 100
    elif pm10 <= 430:
        return 300 + ((pm10 - 350) * 100) / 80
    else:
        return 400 + ((pm10 - 430) * 100) / 80

def calculate_aqi_pm25(pm25):
    if pm25 <= 30:
        return (pm25 * 50) / 30
    elif pm25 <= 60:
        return 50 + ((pm25 - 30) * 50) / 30
    elif pm25 <= 90:
        return 100 + ((pm25 - 60) * 100) / 30
    elif pm25 <= 120:
        return 200 + ((pm25 - 90) * 100) / 30
    elif pm25 <= 250:
        return 300 + ((pm25 - 120) * 100) / 130
    else:
        return 400 + ((pm25 - 250) * 100) / 130

def plot_aqi():
    df = pd.read_csv("Pollutant_Radar.csv").dropna(subset=['pollutant_id', 'pollutant_avg'])

    pm10_data = df[df['pollutant_id'] == 'PM10'].copy()
    pm10_data['AQI'] = pm10_data['pollutant_avg'].apply(calculate_aqi_pm10)

    pm25_data = df[df['pollutant_id'] == 'PM2.5'].copy()
    pm25_data['AQI'] = pm25_data['pollutant_avg'].apply(calculate_aqi_pm25)

    for data, pollutant in zip([pm10_data, pm25_data], ['PM10', 'PM2.5']):
        if not data.empty:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=data, x='pollutant_avg', y='AQI', alpha=0.6)
            sns.regplot(data=data, x='pollutant_avg', y='AQI', scatter=False, color='red', line_kws={'linestyle': '--'})
            plt.title(f'{pollutant} vs Air Quality Index (AQI)', fontsize=14, pad=20)
            plt.xlabel(f'{pollutant} Concentration (µg/m³)', fontsize=12)
            plt.ylabel('Air Quality Index (AQI)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

# Main script
if __name__ == "__main__":
    df, label_encoders = load_and_clean_data()
    train_models(df, target_column='pollutant_id')
    plot_city_pollutants()
    plot_aqi()
