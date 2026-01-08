import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==============================
# SAVE / LOAD
# ==============================

def save_model(clf, scaler, model_file="btc_model.pkl", scaler_file="btc_scaler.pkl"):
    joblib.dump(clf, model_file)
    joblib.dump(scaler, scaler_file)


def load_model(model_file="btc_model.pkl", scaler_file="btc_scaler.pkl"):
    clf = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    return clf, scaler


# ==============================
# LABEL CREATION
# ==============================

def create_labels(df):
    """
    Simple regime labels based on current features.
    """
    df = df.copy()

    df["trend"] = df["trend"].astype(float)
    df["volatility"] = df["volatility"].astype(float)

    conditions = [
        df["trend"] > 0.001,
        (df["trend"].abs() <= 0.001) & (df["volatility"] < 0.01),
        df["volatility"] >= 0.01,
    ]

    choices = ["Trend", "Range", "High Volatility"]

    df["regime"] = np.select(conditions, choices, default="Unknown")
    return df


# ==============================
# MODEL TRAINING (CLOUD SAFE)
# ==============================

def train_model(df):
    features_list = ["return", "volatility", "trend"]

    X = df[features_list].copy()
    y = df["regime"].copy()

    # --- SAFETY CLEANING ---
    mask = np.isfinite(X).all(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]

    if len(X) < 50:
        raise ValueError("Not enough clean data to train model.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=5,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    return clf, scaler


# ==============================
# ENSURE MODEL EXISTS (DEPLOYMENT)
# ==============================

def ensure_model_exists(ticker="BTC-USD", interval="1h", period="730d"):
    """
    Streamlit Cloud starts with no .pkl files.
    Train + save model if missing.
    """
    model_file = "btc_model.pkl"
    scaler_file = "btc_scaler.pkl"

    if os.path.exists(model_file) and os.path.exists(scaler_file):
        return

    import data
    import features

    df = data.get_price_data(ticker=ticker, interval=interval, period=period)
    df = features.compute_features(df)
    df = create_labels(df)

    try:
        clf, scaler = train_model(df)
        save_model(clf, scaler, model_file, scaler_file)
        print("Model trained and saved successfully.")
    except ValueError as e:
        print(f"Model training skipped: {e}")


# ==============================
# LOCAL TEST / MANUAL TRAIN
# ==============================

if __name__ == "__main__":
    ensure_model_exists("BTC-USD")
    print("Model ensured (trained if missing).")


