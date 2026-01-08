from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

# ------------------------
# Save / Load
# ------------------------

def save_model(clf, scaler, model_file="btc_model.pkl", scaler_file="btc_scaler.pkl"):
    joblib.dump(clf, model_file)
    joblib.dump(scaler, scaler_file)

def load_model(model_file="btc_model.pkl", scaler_file="btc_scaler.pkl"):
    clf = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    return clf, scaler

def ensure_model_exists(ticker="BTC-USD", interval="1h", period="730d"):
    """
    Streamlit Cloud starts with no .pkl files.
    This trains + saves them if missing.
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

    clf, scaler = train_model(df)
    save_model(clf, scaler, model_file=model_file, scaler_file=scaler_file)

# ------------------------
# Label creation
# ------------------------

def create_labels(df):
    """
    Simple regime labels based on current features (kept consistent with your existing pipeline).
    """
    df = df.copy()

    df["trend"] = df["trend"].astype(float)
    df["volatility"] = df["volatility"].astype(float)

    conditions = [
        df["trend"] > 0.001,
        (df["trend"].abs() <= 0.001) & (df["volatility"] < 0.01),
        df["volatility"] >= 0.01
    ]
    choices = ["Trend", "Range", "High Volatility"]

    df["regime"] = np.select(conditions, choices, default="Unknown")
    return df

# ------------------------
# Training
# ------------------------

def train_model(df):
    features_list = ["return", "volatility", "trend"]
    X = df[features_list]
    y = df["regime"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Print only when run directly
    return clf, scaler

# ------------------------
# Main (local training)
# ------------------------

if __name__ == "__main__":
    ensure_model_exists("BTC-USD")
    print("Model ensured (trained/saved if missing).")




