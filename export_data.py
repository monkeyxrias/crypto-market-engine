import data
import features

# 1. Get BTC data
df = data.get_btc_data()

# 2. Compute features
df = features.compute_features(df)

# 3. Save to CSV
df.to_csv("data.csv", index=False)

print("data.csv created successfully")
