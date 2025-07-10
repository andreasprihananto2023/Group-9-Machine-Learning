import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import xgboost as xgb
import joblib

# Load data
df = pd.read_excel("Enhanced_pizza_data.xlsx")

# Map bulan
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
if 'Order Month' in df.columns:
    df['Order Month'] = df['Order Month'].map(month_map)

# Bersihkan data
df.dropna(inplace=True)

# Label Encoding
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Fitur per model
features_eta = ['Order Month', 'Pizza Size', 'Pizza Type', 'Toppings Count', 'Distance (km)', 'Traffic Level', 'Order Hour']
features_delay = ['Pizza Complexity', 'Restaurant Avg Time', 'Toppings Count', 'Topping Density', 'Traffic Level', 'Is Weekend', 'Order Hour']
features_class = ['Distance (km)', 'Traffic Level', 'Is Weekend', 'Order Hour', 'Pizza Complexity']

# ================== MODEL 1: ETA ==================
X_eta = df[features_eta]
y_eta = df['Delivery Duration (min)']
X_train_eta, X_test_eta, y_train_eta, y_test_eta = train_test_split(X_eta, y_eta, test_size=0.2, random_state=42)
reg_eta = xgb.XGBRegressor(objective="reg:squarederror")
reg_eta.fit(X_train_eta, y_train_eta)
print("Model ETA trained. R²:", r2_score(y_test_eta, reg_eta.predict(X_test_eta)))
joblib.dump(reg_eta, "model_eta.pkl")

# ================== MODEL 2: DELAY ==================
X_delay = df[features_delay]
y_delay = df['Delay (min)']
X_train_delay, X_test_delay, y_train_delay, y_test_delay = train_test_split(X_delay, y_delay, test_size=0.2, random_state=42)
reg_delay = RandomForestRegressor()
reg_delay.fit(X_train_delay, y_train_delay)
print("Model Delay trained. R²:", r2_score(y_test_delay, reg_delay.predict(X_test_delay)))
joblib.dump(reg_delay, "model_delay.pkl")

# ================== MODEL 3: IS_DELAYED ==================
X_class = df[features_class]
y_class = df['Is Delayed']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train_class, y_train_class)
print("Model Classification trained. Accuracy:", accuracy_score(y_test_class, clf.predict(X_test_class)))
joblib.dump(clf, "model_is_delayed.pkl")

# Optional: Visualisasi ETA
plt.figure(figsize=(8, 5))
plt.scatter(y_test_eta, reg_eta.predict(X_test_eta), alpha=0.6, color='green')
plt.xlabel("Actual Delivery Time (min)")
plt.ylabel("Predicted ETA (min)")
plt.title("ETA: Prediksi vs Aktual")
plt.plot([y_test_eta.min(), y_test_eta.max()], [y_test_eta.min(), y_test_eta.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()
