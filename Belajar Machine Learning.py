import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import xgboost as xgb
import joblib



df = pd.read_excel("Enhanced_pizza_data.xlsx")


month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
if 'Order Month' in df.columns:
    df['Order Month'] = df['Order Month'].map(month_map)


df.dropna(inplace=True)

categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


features = [
    'Order Month', 'Pizza Size', 'Pizza Type', 'Toppings Count', 'Distance (km)',
    'Traffic Level', 'Is Weekend', 'Topping Density',
    'Estimated Duration (min)', 'Pizza Complexity',
    'Traffic Impact', 'Order Hour', 'Restaurant Avg Time'
]



X = df[features]
y_class = df['Is Delayed'] 

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)

print("\n=== MODEL 1: Prediksi Is_Delayed ===")
print("Akurasi:", accuracy_score(y_test, y_pred_class))


y_delay = df['Delay (min)']

X_train_delay, X_test_delay, y_train_delay, y_test_delay = train_test_split(X, y_delay, test_size=0.2, random_state=42)

reg_delay = RandomForestRegressor()
reg_delay.fit(X_train_delay, y_train_delay)
y_pred_delay = reg_delay.predict(X_test_delay)

print("\n=== MODEL 2: Prediksi Delay (min) ===")
print("RMSE:", np.sqrt(mean_squared_error(y_test_delay, y_pred_delay)))
print("R² Score:", r2_score(y_test_delay, y_pred_delay))

y_eta = df['Delivery Duration (min)']

X_train_eta, X_test_eta, y_train_eta, y_test_eta = train_test_split(X, y_eta, test_size=0.2, random_state=42)

reg_eta = xgb.XGBRegressor(objective="reg:squarederror")
reg_eta.fit(X_train_eta, y_train_eta)
y_pred_eta = reg_eta.predict(X_test_eta)

print("\n=== MODEL 3: Prediksi Delivery Duration (ETA) ===")
print("RMSE:", np.sqrt(mean_squared_error(y_test_eta, y_pred_eta)))
print("R² Score:", r2_score(y_test_eta, y_pred_eta))

joblib.dump(clf, "model_is_delayed.pkl")
joblib.dump(reg_delay, "model_delay.pkl")
joblib.dump(reg_eta, "model_eta.pkl")

plt.figure(figsize=(8, 5))
plt.scatter(y_test_eta, y_pred_eta, alpha=0.6, color='green')
plt.xlabel("Actual Delivery Time (min)")
plt.ylabel("Predicted ETA (min)")
plt.title("Prediksi vs Realita Waktu Pengiriman")
plt.plot([y_test_eta.min(), y_test_eta.max()], [y_test_eta.min(), y_test_eta.max()], 'r--')  # garis 45 derajat
plt.grid(True)
plt.tight_layout()
plt.show()
