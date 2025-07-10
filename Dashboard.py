import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Load Data ===
df = pd.read_excel("Enhanced_pizza_data.xlsx")

# === 2. Map Bulan ===
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df['Order Month'] = df['Order Month'].map(month_map)

# === 3. Drop Nulls ===
df.dropna(inplace=True)

# === 4. Pilih fitur numerik ===
features = [
    'Toppings Count', 'Distance (km)', 'Traffic Level',
    'Topping Density', 'Estimated Duration (min)',
    'Pizza Complexity', 'Traffic Impact', 'Restaurant Avg Time',
    'Delay (min)'  # kamu bisa tambahkan ini untuk mengelompokkan berdasarkan keterlambatan juga
]
X = df[features].copy()

# Pastikan semua numeric
X = X.apply(pd.to_numeric, errors='coerce')
X.dropna(inplace=True)

# === 5. Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 6. Clustering ===
kmeans = KMeans(n_clusters=3, random_state=42)
df = df.loc[X.index]  # sinkronkan index setelah dropna
df['Cluster'] = kmeans.fit_predict(X_scaled)

# === 7. Visualisasi ===
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Delay (min)', data=df)
plt.title("Perbandingan Keterlambatan per Kluster")
plt.xlabel("Cluster")
plt.ylabel("Delay (min)")
plt.grid(True)
plt.tight_layout()
plt.show()
