# Dashboard.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Load Data ===
df = pd.read_excel("Enhanced_pizza_data.xlsx")

# === 2. Ubah Nama Bulan ke Angka ===
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
if 'Order Month' in df.columns:
    df['Order Month'] = df['Order Month'].map(month_map)

# === 3. Hapus Baris Kosong ===
df.dropna(inplace=True)

# === 4. Pilih Fitur Numerik untuk Clustering ===
features = [
    'Toppings Count', 'Distance (km)', 'Traffic Level',
    'Topping Density', 'Estimated Duration (min)',
    'Pizza Complexity', 'Traffic Impact', 'Restaurant Avg Time',
    'Delay (min)'  # Target untuk dilihat dalam distribusi antar klaster
]
X = df[features].copy()

# === 5. Pastikan Semua Data Numerik ===
X_clean = X.select_dtypes(include=['int64', 'float64']).copy()
X_clean.dropna(inplace=True)

# Sinkronkan kembali baris df sesuai X_clean
df = df.loc[X_clean.index]

# === 6. Scaling Data ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# === 7. Clustering dengan KMeans ===
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# === 8. Visualisasi Boxplot Keterlambatan per Cluster ===
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Delay (min)', data=df)
plt.title("Perbandingan Keterlambatan per Kluster")
plt.xlabel("Cluster")
plt.ylabel("Delay (min)")
plt.grid(True)
plt.tight_layout()
plt.show()
