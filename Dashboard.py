# ClusterDashboard.py

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# === Load data ===
df = pd.read_excel("Enhanced_pizza_data.xlsx")

# Mapping bulan ke angka jika perlu
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
if 'Order Month' in df.columns:
    df['Order Month'] = df['Order Month'].map(month_map)

# Drop nulls
df.dropna(inplace=True)

# === Fitur untuk clustering ===
features = [
    'Toppings Count', 'Distance (km)', 'Traffic Level',
    'Topping Density', 'Estimated Duration (min)',
    'Pizza Complexity', 'Traffic Impact', 'Restaurant Avg Time'
]

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Clustering ===
n_clusters = st.slider("Jumlah Cluster", 2, 8, 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# === Statistik tiap cluster ===
cluster_summary = df.groupby('Cluster').agg(
    Jumlah_Pesanan=('Cluster', 'count'),
    Rata2_Delay=('Delay (min)', 'mean'),
    Rata2_Estimasi=('Estimated Duration (min)', 'mean')
).sort_values(by='Rata2_Delay', ascending=False).reset_index()

# === Tampilan ===
st.title("Dashboard Cluster Keterlambatan Pizza")

st.subheader("Ringkasan Cluster")
st.dataframe(cluster_summary)

# === Visualisasi PCA 2D ===
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'] = pca_result[:,0]
df['PCA2'] = pca_result[:,1]

st.subheader("Visualisasi Cluster")
fig, ax = plt.subplots()
scatter = ax.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='Set1', alpha=0.7)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
st.pyplot(fig)

# === Opsi tampilkan data mentah ===
if st.checkbox("Tampilkan Data Mentah"):
    st.write(df.head(100))
