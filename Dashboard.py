import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# === Judul Dashboard ===
st.title("📊 Clustering Keterlambatan Pengiriman Pizza")

# === Upload file ===
uploaded_file = st.file_uploader("Upload file Excel pizza", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # === Map bulan ke angka ===
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    if 'Order Month' in df.columns:
        df['Order Month'] = df['Order Month'].map(month_map)

    df.dropna(inplace=True)

    # === Fitur numerik untuk clustering ===
    features = [
        'Toppings Count', 'Distance (km)', 'Traffic Level',
        'Topping Density', 'Estimated Duration (min)',
        'Pizza Complexity', 'Traffic Impact', 'Restaurant Avg Time',
        'Delay (min)'
    ]
    X = df[features].copy()
    X_clean = X.apply(pd.to_numeric, errors='coerce')
    X_clean.dropna(inplace=True)
    df = df.loc[X_clean.index]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Clustering
    k = st.slider("Jumlah Cluster", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Rata-rata delay tiap cluster
    st.subheader("📈 Rata-rata Delay per Cluster")
    delay_mean = df.groupby('Cluster')['Delay (min)'].mean().sort_values(ascending=False)
    st.dataframe(delay_mean)

    # Boxplot Visualisasi
    st.subheader("📦 Visualisasi Delay per Cluster")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='Cluster', y='Delay (min)', data=df, palette='Set2', ax=ax)
    ax.set_title("Perbandingan Keterlambatan per Cluster")
    st.pyplot(fig)

    # Lihat order dari cluster dengan delay tertinggi
    st.subheader("🔍 Order dari Cluster Terlambat")
    cluster_terlambat = delay_mean.index[0]
    st.write(f"Cluster dengan delay tertinggi: {cluster_terlambat}")
    st.dataframe(df[df['Cluster'] == cluster_terlambat].head(10))

else:
    st.info("Silakan upload file Excel terlebih dahulu.")
