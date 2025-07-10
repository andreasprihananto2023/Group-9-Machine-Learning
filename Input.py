import streamlit as st
import pandas as pd
import joblib

# === Load model dan fitur ===
clf = joblib.load("model_is_delayed.pkl")
reg_delay = joblib.load("model_delay.pkl")
reg_eta = joblib.load("model_eta.pkl")

# === Daftar fitur ===
expected_features = [
    'Order Month', 'Pizza Size', 'Pizza Type', 'Toppings Count', 'Distance (km)',
    'Traffic Level', 'Is Weekend', 'Topping Density',
    'Estimated Duration (min)', 'Pizza Complexity', 'Order Hour'
]

# === Judul Aplikasi ===
st.title("📦 Prediksi Pengiriman Pizza")

# === Form Input ===
month = st.selectbox("Bulan Pemesanan", list(range(1, 13)), index=6)
pizza_size = st.selectbox("Ukuran Pizza", {"Small": 0, "Medium": 1, "Large": 2})
pizza_type = st.selectbox("Tipe Pizza", {
    "Vegetarian": 0,
    "Meat Lovers": 1,
    "Cheese Only": 2,
    "Supreme": 3,
    "Seafood": 4
})
toppings = st.slider("Jumlah Topping", 1, 6, 3)
distance = st.number_input("Jarak (km)", min_value=0.5, max_value=50.0, value=5.0, step=0.1)
traffic = st.selectbox("Tingkat Kemacetan", {"Rendah": 0, "Sedang": 1, "Padat": 2, "Macet Total": 3})
weekend = st.selectbox("Hari Pemesanan", {"Hari Biasa": 0, "Akhir Pekan": 1})
density = st.slider("Kepadatan Topping", 0.0, 1.0, 0.5)
est_dur = st.slider("Estimasi Waktu Sistem (menit)", 5, 60, 25)
complexity = st.selectbox("Kompleksitas Pizza", {"Mudah": 0, "Sedang": 1, "Rumit": 2})
order_hour = st.slider("Jam Pemesanan", 0, 23, 18)

# === Prediksi ===
if st.button("🔍 Prediksi"):
    input_dict = {
        'Order Month': month,
        'Pizza Size': pizza_size,
        'Pizza Type': pizza_type,
        'Toppings Count': toppings,
        'Distance (km)': distance,
        'Traffic Level': traffic,
        'Is Weekend': weekend,
        'Topping Density': density,
        'Estimated Duration (min)': est_dur,
        'Pizza Complexity': complexity,
        'Order Hour': order_hour
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[expected_features]
    input_df = input_df.astype(float)  # Konversi ke float agar cocok untuk XGBoost

    pred_eta = reg_eta.predict(input_df)[0]
    pred_delay = reg_delay.predict(input_df)[0]
    is_delayed = clf.predict(input_df)[0]

    # === Output ===
    st.subheader("📊 Hasil Prediksi")
    st.write(f"🕒 **Estimasi Pengiriman:** {pred_eta:.2f} menit")
    st.write(f"⏱️ **Perkiraan Keterlambatan:** {pred_delay:.2f} menit")
    st.write(f"⚠️ **Kemungkinan Telat:** {'YA' if is_delayed else 'TIDAK'}")
