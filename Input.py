import streamlit as st
import pandas as pd
import joblib

# Load model
clf = joblib.load("model_is_delayed.pkl")
reg_delay = joblib.load("model_delay.pkl")
reg_eta = joblib.load("model_eta.pkl")

st.title("Prediksi Pengiriman Pizza 🍕")

# Form input sebagai dropdown
with st.form("input_form"):
    month = st.selectbox("Bulan Pemesanan", list(range(1, 13)))
    pizza_size = st.selectbox("Ukuran Pizza", [0, 1, 2], format_func=lambda x: ["Small", "Medium", "Large"][x])
    pizza_type = st.selectbox("Tipe Pizza", [0, 1, 2], format_func=lambda x: ["Veg", "Meat", "Cheese"][x])
    toppings = st.selectbox("Jumlah Topping", list(range(1, 7)))
    distance = st.slider("Jarak (km)", 0.5, 20.0, 5.0, step=0.1)  # Tetap slider
    traffic = st.selectbox("Tingkat Kemacetan", [0, 1, 2, 3], format_func=lambda x: ["Rendah", "Sedang", "Padat", "Macet"][x])
    weekend = st.selectbox("Apakah Akhir Pekan?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    density = st.selectbox("Kepadatan Topping", [round(i * 0.1, 1) for i in range(0, 11)])  # 0.0 - 1.0
    est_dur = st.selectbox("Estimasi Waktu Sistem (menit)", list(range(5, 61, 5)))
    complexity = st.selectbox("Kompleksitas Pizza", [0, 1, 2], format_func=lambda x: ["Mudah", "Sedang", "Rumit"][x])
    traffic_impact = st.selectbox("Dampak Lalu Lintas", [0, 1, 2, 3])
    order_hour = st.selectbox("Jam Pemesanan", list(range(0, 24)))
    avg_rest_time = st.selectbox("Waktu Rata-Rata Restoran (menit)", list(range(5, 61, 5)))

    submitted = st.form_submit_button("Prediksi")

# Prediksi saat tombol ditekan
if submitted:
    input_df = pd.DataFrame([{
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
        'Traffic Impact': traffic_impact,
        'Order Hour': order_hour,
        'Restaurant Avg Time': avg_rest_time
    }])

    pred_eta = reg_eta.predict(input_df)[0]
    pred_delay = reg_delay.predict(input_df)[0]
    is_delayed = clf.predict(input_df)[0]

    st.subheader("Hasil Prediksi 📊")
    st.write(f"🕒 **Estimasi Pengiriman:** {pred_eta:.2f} menit")
    st.write(f"⏱️ **Prediksi Keterlambatan:** {pred_delay:.2f} menit")
    st.write(f"⚠️ **Kemungkinan Telat:** {'YA' if is_delayed else 'TIDAK'}")
