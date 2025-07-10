import streamlit as st
import pandas as pd
import joblib

clf = joblib.load("model_is_delayed.pkl")
reg_delay = joblib.load("model_delay.pkl")
reg_eta = joblib.load("model_eta.pkl")

st.title("Prediksi Pengiriman Pizza")

month = st.selectbox("Bulan Pemesanan", list(range(1, 13)))
pizza_size = st.selectbox("Ukuran Pizza (0: Small, 1: Medium, 2: Large)", [0, 1, 2])
pizza_type = st.selectbox("Tipe Pizza (0: Veg, 1: Meat, 2: Cheese)", [0, 1, 2])
toppings = st.slider("Jumlah Topping", 1, 6, 3)
distance = st.slider("Jarak (km)", 0.5, 20.0, 5.0)
traffic = st.selectbox("Tingkat Kemacetan (0: Rendah - 3: Tinggi)", [0, 1, 2, 3])
weekend = st.selectbox("Akhir Pekan?", [0, 1])
density = st.slider("Kepadatan Topping", 0.0, 1.0, 0.5)
est_dur = st.slider("Estimasi Waktu Sistem (menit)", 5, 60, 25)
complexity = st.selectbox("Kompleksitas Pizza (0: Mudah - 2: Rumit)", [0, 1, 2])
traffic_impact = st.selectbox("Dampak Lalu Lintas", [0, 1, 2, 3])
order_hour = st.slider("Jam Pemesanan", 0, 23, 18)
avg_rest_time = st.slider("Waktu Rata-Rata Restoran", 5, 60, 20)

if st.button("Prediksi Pengiriman"):
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

    st.subheader("Hasil Prediksi")
    st.write(f"🕒 **Estimasi Pengiriman:** {pred_eta:.2f} menit")
    st.write(f"⏱️ **Prediksi Keterlambatan:** {pred_delay:.2f} menit")
    st.write(f"⚠️ **Kemungkinan Telat:** {'YA' if is_delayed else 'TIDAK'}")
