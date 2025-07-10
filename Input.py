import streamlit as st
import pandas as pd
import joblib
import random


clf = joblib.load("model_is_delayed.pkl")
reg_delay = joblib.load("model_delay.pkl")
reg_eta = joblib.load("model_eta.pkl")

st.title("Prediksi Pengiriman Pizza 🍕")

def generate_random_input():
    return {
        'Order Month': random.randint(1, 12),
        'Pizza Size': random.choice([0, 1, 2]),
        'Pizza Type': random.choice([0, 1, 2]),
        'Toppings Count': random.randint(1, 6),
        'Distance (km)': round(random.uniform(0.5, 20.0), 1),
        'Traffic Level': random.randint(0, 3),
        'Is Weekend': random.randint(0, 1),
        'Topping Density': round(random.uniform(0.0, 1.0), 1),
        'Estimated Duration (min)': random.choice(list(range(5, 61, 5))),
        'Pizza Complexity': random.randint(0, 2),
        'Traffic Impact': random.randint(0, 3),
        'Order Hour': random.randint(0, 23),
        'Restaurant Avg Time': random.choice(list(range(5, 61, 5)))
    }

if 'random_values' not in st.session_state:
    st.session_state.random_values = generate_random_input()

# Tombol untuk generate random
if st.button("🎲 Isi Otomatis (Random)"):
    st.session_state.random_values = generate_random_input()

# --- Form input
with st.form("input_form"):
    rv = st.session_state.random_values

    month = st.selectbox("Bulan Pemesanan", list(range(1, 13)), index=rv['Order Month'] - 1)
    pizza_size = st.selectbox("Ukuran Pizza", [0, 1, 2], format_func=lambda x: ["Small", "Medium", "Large"][x], index=rv['Pizza Size'])
    pizza_type = st.selectbox("Tipe Pizza", [0, 1, 2], format_func=lambda x: ["Veg", "Meat", "Cheese"][x], index=rv['Pizza Type'])
    toppings = st.selectbox("Jumlah Topping", list(range(1, 7)), index=rv['Toppings Count'] - 1)
    distance = st.slider("Jarak (km)", 0.5, 20.0, value=rv['Distance (km)'], step=0.1)
    traffic = st.selectbox("Tingkat Kemacetan", [0, 1, 2, 3], format_func=lambda x: ["Rendah", "Sedang", "Padat", "Macet"][x], index=rv['Traffic Level'])
    weekend = st.selectbox("Apakah Akhir Pekan?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak", index=rv['Is Weekend'])
    density = st.selectbox("Kepadatan Topping", [round(i * 0.1, 1) for i in range(11)], index=int(rv['Topping Density'] * 10))
    est_dur = st.selectbox("Estimasi Waktu Sistem (menit)", list(range(5, 61, 5)), index=(rv['Estimated Duration (min)'] // 5) - 1)
    complexity = st.selectbox("Kompleksitas Pizza", [0, 1, 2], format_func=lambda x: ["Mudah", "Sedang", "Rumit"][x], index=rv['Pizza Complexity'])
    traffic_impact = st.selectbox("Dampak Lalu Lintas", [0, 1, 2, 3], index=rv['Traffic Impact'])
    order_hour = st.selectbox("Jam Pemesanan", list(range(24)), index=rv['Order Hour'])
    avg_rest_time = st.selectbox("Waktu Rata-Rata Restoran (menit)", list(range(5, 61, 5)), index=(rv['Restaurant Avg Time'] // 5) - 1)

    submitted = st.form_submit_button("Prediksi")

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
