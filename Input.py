import streamlit as st
import pandas as pd
import joblib
import random

# Load model
clf = joblib.load("model_is_delayed.pkl")
reg_delay = joblib.load("model_delay.pkl")
reg_eta = joblib.load("model_eta.pkl")

st.title("Prediksi Pengiriman Pizza 🍕")

# Fitur dari setiap model
features_eta = ['Order Month', 'Pizza Size', 'Pizza Type', 'Toppings Count', 'Distance (km)', 'Traffic Level', 'Order Hour']
features_delay = ['Pizza Complexity', 'Restaurant Avg Time', 'Toppings Count', 'Topping Density', 'Traffic Level', 'Is Weekend', 'Order Hour']
features_class = ['Distance (km)', 'Traffic Level', 'Is Weekend', 'Order Hour', 'Pizza Complexity']

# Randomizer
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
        'Pizza Complexity': random.randint(0, 2),
        'Order Hour': random.randint(0, 23),
        'Restaurant Avg Time': random.choice(list(range(5, 61, 5))),
    }

# Simpan state random
if 'random_values' not in st.session_state:
    st.session_state.random_values = generate_random_input()

# Tombol random
if st.button("🎲 Isi Otomatis (Random)"):
    st.session_state.random_values = generate_random_input()

# Form Input
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
    complexity = st.selectbox("Kompleksitas Pizza", [0, 1, 2], format_func=lambda x: ["Mudah", "Sedang", "Rumit"][x], index=rv['Pizza Complexity'])
    order_hour = st.selectbox("Jam Pemesanan", list(range(24)), index=rv['Order Hour'])
    avg_rest_time = st.selectbox("Waktu Rata-Rata Restoran (menit)", list(range(5, 61, 5)), index=(rv['Restaurant Avg Time'] // 5) - 1)

    submitted = st.form_submit_button("Prediksi")

# Prediksi
if submitted:
    # Gabungkan semua input ke satu DataFrame
    full_input = pd.DataFrame([{
        'Order Month': month,
        'Pizza Size': pizza_size,
        'Pizza Type': pizza_type,
        'Toppings Count': toppings,
        'Distance (km)': distance,
        'Traffic Level': traffic,
        'Is Weekend': weekend,
        'Topping Density': density,
        'Pizza Complexity': complexity,
        'Order Hour': order_hour,
        'Restaurant Avg Time': avg_rest_time
    }])

    # Subset sesuai model
    input_eta = full_input[features_eta]
    input_delay = full_input[features_delay]
    input_class = full_input[features_class]

    # Prediksi
    pred_eta = reg_eta.predict(input_eta)[0]
    pred_delay = reg_delay.predict(input_delay)[0]
    is_delayed = clf.predict(input_class)[0]

    # Output
    st.subheader("Hasil Prediksi 📊")
    st.write(f"🕒 **Estimasi Pengiriman:** {pred_eta:.2f} menit")
    st.write(f"⏱️ **Prediksi Keterlambatan:** {pred_delay:.2f} menit")
    st.write(f"⚠️ **Kemungkinan Telat:** {'YA' if is_delayed else 'TIDAK'}")
