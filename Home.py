import streamlit as st

st.set_page_config(page_title="Prediksi Penyakit Paru-Paru", layout="wide")

st.title("🫁 Aplikasi Prediksi Penyakit Paru-Paru")
st.markdown("""
Selamat datang di aplikasi prediksi penyakit paru-paru menggunakan **Naïve Bayes**.

📂 Menu:
- **Upload Dataset** → untuk mengunggah data & melatih model.
- **Dashboard** → untuk melihat hasil evaluasi model (akurasi, confusion matrix, laporan).
""")
