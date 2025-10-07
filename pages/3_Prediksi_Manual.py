import streamlit as st
import matplotlib.pyplot as plt
from services.ml_service import load_model, predict_input, save_prediction

st.title("🩺 Prediksi Manual Gejala Pasien")

# Load model & encoders
model, encoders, error = load_model()
if error:
    st.warning(error)
    st.stop()

st.subheader("📝 Input Gejala Pasien")

# helper: ambil opsi dari encoder agar pasti cocok
def opts(col, fallback):
    return list(encoders[col].classes_) if col in encoders else fallback

# form: TAMPILKAN OPSI DARI ENCODER (BUKAN HARDCODE)
input_data = {
    "Usia": st.selectbox("Usia", opts("Usia", ["Muda", "Tua"])),
    "Jenis_Kelamin": st.selectbox("Jenis Kelamin", opts("Jenis_Kelamin", ["Pria", "Wanita"])),
    "Merokok": st.selectbox("Merokok", opts("Merokok", ["Aktif", "Pasif", "Tidak"])),
    "Bekerja": st.selectbox("Bekerja", opts("Bekerja", ["Ya", "Tidak"])),
    "Rumah_Tangga": st.selectbox("Rumah Tangga", opts("Rumah_Tangga", ["Ya", "Tidak"])),
    "Aktivitas_Begadang": st.selectbox("Aktivitas Begadang", opts("Aktivitas_Begadang", ["Sering", "Jarang", "Tidak"])),
    "Aktivitas_Olahraga": st.selectbox("Aktivitas Olahraga", opts("Aktivitas_Olahraga", ["Sering", "Jarang", "Tidak"])),
    "Asuransi": st.selectbox("Asuransi", opts("Asuransi", ["Ya", "Tidak"])),
    "Penyakit_Bawaan": st.selectbox("Penyakit Bawaan", opts("Penyakit_Bawaan", ["Ya", "Tidak"]))
}

# (opsional) debug: tampilkan kelas yang dikenal encoder
with st.expander("🔧 Lihat kelas yang dikenali encoder"):
    st.json({k: list(v.classes_) for k, v in encoders.items() if hasattr(v, "classes_")})

if st.button("🔮 Prediksi"):
    try:
        hasil, proba, proba_dict, kelas = predict_input(input_data, model, encoders)
        st.success(f"✅ Hasil Prediksi: **{hasil}**")

        st.write("📊 Probabilitas:", proba_dict)

        # Visualisasi probabilitas
        st.subheader("📈 Visualisasi Probabilitas Prediksi")
        fig, ax = plt.subplots()
        ax.bar(kelas, proba)
        ax.set_ylabel("Probabilitas")
        ax.set_xlabel("Kelas")
        ax.set_title("Probabilitas Prediksi per Kelas")
        for i, v in enumerate(proba):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
        st.pyplot(fig)

        # Simpan riwayat
        file_path = save_prediction(input_data, hasil, proba_dict)
        st.info(f"📁 Hasil prediksi disimpan ke: **{file_path}**")
        with open(file_path, "rb") as f:
            st.download_button("💾 Download Semua Prediksi (CSV)", f, file_name="predictions.csv")

    except ValueError as e:
        # Pesan ramah jika masih ada label tak dikenal
        st.error(f"❌ Input tidak dikenali: {e}")
        st.info("💡 Solusi: pilih nilai sesuai daftar pada 'Lihat kelas yang dikenali encoder', atau retrain model dengan dataset yang nilainya konsisten.")
