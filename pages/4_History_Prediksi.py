import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.title("📜 History Prediksi Pasien")

file_path = "results/predictions.csv"

# === CEK FILE HISTORY ===
if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
    st.warning("⚠️ Belum ada history prediksi. Silakan lakukan prediksi manual di menu **Prediksi Manual**.")
else:
    try:
        # === LOAD DATA ===
        df = pd.read_csv(file_path)

        if df.empty:
            st.warning("⚠️ File history kosong. Silakan lakukan prediksi manual di menu **Prediksi Manual**.")
        else:
            # === METRICS RINGKAS ===
            st.subheader("📌 Ringkasan Prediksi")
            total_prediksi = len(df)
            ya_count = df["Prediksi"].value_counts().get("Ya", 0)
            tidak_count = df["Prediksi"].value_counts().get("Tidak", 0)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Prediksi", total_prediksi)
            col2.metric("Jumlah 'Ya'", ya_count)
            col3.metric("Jumlah 'Tidak'", tidak_count)

            # === TABEL RIWAYAT ===
            st.subheader("📊 Tabel Riwayat Prediksi")
            st.dataframe(df, use_container_width=True)

            # === PIE CHART DISTRIBUSI HASIL ===
            st.subheader("📈 Distribusi Hasil Prediksi")
            if "Prediksi" in df.columns:
                distribusi = df["Prediksi"].value_counts()

                fig, ax = plt.subplots()
                ax.pie(distribusi, labels=distribusi.index, autopct="%1.1f%%", 
                       startangle=90, colors=["lightcoral", "skyblue"])
                ax.axis("equal")
                st.pyplot(fig)
            else:
                st.warning("⚠️ Kolom 'Prediksi' tidak ditemukan di history.")

            # === FILTER BERDASARKAN HASIL ===
            st.subheader("🔍 Filter Data")
            hasil_filter = st.selectbox("Filter berdasarkan hasil prediksi:", ["Semua"] + df["Prediksi"].unique().tolist())
            if hasil_filter != "Semua":
                df = df[df["Prediksi"] == hasil_filter]

            # === FILTER BERDASARKAN WAKTU ===
            if "Waktu" in df.columns:
                df["Waktu"] = pd.to_datetime(df["Waktu"], errors="coerce")
                if df["Waktu"].notna().any():
                    waktu_min = df["Waktu"].min().date()
                    waktu_max = df["Waktu"].max().date()
                    start_date, end_date = st.date_input("Rentang waktu:", [waktu_min, waktu_max])

                    if isinstance(start_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp):
                        df = df[(df["Waktu"].dt.date >= start_date.date()) & (df["Waktu"].dt.date <= end_date.date())]

            # === TAMPILKAN DATA TERFILTER ===
            st.subheader("📋 Data Terfilter")
            st.dataframe(df, use_container_width=True)

            # === TOMBOL DOWNLOAD ===
            st.download_button(
                "💾 Download Riwayat (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name="history_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Gagal membaca file history: {e}")
