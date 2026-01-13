import streamlit as st
from utils.helpers import load_dataset
from services.ml_service import train_and_evaluate
import os
import requests

st.title("📂 Upload Dataset & Training")

uploaded_file = st.file_uploader("Upload dataset (Excel/CSV)", type=["csv", "xlsx"])

if uploaded_file:
    df = load_dataset(uploaded_file)
    st.subheader("📊 Data Sample")
    st.dataframe(df.head(), use_container_width=True)

    target_col = st.selectbox("Pilih kolom target:", df.columns, index=len(df.columns)-1)

    if st.button("Latih & Simpan Model"):
        acc, cm, report = train_and_evaluate(df, target_col)

        st.success(f"✅ Model berhasil dilatih & disimpan. Akurasi: {acc:.2%}")

        # Simpan hasil evaluasi ke session_state agar bisa dipakai di dashboard
        st.session_state["evaluation"] = {
            "acc": acc,
            "cm": cm.tolist(),
            "report": report,
            "target_col": target_col
        }

        api_url = os.getenv("API_URL", "http://127.0.0.1:8001")
        try:
            resp = requests.post(f"{api_url}/reload-model", timeout=5)
            if resp.ok:
                st.info("Model API berhasil di-reload.")
            else:
                st.warning(f"Gagal reload model API. Status: {resp.status_code}")
        except requests.RequestException as e:
            st.warning(f"Gagal menghubungi API untuk reload model: {e}")

        model_path = "models/naive_bayes_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                st.download_button(
                    label="💾 Download Model",
                    data=f,
                    file_name="naive_bayes_model.pkl"
                )
