import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.title("📊 Dashboard Evaluasi Model")

if "evaluation" not in st.session_state:
    st.warning("⚠️ Belum ada model yang dilatih. Silakan upload dataset dulu di menu **Upload Dataset**.")
else:
    eval_data = st.session_state["evaluation"]

    # Metric akurasi
    st.metric("Akurasi Model", f"{eval_data['acc']:.2%}")

    # Confusion Matrix
    cm = np.array(eval_data["cm"])
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    im = ax.matshow(cm, cmap="Blues")
    plt.title("Confusion Matrix", pad=20)
    fig.colorbar(im)
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    st.json(eval_data["report"])
