import time
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from services.ml_service import (
    build_encoding_table,
    drop_non_feature_columns,
    evaluate_model,
    load_model,
    preprocess_data,
    save_model_pack,
    split_data,
    train_model,
)

MODEL_PATH = "models/naive_bayes_model.pkl"


@st.cache_data
def load_data(file_bytes, file_name):
    if file_name.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))
    return pd.read_excel(BytesIO(file_bytes))


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def render_table(df, caption, filename, key):
    st.dataframe(df, use_container_width=True)
    st.caption(caption)
    st.download_button(
        label=f"Download CSV: {filename}",
        data=df_to_csv_bytes(df),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def render_figure(fig, caption, filename, key):
    st.pyplot(fig, use_container_width=True)
    st.caption(caption)
    st.download_button(
        label=f"Download PNG: {filename}",
        data=fig_to_png_bytes(fig),
        file_name=filename,
        mime="image/png",
        key=key,
    )
    plt.close(fig)


def decode_labels(values, encoder):
    if encoder is None or not hasattr(encoder, "inverse_transform"):
        return np.array(values)
    return encoder.inverse_transform(np.array(values))


st.title("Pipeline Report (Streamlit Output)")
st.write("Gunakan halaman ini untuk melihat bukti setiap tahap pipeline 4.1 sampai 4.8.")

uploaded_file = st.file_uploader("Upload dataset (Excel/CSV)", type=["csv", "xlsx"])
if uploaded_file:
    df_raw = load_data(uploaded_file.getvalue(), uploaded_file.name)
    st.session_state["uploaded_df"] = df_raw
    st.session_state["uploaded_filename"] = uploaded_file.name
elif "uploaded_df" in st.session_state:
    df_raw = st.session_state["uploaded_df"]
else:
    st.info("Upload dataset terlebih dahulu untuk menampilkan pipeline report.")
    st.stop()

df_clean, dropped_cols = drop_non_feature_columns(df_raw)
columns_list = list(df_clean.columns)

default_target = st.session_state.get("target_col")
default_index = columns_list.index(default_target) if default_target in columns_list else len(columns_list) - 1
target_col = st.selectbox("Pilih kolom target untuk report:", columns_list, index=default_index)
st.session_state["target_col"] = target_col

data_id = (df_clean.shape, tuple(columns_list), target_col)
if st.session_state.get("pipeline_data_id") != data_id:
    st.session_state["pipeline_data_id"] = data_id
    st.session_state.pop("pipeline_training", None)

X, y, encoders = preprocess_data(df_clean, target_col)
feature_cols = list(X.columns)
encoded_df = X.copy()
encoded_df[target_col] = y

target_encoder = encoders.get(target_col)
if target_encoder is not None and hasattr(target_encoder, "classes_"):
    class_labels = list(target_encoder.classes_)
else:
    class_labels = sorted(np.unique(y).tolist())


with st.expander("4.1 Load Dataset", expanded=True):
    st.subheader("4.1 Load Dataset")
    st.write(f"Rows: {df_clean.shape[0]} | Columns: {df_clean.shape[1]}")

    columns_summary = pd.DataFrame({
        "Column": df_clean.columns,
        "Dtype": [str(t) for t in df_clean.dtypes],
    })
    render_table(
        columns_summary,
        "Table 4.1 Dataset Columns Summary",
        "table_4_1_columns_summary.csv",
        "table_4_1_columns_summary",
    )

    render_table(
        df_clean.head(10),
        "Figure 4.1 Dataset Preview (first 10 rows)",
        "figure_4_1_dataset_preview.csv",
        "figure_4_1_dataset_preview",
    )

    target_counts = (
        df_clean[target_col]
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="Count")
    )
    render_table(
        target_counts,
        "Table 4.2 Target Class Distribution (Counts)",
        "table_4_2_target_distribution.csv",
        "table_4_2_target_distribution",
    )

    fig, ax = plt.subplots()
    ax.bar(target_counts["Class"].astype(str), target_counts["Count"], color=["#6baed6", "#fb6a4a"])
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Target Class Distribution")
    render_figure(
        fig,
        "Figure 4.2 Target Class Distribution (Ya vs Tidak)",
        "figure_4_2_target_distribution.png",
        "figure_4_2_target_distribution",
    )


with st.expander("4.2 Data Preparation (Preprocessing)", expanded=True):
    st.subheader("4.2 Data Preparation (Preprocessing)")

    missing_values = df_clean.isna().sum().reset_index()
    missing_values.columns = ["Column", "Missing_Values"]
    total_missing = int(missing_values["Missing_Values"].sum())
    duplicate_count = int(df_clean.duplicated().sum())

    render_table(
        missing_values,
        "Figure 4.3 Missing Values & Duplicates Check Output",
        "figure_4_3_missing_values.csv",
        "figure_4_3_missing_values",
    )
    st.write(f"Total missing values: {total_missing}")
    st.write(f"Duplicate rows count: {duplicate_count}")
    st.write(f"Drop kolom non-feature: {', '.join(dropped_cols) if dropped_cols else 'Tidak ada'}")

    encoding_table = build_encoding_table(encoders)
    if not encoding_table.empty:
        render_table(
            encoding_table,
            "Table 4.3 Encoding Mapping (LabelEncoder)",
            "table_4_3_encoding_mapping.csv",
            "table_4_3_encoding_mapping",
        )

    render_table(
        encoded_df.head(10),
        "Figure 4.4 Encoded Dataset Preview (10 rows)",
        "figure_4_4_encoded_preview.csv",
        "figure_4_4_encoded_preview",
    )


with st.expander("4.3 Feature Extraction/Selection", expanded=True):
    st.subheader("4.3 Feature Extraction/Selection")
    st.write(f"Fitur X (total {len(feature_cols)}): {', '.join(feature_cols)}")
    st.write(f"Target y: {target_col}")

    diagram_text = f"Dataset -> pilih {len(feature_cols)} fitur -> X\\nDataset -> pilih {target_col} -> y"
    st.code(diagram_text, language="text")
    st.caption("Figure 4.5 X/Y Split Diagram")

    selected_df = df_clean[feature_cols + [target_col]]
    render_table(
        selected_df.head(10),
        "Figure 4.6 Feature-Selected Dataset Preview (10 rows)",
        "figure_4_6_feature_selected_preview.csv",
        "figure_4_6_feature_selected_preview",
    )


with st.expander("4.4 Split 80:20 (Training/Testing)", expanded=True):
    st.subheader("4.4 Split 80:20 (Training/Testing)")

    X_train, X_test, y_train, y_test = split_data(X, y)
    st.write(f"Train size: {len(X_train)} rows")
    st.write(f"Test size: {len(X_test)} rows")

    y_train_labels = decode_labels(y_train, target_encoder)
    y_test_labels = decode_labels(y_test, target_encoder)

    train_counts = pd.Series(y_train_labels).value_counts()
    test_counts = pd.Series(y_test_labels).value_counts()
    labels = sorted(set(train_counts.index).union(test_counts.index))

    dist_df = pd.DataFrame({
        "Class": labels,
        "Train": [int(train_counts.get(lbl, 0)) for lbl in labels],
        "Test": [int(test_counts.get(lbl, 0)) for lbl in labels],
    })
    render_table(
        dist_df,
        "Table 4.4 Class Distribution (Train vs Test)",
        "table_4_4_class_distribution.csv",
        "table_4_4_class_distribution",
    )

    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, dist_df["Train"], width, label="Train")
    ax.bar(x + width / 2, dist_df["Test"], width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution: Train vs Test")
    ax.legend()
    render_figure(
        fig,
        "Figure 4.8 Class Distribution: Train vs Test",
        "figure_4_8_train_test_distribution.png",
        "figure_4_8_train_test_distribution",
    )

    split_code = (
        "X_train, X_test, y_train, y_test = train_test_split(\\n"
        "    X, y, test_size=0.2, random_state=42, stratify=y\\n"
        ")"
    )
    st.code(split_code, language="python")
    st.caption("Figure 4.9 train_test_split Snippet")


with st.expander("4.5 Model Design", expanded=True):
    st.subheader("4.5 Model Design")
    st.write(
        "CategoricalNB dipilih karena semua fitur bersifat kategorikal diskrit "
        "dan model ini menghitung peluang berbasis frekuensi kategori."
    )

    nb_table = pd.DataFrame(
        [
            {"Variant": "GaussianNB", "Data Type": "Numerik kontinu", "Catatan": "Distribusi normal"},
            {"Variant": "MultinomialNB", "Data Type": "Count/frekuensi", "Catatan": "Teks, counts"},
            {"Variant": "BernoulliNB", "Data Type": "Biner", "Catatan": "0/1"},
            {"Variant": "CategoricalNB", "Data Type": "Kategorikal", "Catatan": "Fitur diskrit"},
        ]
    )
    render_table(
        nb_table,
        "Table 4.5 Perbandingan Varian Naive Bayes",
        "table_4_5_nb_variants.csv",
        "table_4_5_nb_variants",
    )

    flow_text = "Input fitur -> prior -> likelihood -> posterior -> pilih kelas"
    st.code(flow_text, language="text")
    st.caption("Figure 4.10 Naive Bayes Design Flow")


with st.expander("4.6 Model Training", expanded=True):
    st.subheader("4.6 Model Training")

    if st.button("Train Model"):
        start_time = time.perf_counter()
        model = train_model(X_train, y_train)
        y_pred, acc, cm, report = evaluate_model(model, X_test, y_test)
        duration = time.perf_counter() - start_time

        pack = save_model_pack(model, encoders, target_col, feature_cols, MODEL_PATH)
        reloaded, load_error = load_model(MODEL_PATH)
        if not load_error:
            pack = reloaded

        st.session_state["pipeline_training"] = {
            "model_pack": pack,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "acc": acc,
            "cm": cm,
            "report": report,
            "duration": duration,
        }
        st.session_state["evaluation"] = {
            "acc": acc,
            "cm": cm.tolist(),
            "report": report,
            "target_col": target_col,
        }

    training_state = st.session_state.get("pipeline_training")
    if not training_state:
        st.warning("Model belum dilatih, silakan training dulu.")
    else:
        st.success("Training selesai.")
        st.write(f"Model saved to: {MODEL_PATH}")
        st.write(f"Training time: {training_state['duration']:.2f} seconds")
        st.caption("Figure 4.13 Training Output / Model Saved")


with st.expander("4.7 Model Testing", expanded=True):
    st.subheader("4.7 Model Testing")
    training_state = st.session_state.get("pipeline_training")
    if not training_state:
        st.warning("Model belum dilatih, silakan training dulu.")
    else:
        st.write("Contoh hasil testing sebelum evaluasi.")
        y_test_labels = decode_labels(training_state["y_test"], target_encoder)
        y_pred_labels = decode_labels(training_state["y_pred"], target_encoder)
        sample_size = min(10, len(y_test_labels))
        sample_df = pd.DataFrame({
            "Index": training_state["X_test"].index[:sample_size],
            "Aktual": y_test_labels[:sample_size],
            "Prediksi": y_pred_labels[:sample_size],
        })
        sample_df["Status"] = np.where(sample_df["Aktual"] == sample_df["Prediksi"], "Benar", "Salah")

        render_table(
            sample_df,
            "Table 4.14 Sample Predictions vs Actual (10 test rows)",
            "table_4_14_sample_predictions.csv",
            "table_4_14_sample_predictions",
        )


with st.expander("4.8 Model Evaluation", expanded=True):
    st.subheader("4.8 Model Evaluation")
    training_state = st.session_state.get("pipeline_training")
    if not training_state:
        st.warning("Model belum dilatih, silakan training dulu.")
    else:
        y_test_labels = decode_labels(training_state["y_test"], target_encoder)
        y_pred_labels = decode_labels(training_state["y_pred"], target_encoder)
        cm_labels = class_labels

        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=cm_labels)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(len(cm_labels)))
        ax.set_yticks(np.arange(len(cm_labels)))
        ax.set_xticklabels(cm_labels)
        ax.set_yticklabels(cm_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(len(cm_labels)):
            for j in range(len(cm_labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
        fig.colorbar(im)
        render_figure(
            fig,
            "Figure 4.15 Confusion Matrix",
            "figure_4_15_confusion_matrix.png",
            "figure_4_15_confusion_matrix",
        )

        report_dict = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        report_df = (
            pd.DataFrame(report_dict)
            .transpose()
            .reset_index()
            .rename(columns={"index": "Label"})
        )
        render_table(
            report_df,
            "Table 4.16 Classification Report",
            "table_4_16_classification_report.csv",
            "table_4_16_classification_report",
        )
