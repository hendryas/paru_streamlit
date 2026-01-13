import time
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
ACC_GOOD_THRESHOLD = 0.90
ACC_OK_THRESHOLD = 0.80


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


def resolve_positive_label(labels, positive_label):
    if not labels:
        return positive_label
    if positive_label in labels:
        return positive_label
    if len(labels) == 2:
        return labels[-1]
    return labels[0]


def get_label_metrics(report_dict, label):
    if label in report_dict:
        return report_dict[label]
    label_key = str(label)
    if label_key in report_dict:
        return report_dict[label_key]
    return {}


def compute_metrics(y_true, y_pred, labels=None, positive_label="Ya"):
    if labels is None:
        labels = sorted(set(y_true).union(set(y_pred)))
    labels = list(labels)
    positive_label = resolve_positive_label(labels, positive_label)

    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .reset_index()
        .rename(columns={"index": "Label"})
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tp_tn_fp_fn = {}
    if cm.shape == (2, 2) and positive_label in labels:
        pos_idx = labels.index(positive_label)
        neg_idx = 1 - pos_idx
        tp_tn_fp_fn = {
            "tp": int(cm[pos_idx, pos_idx]),
            "fn": int(cm[pos_idx, neg_idx]),
            "fp": int(cm[neg_idx, pos_idx]),
            "tn": int(cm[neg_idx, neg_idx]),
        }

    return {
        "accuracy": float(accuracy),
        "report_dict": report_dict,
        "report_df": report_df,
        "cm": cm,
        "labels": labels,
        "positive_label": positive_label,
        "tp_tn_fp_fn": tp_tn_fp_fn,
    }


def performance_label(acc):
    if acc >= ACC_GOOD_THRESHOLD:
        return "baik"
    if acc >= ACC_OK_THRESHOLD:
        return "cukup baik"
    return "masih perlu peningkatan"


def build_discussion_text(acc, report_dict, cm, labels, positive_label="Ya"):
    perf = performance_label(acc)
    acc_pct = acc * 100
    pos_label = resolve_positive_label(labels, positive_label)
    pos_metrics = get_label_metrics(report_dict, pos_label)
    recall_pos = pos_metrics.get("recall")
    precision_pos = pos_metrics.get("precision")

    sentences = [
        f"Secara umum, performa model {perf} dengan akurasi {acc_pct:.2f}%.",
    ]

    if recall_pos is not None:
        recall_pct = recall_pos * 100
        precision_pct = (precision_pos or 0.0) * 100
        if recall_pos >= 0.85:
            sentences.append(
                f"Model cukup efektif mendeteksi kasus kelas {pos_label} "
                f"(recall {recall_pct:.2f}%, precision {precision_pct:.2f}%)."
            )
        else:
            sentences.append(
                f"Recall kelas {pos_label} masih terbatas "
                f"(recall {recall_pct:.2f}%), sehingga deteksi kasus terindikasi perlu perhatian."
            )
    else:
        sentences.append(
            f"Recall untuk kelas {pos_label} belum tersedia, sehingga interpretasi sensitivitas terbatas."
        )

    if cm is not None and cm.shape == (2, 2) and pos_label in labels:
        pos_idx = labels.index(pos_label)
        neg_idx = 1 - pos_idx
        tp = int(cm[pos_idx, pos_idx])
        fn = int(cm[pos_idx, neg_idx])
        fp = int(cm[neg_idx, pos_idx])
        total_pos = tp + fn
        fn_rate = (fn / total_pos) if total_pos else 0.0
        if fn > fp or fn_rate > 0.15:
            sentences.append(
                "Jumlah false negative relatif tinggi sehingga ada potensi kasus positif yang terlewat."
            )
        else:
            sentences.append(
                "False negative relatif terkendali sehingga risiko kasus positif terlewat lebih rendah."
            )
    else:
        sentences.append(
            "Confusion matrix multi-kelas atau tidak lengkap, sehingga analisis FN/FP biner tidak dilakukan."
        )

    sentences.append(
        "Perbaikan dapat dilakukan melalui penyesuaian fitur, penambahan data, dan validasi yang lebih kuat."
    )
    return " ".join(sentences)


def compute_cleaning_summary(df_before, df_after, id_col_candidates=None):
    if id_col_candidates is None:
        id_col_candidates = ["No", "no", "NO"]

    def has_id_column(df):
        cols = [str(c).strip().lower() for c in df.columns]
        return any(str(candidate).strip().lower() in cols for candidate in id_col_candidates)

    has_before = has_id_column(df_before)
    has_after = has_id_column(df_after)

    if not has_before and not has_after:
        no_before = "N/A"
        no_after = "N/A"
    else:
        no_before = "Yes" if has_before else "No"
        no_after = "Yes" if has_after else "No"

    rows = [
        {
            "Component": "Total records",
            "Before Cleaning": int(len(df_before)),
            "After Cleaning": int(len(df_after)),
        },
        {
            "Component": "Total columns",
            "Before Cleaning": int(df_before.shape[1]),
            "After Cleaning": int(df_after.shape[1]),
        },
        {
            "Component": "Missing values (total)",
            "Before Cleaning": int(df_before.isna().sum().sum()),
            "After Cleaning": int(df_after.isna().sum().sum()),
        },
        {
            "Component": "Duplicate rows",
            "Before Cleaning": int(df_before.duplicated().sum()),
            "After Cleaning": int(df_after.duplicated().sum()),
        },
        {
            "Component": "\"No\" column used as feature (Yes/No)",
            "Before Cleaning": no_before,
            "After Cleaning": no_after,
        },
    ]

    return pd.DataFrame(rows)


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
    st.warning("Dataset belum tersedia, silakan upload/load dataset terlebih dahulu.")
    st.stop()

df_before = df_raw.copy()
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

    summary_df = compute_cleaning_summary(df_before, df_clean)
    st.markdown("### Table 4.3 Dataset Summary: Before vs After Cleaning")
    st.dataframe(summary_df, use_container_width=True)
    st.caption("Summary computed from raw dataset vs cleaned dataset (duplicates/missing handled, ID columns removed).")

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
        if "y_test" not in training_state or "y_pred" not in training_state:
            st.warning(
                "Belum ada hasil prediksi untuk evaluasi. Jalankan Training + Testing terlebih dahulu."
            )
            st.stop()

        y_test_labels = decode_labels(training_state["y_test"], target_encoder)
        y_pred_labels = decode_labels(training_state["y_pred"], target_encoder)
        cm_labels = class_labels

        metrics = compute_metrics(
            y_test_labels,
            y_pred_labels,
            labels=cm_labels,
            positive_label="Ya",
        )
        st.session_state["eval_metrics"] = metrics

        acc_pct = metrics["accuracy"] * 100
        pos_label = metrics["positive_label"]
        pos_metrics = get_label_metrics(metrics["report_dict"], pos_label)
        precision_pos = pos_metrics.get("precision", 0.0) * 100
        recall_pos = pos_metrics.get("recall", 0.0) * 100
        f1_pos = pos_metrics.get("f1-score", 0.0) * 100

        metric_cols = st.columns(4)
        metric_cols[0].metric("Accuracy", f"{acc_pct:.2f}%")
        metric_cols[1].metric(f"Precision ({pos_label})", f"{precision_pos:.2f}%")
        metric_cols[2].metric(f"Recall ({pos_label})", f"{recall_pos:.2f}%")
        metric_cols[3].metric(f"F1 ({pos_label})", f"{f1_pos:.2f}%")

        st.write(
            f"Berdasarkan hasil pengujian, model memperoleh nilai akurasi sebesar {acc_pct:.2f}%."
        )
        st.write(
            f"Performa model pada data uji dikategorikan: {performance_label(metrics['accuracy'])}."
        )

        st.markdown("#### 4.8.4 Discussion Summary")
        discussion_text = build_discussion_text(
            metrics["accuracy"],
            metrics["report_dict"],
            metrics["cm"],
            metrics["labels"],
            positive_label="Ya",
        )
        st.write(discussion_text)

        cm = metrics["cm"]
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

        render_table(
            metrics["report_df"],
            "Table 4.16 Classification Report",
            "table_4_16_classification_report.csv",
            "table_4_16_classification_report",
        )
