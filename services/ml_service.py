from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import pandas as pd
from datetime import datetime

def preprocess_data(df, target_col):
    """
    Melakukan encoding data kategorikal & memisahkan fitur dan target.
    """

    # --- NORMALISASI NAMA KOLOM & DROP KOLOM NON-FEATURE ---
    df = df.copy()
    # rapikan nama kolom
    df.columns = [str(c).strip() for c in df.columns]
    # buang kolom-kolom ID/nomor yang tidak informatif
    drop_cols = [c for c in ["No", "ID", "Id", "index"] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # (opsional) normalisasi teks seperti sebelumnya kalau kamu pakai
    # df = _normalize_df(df)

    # --- PISAH FITUR & TARGET ---
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # --- ENCODING ---
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    if y.dtype == "object":
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))
        label_encoders[target_col] = le_target

    return X, y, label_encoders

def train_and_evaluate(df, target_col, model_path="models/naive_bayes_model.pkl"):
    """
    Training Naïve Bayes dan evaluasi dengan train-test split.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    X, y, encoders = preprocess_data(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = CategoricalNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    joblib.dump((model, encoders), model_path)

    return acc, cm, report

def load_model(model_path="models/naive_bayes_model.pkl"):
    """
    Memuat model Naïve Bayes beserta encoder.
    """
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        return None, None, "⚠️ Model belum ada atau file kosong. Silakan latih ulang model."
    try:
        model, encoders = joblib.load(model_path)
        return model, encoders, None
    except Exception as e:
        return None, None, f"❌ Gagal memuat model: {e}"

def predict_input(input_dict, model, encoders):
    """
    Melakukan prediksi dari input pasien (dict).
    Mengembalikan hasil prediksi (label), probabilitas, dan dict probabilitas.
    """
    df_input = pd.DataFrame([input_dict])

    # Encode sesuai encoder hasil training
    for col in df_input.columns:
        if col in encoders:
            le = encoders[col]
            df_input[col] = le.transform(df_input[col])

    # Prediksi
    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]

    # Decode hasil prediksi
    target_col = list(encoders.keys())[-1]
    hasil = encoders[target_col].inverse_transform([pred])[0]
    kelas = encoders[target_col].classes_
    proba_dict = {kelas[i]: float(proba[i]) for i in range(len(kelas))}

    return hasil, proba, proba_dict, kelas

def save_prediction(input_dict, hasil, proba_dict, file_path="results/predictions.csv"):
    """
    Menyimpan hasil prediksi pasien ke file CSV.
    Jika file belum ada → buat baru dengan header.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Tambahkan hasil prediksi & timestamp ke input
    record = input_dict.copy()
    record["Prediksi"] = hasil
    record.update({f"Prob_{k}": v for k, v in proba_dict.items()})
    record["Waktu"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_new = pd.DataFrame([record])

    if not os.path.exists(file_path):
        df_new.to_csv(file_path, index=False)
    else:
        df_new.to_csv(file_path, mode="a", index=False, header=False)

    return file_path
