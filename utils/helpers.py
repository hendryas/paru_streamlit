import pandas as pd

def load_dataset(uploaded_file):
    """Membaca dataset dari CSV/XLSX"""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)
