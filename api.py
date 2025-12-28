# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np

from services.ml_service import load_model, predict_input

model, encoders, error = load_model()
if error:
    raise RuntimeError(error)

app = FastAPI(title="Prediksi Paru API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000", "http://localhost:8000",
        "http://127.0.0.1:5173", "http://localhost:5173",
        "http://192.168.1.19:8000", "http://192.168.1.19:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientInput(BaseModel):
    Usia: str
    Jenis_Kelamin: str
    Merokok: str
    Bekerja: str
    Rumah_Tangga: str
    Aktivitas_Begadang: str
    Aktivitas_Olahraga: str
    Asuransi: str
    Penyakit_Bawaan: str

@app.post("/predict")
def predict(data: PatientInput):
    try:
        inp = data.model_dump()
        hasil, proba, proba_dict, kelas = predict_input(inp, model, encoders)

        # --- CAST ke tipe Python murni ---
        proba_py  = [float(x) for x in (proba.tolist() if hasattr(proba, "tolist") else proba)]
        kelas_py  = [str(x)   for x in (kelas.tolist() if hasattr(kelas, "tolist") else kelas)]
        proba_map = {str(k): float(v) for k, v in proba_dict.items()}
        hasil_py  = str(hasil)

        return JSONResponse(content={
            "prediction": hasil_py,
            "proba": proba_py,
            "classes": kelas_py,
            "proba_dict": proba_map
        })

    except ValueError as e:
        # misal label tak dikenal / tidak ada di encoder
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
