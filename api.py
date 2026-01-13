# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import threading

from services.ml_service import load_model, predict_input

model_lock = threading.Lock()
pack = None
model = None
encoders = None
target_col = None
feature_cols = None

def _set_pack(new_pack):
    global pack, model, encoders, target_col, feature_cols
    pack = new_pack
    if new_pack:
        model = new_pack["model"]
        encoders = new_pack["encoders"]
        target_col = new_pack["target_col"]
        feature_cols = new_pack["feature_cols"]
    else:
        model = None
        encoders = None
        target_col = None
        feature_cols = None

def _pack_info(active_pack):
    if not active_pack:
        return {"classes": [], "feature_cols": [], "target_col": None}
    enc = active_pack.get("encoders", {})
    target = active_pack.get("target_col")
    if target in enc:
        classes = [str(x) for x in enc[target].classes_]
    else:
        classes = [str(x) for x in getattr(active_pack["model"], "classes_", [])]
    return {
        "classes": classes,
        "feature_cols": active_pack.get("feature_cols", []),
        "target_col": target
    }

initial_pack, load_error = load_model()
if load_error:
    print(load_error)
else:
    _set_pack(initial_pack)

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
        with model_lock:
            if pack is None:
                raise HTTPException(
                    status_code=500,
                    detail="Model belum ada. Latih model dan panggil /reload-model."
                )
            active_pack = pack
        inp = data.model_dump()
        hasil, proba, proba_dict, kelas = predict_input(inp, active_pack)

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

    except HTTPException:
        raise
    except ValueError as e:
        # misal label tak dikenal / tidak ada di encoder
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

@app.post("/reload-model")
def reload_model():
    new_pack, error = load_model()
    if error:
        raise HTTPException(status_code=500, detail=error)
    with model_lock:
        _set_pack(new_pack)
        info = _pack_info(pack)
    return JSONResponse(content={
        "status": "ok",
        **info
    })
