"""
Chili Leaf Disease API
======================
Backend FastAPI untuk menyajikan model klasifikasi penyakit daun cabai
(ResNet18 Transfer Learning) plus estimasi keparahan berbasis segmentasi HSV
& ambang pakar lewat endpoint REST.

Model checkpoint dihasilkan oleh notebook:
  klasifikasi_dan_estimasi_tingkat_keparahan_penyakit_daun_cabai.ipynb

Cara jalanin:
    pip install -r requirements.txt
    # Pastikan file model.pth ada di folder yang sama.
    # model.pth dihasilkan otomatis saat notebook selesai training.
    uvicorn chili_api:app --reload --port 8000
"""

import io
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ==========================================
# KONFIGURASI
# ==========================================
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Statistik normalisasi ImageNet (sama persis dengan notebook)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Ambang keparahan berdasarkan referensi pakar (James 1971; skala gejala virus cabai)
# Diseased ratio (0..1) -> label keparahan
SEVERITY_BINS = [0.0, 0.25, 0.50, 0.75, 1.01]
SEVERITY_NAMES = ["Ringan", "Sedang", "Berat", "Sangat Berat"]

# Kelas sehat dikecualikan dari estimasi keparahan
HEALTHY_KEYS = ("healthy", "sehat", "normal")

# ==========================================
# INISIALISASI MODEL (ResNet18)
# ==========================================
print(f"Mempersiapkan arsitektur ResNet18 dari '{MODEL_PATH}'...")
model = None
CLASS_NAMES = []
CHECKPOINT_META = {}


def build_model(num_classes):
    """Bangun arsitektur ResNet18 dengan head sesuai jumlah kelas (sama dengan notebook)."""
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


try:
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # Checkpoint baru dari notebook berformat dict dengan key 'model_state', 'class_names', dll.
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        CLASS_NAMES = ckpt["class_names"]
        model = build_model(len(CLASS_NAMES))
        model.load_state_dict(ckpt["model_state"])
        CHECKPOINT_META = {
            "epoch": ckpt.get("epoch"),
            "val_acc": round(ckpt.get("val_acc", 0), 4),
            "val_loss": round(ckpt.get("val_loss", 0), 4),
            "img_size": ckpt.get("img_size", 224),
        }
        print(f"Checkpoint dimuat (epoch {CHECKPOINT_META['epoch']}, "
              f"val_acc {CHECKPOINT_META['val_acc']:.4f})")
    else:
        # Fallback: format lama (raw state_dict) — butuh class_names.json terpisah
        fallback_path = os.environ.get("CLASS_NAMES_PATH", "class_names.json")
        with open(fallback_path, "r", encoding="utf-8") as f:
            CLASS_NAMES = json.load(f)
        model = build_model(len(CLASS_NAMES))
        state_dict = ckpt if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict)
        print(f"Model dimuat (format lama, class_names dari '{fallback_path}')")

    model.to(device)
    model.eval()
    print(f"Model siap di {device} | {len(CLASS_NAMES)} kelas: {CLASS_NAMES}")

except FileNotFoundError:
    print(f"File '{MODEL_PATH}' tidak ditemukan. "
          "Letakkan file model.pth (dari notebook) di folder yang sama dengan chili_api.py.")
    model = None
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None

# Transformasi gambar saat inferensi (tanpa augmentasi, sama dengan eval_transform di notebook)
IMG_SIZE = CHECKPOINT_META.get("img_size", 224)
inference_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ==========================================
# FUNGSI ESTIMASI KEPARAHAN (HSV Segmentasi)
# ==========================================
def compute_diseased_ratio(img_rgb: np.ndarray, predicted_class: str = "") -> float:
    """
    Hitung rasio area sakit terhadap total area daun menggunakan segmentasi warna HSV.
    Warna (HSV Mask) yang dicari akan disesuaikan secara dinamis berdasarkan `predicted_class`.

    - Hijau (green) = daun sehat
    - Coklat (brown) / Kuning (yellow) = lesi/area sakit umum
    - Putih (white) = area sakit untuk penyakit White Spot
    - Leaf area = green + diseased (background diabaikan)

    Returns:
        Rasio 0.0-1.0. Mengembalikan 0.0 jika area daun tidak terdeteksi.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 1. Base Mask (Daun Sehat)
    green = cv2.inRange(hsv, (25, 40, 40), (95, 255, 255))
    
    # 2. Disease Masks
    brown = cv2.inRange(hsv, (5, 40, 30), (25, 255, 220))
    yellow = cv2.inRange(hsv, (26, 60, 60), (35, 255, 255))
    # Putih: hue bebas, saturasi rendah (<60), kecerahan tinggi (>150)
    white = cv2.inRange(hsv, (0, 0, 150), (180, 60, 255))

    # 3. Pilih Mask Berdasarkan Kelas Prediksi
    pred_lower = predicted_class.lower()
    
    if "white" in pred_lower or "putih" in pred_lower:
        # Penyakit White Spot (bercak abu/putih)
        diseased = white
    elif "curl" in pred_lower or "nutrition" in pred_lower:
        # Virus keriting atau defisiensi nutrisi sering didominasi warna kuning (klorosis)
        diseased = yellow
    elif "cercospora" in pred_lower or "bacterial" in pred_lower:
        # Bercak nekrotik dengan/tanpa halo kuning
        diseased = cv2.bitwise_or(brown, yellow)
    else:
        # Fallback: gunakan kombinasi coklat, kuning, dan putih
        diseased_by = cv2.bitwise_or(brown, yellow)
        diseased = cv2.bitwise_or(diseased_by, white)

    leaf = cv2.bitwise_or(green, diseased)  # daun = sehat + lesi (abaikan background)

    leaf_area = int(np.count_nonzero(leaf))
    if leaf_area == 0:
        return 0.0

    return int(np.count_nonzero(diseased)) / leaf_area


def get_severity_label(ratio: float) -> str:
    """
    Petakan diseased_ratio ke label keparahan berdasarkan ambang pakar.
    Rujukan: James (1971); skala gejala virus cabai.
      Ringan:       0-25%
      Sedang:      25-50%
      Berat:       50-75%
      Sangat Berat: 75-100%
    """
    if ratio <= 0.0:
        return "Sehat"
    for i, (lo, hi) in enumerate(zip(SEVERITY_BINS[:-1], SEVERITY_BINS[1:])):
        if lo < ratio <= hi:
            return SEVERITY_NAMES[i]
    return SEVERITY_NAMES[-1]  # fallback: Sangat Berat


# ==========================================
# KONFIGURASI FASTAPI & ENDPOINTS
# ==========================================
app = FastAPI(
    title="Chili Leaf Disease API",
    version="2.0.0",
    description=(
        "API klasifikasi penyakit daun cabai (ResNet18 Transfer Learning) "
        "dan estimasi tingkat keparahan (segmentasi HSV + ambang pakar)."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint — info dasar API."""
    return {
        "message": "Chili Leaf Disease API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health")
def health():
    """Cek status server dan model."""
    return {
        "status": "ok",
        "device": str(device),
        "model_loaded": model is not None,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "severity_levels": SEVERITY_NAMES,
        "checkpoint": CHECKPOINT_META if CHECKPOINT_META else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prediksi jenis penyakit daun cabai (supervised - ResNet18)
    dan estimasi tingkat keparahan (unsupervised - segmentasi HSV + ambang pakar).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum siap dimuat.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah harus berupa gambar.")

    raw_bytes = await file.read()

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Gagal membaca gambar. Pastikan format file valid.")

    # --- Prediksi kelas penyakit (Supervised - CNN ResNet18) ---
    img_tensor = inference_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class_id = int(probs.argmax())
        confidence = float(probs[pred_class_id])
        all_probs = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}

    predicted_class = CLASS_NAMES[pred_class_id]

    # --- Estimasi keparahan (segmentasi HSV + ambang pakar) ---
    # Keparahan hanya dihitung untuk kelas sakit (bukan daun sehat)
    is_healthy = any(k in predicted_class.lower() for k in HEALTHY_KEYS)

    if is_healthy:
        severity_percent = 0.0
        severity_label = "Sehat"
    else:
        img_rgb = np.array(pil_img)
        ratio = compute_diseased_ratio(img_rgb, predicted_class)
        severity_percent = round(ratio * 100, 2)
        severity_label = get_severity_label(ratio)

    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": all_probs,
        "severity_percent": severity_percent,
        "severity_label": severity_label,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
