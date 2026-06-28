# Chili Leaf Disease API

Backend FastAPI yang menyajikan model klasifikasi penyakit daun cabai
(ResNet18 Transfer Learning) plus estimasi tingkat keparahan (segmentasi HSV
+ ambang pakar) lewat endpoint REST.

Model dihasilkan oleh notebook:
`klasifikasi_dan_estimasi_tingkat_keparahan_penyakit_daun_cabai.ipynb`

## 1. Setup

```bash
pip install -r requirements.txt
```

Letakkan file `model.pth` (hasil training dari notebook) di folder yang sama
dengan `chili_api.py`. File ini berisi checkpoint lengkap: bobot model,
nama kelas, dan metadata training.

## 2. Jalankan server

```bash
uvicorn chili_api:app --reload --port 8000
```

Server akan jalan di `http://localhost:8000`. Cek dengan buka
`http://localhost:8000/health` di browser — kalau muncul `{"status": "ok", ...}`
berarti model sudah berhasil dimuat.

## 3. Endpoint

### `GET /health`

Cek status server, model, dan metadata checkpoint.

### `POST /predict`

Kirim sebagai `multipart/form-data` dengan field `file` berisi gambar daun cabai.

Contoh dari JavaScript (fetch):

```javascript
async function predictDisease(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Gagal mendapat prediksi dari server");
  }

  return response.json();
}

// Contoh pemakaian dengan input file dari <input type="file">
const fileInput = document.querySelector("#imageInput");
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  const result = await predictDisease(file);
  console.log(result);
});
```

Contoh response (daun sakit):

```json
{
  "predicted_class": "bercak daun",
  "confidence": 0.9312,
  "probabilities": {
    "bercak daun": 0.9312,
    "daun sehat": 0.0205,
    "thrips": 0.0301,
    "virus kuning": 0.0182
  },
  "severity_percent": 32.15,
  "severity_label": "Sedang"
}
```

Contoh response (daun sehat):

```json
{
  "predicted_class": "daun sehat",
  "confidence": 0.9876,
  "probabilities": {
    "bercak daun": 0.0041,
    "daun sehat": 0.9876,
    "thrips": 0.0052,
    "virus kuning": 0.0031
  },
  "severity_percent": 0.0,
  "severity_label": "Sehat"
}
```

### Penjelasan Response

| Field | Penjelasan |
|---|---|
| `predicted_class` | Jenis penyakit hasil klasifikasi (Supervised — CNN ResNet18) |
| `confidence` | Probabilitas prediksi tertinggi (0.0–1.0) |
| `probabilities` | Probabilitas untuk semua kelas |
| `severity_percent` | Persentase area daun yang terinfeksi (0–100%) — dihitung dari segmentasi HSV |
| `severity_label` | Tingkat keparahan: **Sehat**, **Ringan** (0–25%), **Sedang** (25–50%), **Berat** (50–75%), **Sangat Berat** (75–100%) |

## 4. Skala Keparahan (Referensi Pakar)

Keparahan diestimasi berdasarkan rasio area lesi terhadap area daun menggunakan
segmentasi warna HSV (bukan K-Means per gambar). Rujukan skala:

| Tingkat | Rasio Area Terinfeksi | Referensi |
|---|---|---|
| Sehat | 0% | — |
| Ringan | >0–25% | James (1971) |
| Sedang | >25–50% | James (1971) |
| Berat | >50–75% | James (1971) |
| Sangat Berat | >75% | James (1971) |

## 5. Catatan

- `model.pth` sudah menyimpan nama kelas di dalamnya — tidak perlu file
  `class_names.json` terpisah.
- Arsitektur model di `chili_api.py` (`build_model()`) harus identik dengan
  yang dipakai saat training di notebook (ResNet18 + Linear head).
- CORS sudah dibuka untuk semua origin (`allow_origins=["*"]`), jadi frontend
  dari domain/port berbeda tidak akan kena error CORS.
- Severity hanya dihitung untuk kelas sakit. Kelas sehat otomatis mendapat
  `severity_label: "Sehat"` dan `severity_percent: 0.0`.
