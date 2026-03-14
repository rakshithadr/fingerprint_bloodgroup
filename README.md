# 🩸 BloodPrint — Fingerprint Blood Group Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-powered system that predicts blood groups from fingerprint images using Deep Learning.**

[Features](#-features) · [Project Structure](#-project-structure) · [Setup](#-setup) · [Usage](#-usage) · [API](#-api-reference)

</div>

---

## 📌 Overview

BloodPrint uses **MobileNetV2 Transfer Learning** to classify fingerprint ridge patterns into one of 8 blood groups (A+, A−, B+, B−, AB+, AB−, O+, O−). The system includes a trained deep learning model, a REST API backend, and a modern web interface that supports both image upload and live camera capture.

> ⚠️ **Disclaimer:** This tool is for academic and research purposes only. It is not a substitute for clinical laboratory blood group testing.

---

## ✨ Features

- 🧠 **Deep Learning Model** — MobileNetV2 with custom classification head, trained in two phases (frozen base + fine-tuning)
- 🌐 **Web Interface** — Dark-themed UI with drag-and-drop upload and live camera/scanner support
- ⚡ **REST API** — Flask backend returning blood group, confidence score, and donation compatibility info
- 📊 **Training Notebook** — Complete Jupyter notebook with dataset exploration, training, evaluation, and result plots
- 🔬 **8 Blood Groups** — A+, A−, B+, B−, AB+, AB−, O+, O−

---

## 🗂️ Project Structure

```
fingerprint_bloodgroup/
│
├── 📓 BloodPrint_Complete_v2.ipynb   ← Training notebook (run this first)
│
├── 📁 backend/
│   ├── app.py                        ← Flask REST API
│   └── requirements.txt
│
├── 📁 frontend/
│   └── index.html                    ← Web interface
│
├── 📁 model/
│   ├── train_model.py                ← Standalone training script
│   └── requirements.txt
│
├── 📁 dataset/                       ← ⚠️ NOT included (add your own)
│   ├── A+/
│   ├── A-/
│   ├── B+/
│   ├── B-/
│   ├── AB+/
│   ├── AB-/
│   ├── O+/
│   └── O-/
│
├── 📁 model_output/                  ← ⚠️ Generated after training
│   ├── blood_group_model.keras
│   ├── blood_group_model.h5
│   ├── class_indices.json
│   ├── training_history.png
│   └── confusion_matrix.png
│
├── .gitignore
└── README.md
```

---

## 💻 System Requirements

| Component | Specification |
|-----------|--------------|
| OS | Windows 10/11, macOS, Linux |
| Python | 3.10 – 3.13 |
| RAM | 8 GB minimum (12 GB recommended) |
| GPU | Optional (NVIDIA with CUDA for faster training) |
| Storage | 2 GB free space |

---

## ⚙️ Setup

### 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fingerprint_bloodgroup.git
cd fingerprint_bloodgroup
```

### 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies

```bash
# Backend
cd backend
pip install -r requirements.txt

# Or install everything at once
pip install tensorflow==2.21.0 numpy matplotlib seaborn scikit-learn Pillow flask flask-cors tqdm opencv-python
```

### 4 — Add your dataset

Create the following folder structure inside the project:

```
dataset/
  A+/    ← fingerprint images for blood group A+
  A-/
  B+/
  B-/
  AB+/
  AB-/
  O+/
  O-/
```

Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`  
Minimum recommended: **100+ images per class**

## 📥 Download Dataset
[Download Dataset](https://drive.google.com/drive/u/0/folders/1t36P2brnNL1inw3wXNnqGZFJS8z048cR)
---

## 📥 Download Pretrained Model
[Download blood_group_model.keras](https://drive.google.com/drive/u/0/folders/1qdWmNWNn4fQslAl5fR1q1MG99saTgCYA)

Place it in the `model_output/` folder before running the backend.

## 🚀 Usage

### Option A — Jupyter Notebook (Recommended)

Open `BloodPrint_Complete_v2.ipynb` in Jupyter and run all cells in order:

```bash
pip install jupyter
jupyter notebook
```

The notebook will:
1. Install packages
2. Explore your dataset
3. Train the model (Phase 1 + Phase 2)
4. Plot accuracy, loss, and confusion matrix
5. Save model files to `model_output/`
6. Test predictions on sample images

---

### Option B — Command Line

```bash
cd model
python train_model.py --dataset ../dataset --epochs 30 --output ../model_output
```

---

### Start the Web Interface

**Step 1 — Start the backend API:**
```bash
cd backend
python app.py
```
Server starts at: `http://localhost:5000`

**Step 2 — Open the frontend:**

Open `frontend/index.html` in your browser.

> Tip: Use VS Code's **Live Server** extension for best experience.

---

## 🧠 Model Architecture

```
Input (224×224×3)
       ↓
MobileNetV2 (pretrained ImageNet, frozen in Phase 1)
       ↓
GlobalAveragePooling2D
       ↓
BatchNormalization
       ↓
Dense(512, ReLU) → Dropout(0.4)
       ↓
Dense(256, ReLU) → Dropout(0.3)
       ↓
Dense(128, ReLU) → Dropout(0.2)
       ↓
Dense(8, Softmax)  ← 8 blood group classes
```

### Training Strategy

| Phase | Base Model | Learning Rate | Purpose |
|-------|-----------|---------------|---------|
| Phase 1 | Frozen | 0.001 | Train custom head only |
| Phase 2 | Top 40 layers unfrozen | 0.00001 | Fine-tune for better accuracy |

---

## 🌐 API Reference

### `GET /health`
Check if the server is running.

**Response:**
```json
{
  "status": "ok",
  "model": "../model_output/blood_group_model.keras"
}
```

---

### `POST /predict`
Predict blood group from a fingerprint image.

**Request:** `multipart/form-data` with field `image`

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@fingerprint.png"
```

**Response:**
```json
{
  "blood_group": "O+",
  "confidence": 94.32,
  "all_probabilities": [
    { "blood_group": "O+",  "confidence": 94.32 },
    { "blood_group": "O-",  "confidence":  2.11 },
    { "blood_group": "A+",  "confidence":  1.43 }
  ],
  "info": {
    "donate": "A+, B+, O+, AB+",
    "receive": "O+, O-",
    "population": "~38%"
  }
}
```

---

## 📊 Results

After training, the following files are saved to `model_output/`:

| File | Description |
|------|-------------|
| `blood_group_model.keras` | Best trained model |
| `blood_group_model.h5` | Backup model (H5 format) |
| `class_indices.json` | Class label mapping |
| `training_history.png` | Accuracy & loss curves |
| `confusion_matrix.png` | Per-class prediction accuracy |
| `sample_prediction.png` | Example test prediction |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow 2.21, Keras |
| Base Model | MobileNetV2 (ImageNet) |
| Backend API | Flask, Flask-CORS |
| Frontend | HTML5, CSS3, Vanilla JS |
| Data Processing | NumPy, Pillow, OpenCV |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |

---

## 👩‍💻 Author

**Rakshitha D R**  
📧 rakshithadr618@gmail.com  
💻 [GitHub](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
Made with ❤️ for research and learning
</div>
