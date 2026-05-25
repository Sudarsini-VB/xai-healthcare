# 🫁 XAI Healthcare — Explainable AI Pneumonia Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-2.3-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Grad--CAM-XAI-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

> **An AI-powered chest X-ray analysis system that detects pneumonia using a CNN and explains its decision with Grad-CAM heatmaps — all wrapped in a modern Flask web application.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Training the Model](#-training-the-model)
- [Running the App](#-running-the-app)
- [How Grad-CAM Works](#-how-grad-cam-works)
- [API / Routes](#-api--routes)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [Disclaimer](#-disclaimer)

---

## 🔭 Overview

This project demonstrates a **production-ready, explainable AI healthcare system** built for:

- **Final Year Projects** — showcases deep learning, computer vision, and XAI
- **Hackathons** — complete, polished, and deployable in minutes
- **Portfolio** — modern UI, clean architecture, real ML pipeline

The system accepts a chest X-ray image, passes it through a trained **Convolutional Neural Network (CNN)**, and returns:

1. **Binary diagnosis** — PNEUMONIA or NORMAL
2. **Confidence percentage** — derived from the sigmoid output
3. **Grad-CAM heatmap** — visual explanation of which lung regions drove the decision

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ Drag-and-drop upload | Upload chest X-rays via drag-and-drop or file browser |
| 🧠 CNN Deep Learning | Custom 3-block CNN trained on the Kaggle Pneumonia dataset |
| 🔥 Grad-CAM XAI | Gradient-weighted Class Activation Maps computed from scratch |
| 📊 Confidence score | Sigmoid output mapped to human-readable percentage |
| 🖼️ Side-by-side comparison | Original X-ray vs Grad-CAM heatmap overlay |
| 📋 Prediction history | In-memory log of all predictions, exposed as JSON at `/history` |
| 📥 Downloadable heatmap | Download Grad-CAM image directly from the result page |
| 📱 Responsive UI | Works on desktop, tablet, and mobile |
| ⚡ Loading spinner | Visual feedback during inference |
| 🎨 Dark medical theme | Professional dark UI with teal/cyan accent palette |

---

## 🏗️ System Architecture

```
User uploads X-ray (JPEG/PNG)
         │
         ▼
   Flask /predict route
         │
         ▼
   preprocess.py
   ┌─────────────────────────────────┐
   │ • OpenCV: BGR → RGB             │
   │ • Resize to 224 × 224           │
   │ • Normalise pixels to [0, 1]    │
   │ • Expand dims → (1, 224, 224, 3)│
   └─────────────────────────────────┘
         │
         ▼
   predict.py
   ┌─────────────────────────────────┐
   │ • Load cached Keras model       │
   │ • model.predict() → sigmoid     │
   │ • Threshold 0.50 → class label  │
   │ • Confidence % calculation      │
   └─────────────────────────────────┘
         │
         ▼
   gradcam.py
   ┌─────────────────────────────────┐
   │ • GradientTape on last Conv2D   │
   │ • Global-avg-pool gradients     │
   │ • Weight feature maps           │
   │ • ReLU → normalise              │
   │ • Resize + COLORMAP_JET overlay │
   │ • Save PNG to static/heatmaps/  │
   └─────────────────────────────────┘
         │
         ▼
   result.html
   • Original X-ray
   • Heatmap side-by-side
   • Confidence bar
   • AI explanation text
   • Download buttons
```

---

## 🛠️ Tech Stack

**Backend**
- Python 3.9+
- Flask 2.3 (routing, templating, file handling)
- Werkzeug (secure file upload)

**AI / ML**
- TensorFlow 2.13 / Keras (CNN model, GradientTape)
- scikit-learn (metrics, confusion matrix)

**Computer Vision**
- OpenCV (image loading, resizing, heatmap blending)
- NumPy (array manipulation)
- Matplotlib (Grad-CAM figure generation)

**Frontend**
- Bootstrap 5.3 (responsive layout)
- Bootstrap Icons
- Vanilla JavaScript (drag-and-drop, preview, spinner)
- Google Fonts: DM Serif Display, DM Sans, JetBrains Mono

---

## 📁 Project Structure

```
xai-healthcare-project/
│
├── dataset/                   # Kaggle dataset goes here
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── models/
│   └── pneumonia_model.h5     # Saved after training
│
├── static/
│   ├── uploads/               # Uploaded X-rays (auto-created)
│   ├── heatmaps/              # Generated Grad-CAM images
│   ├── images/                # Training curves, confusion matrix
│   ├── css/
│   │   └── style.css          # Full custom stylesheet
│   └── js/
│       └── script.js          # Upload UX, drag-and-drop, spinner
│
├── templates/
│   ├── layout.html            # Base template (navbar, footer)
│   ├── index.html             # Home / upload page
│   ├── result.html            # Prediction results page
│   ├── about.html             # About page
│   ├── 404.html               # 404 error page
│   └── 500.html               # 500 error page
│
├── preprocess.py              # Image loading, normalisation, augmentation
├── train_model.py             # CNN architecture, training, evaluation
├── gradcam.py                 # Grad-CAM from scratch (GradientTape)
├── predict.py                 # Full prediction pipeline
├── app.py                     # Flask app (routes, error handlers)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourname/xai-healthcare-project.git
cd xai-healthcare-project
```

### 2. Create a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you encounter issues with `opencv-python-headless` on Windows, try `opencv-python` instead.

---

## 📂 Dataset Setup

This project uses the **[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** dataset from Kaggle.

**Download steps:**

1. Install Kaggle CLI: `pip install kaggle`
2. Set up your Kaggle API key (`~/.kaggle/kaggle.json`)
3. Run:
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip
   ```
4. Move the folders so the structure matches:
   ```
   dataset/
   ├── train/NORMAL/   (1341 images)
   ├── train/PNEUMONIA/(3875 images)
   ├── val/NORMAL/     (8 images)
   ├── val/PNEUMONIA/  (8 images)
   ├── test/NORMAL/    (234 images)
   └── test/PNEUMONIA/ (390 images)
   ```

> The original dataset's `val/` folder is very small. You can move some training images into it for better validation, or use `validation_split` in the `ImageDataGenerator`.

---

## 🏋️ Training the Model

```bash
python train_model.py
```

This will:
- Load data generators with augmentation
- Build the 3-block CNN
- Train for up to 10 epochs (EarlyStopping applies)
- Save the best checkpoint to `models/pneumonia_model.h5`
- Print training, validation, and test accuracy
- Generate confusion matrix → `static/images/confusion_matrix.png`
- Generate training curves → `static/images/training_history.png`

**Expected accuracy:** ~90–95% on the test set.

---

## 🚀 Running the App

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

> Make sure `models/pneumonia_model.h5` exists before launching the app.
> If you want to test the UI before training, you can use any Keras model saved at that path.

---

## 🔥 How Grad-CAM Works

Grad-CAM (**Gradient-weighted Class Activation Mapping**) explains CNN predictions without modifying the model architecture.

**Algorithm (implemented in `gradcam.py`):**

1. **Sub-model creation** — Build a model that outputs both the last Conv2D feature maps AND the final prediction simultaneously.
2. **Forward pass with GradientTape** — Record operations so we can compute gradients.
3. **Gradient computation** — Differentiate the predicted class score with respect to each feature map in the last conv layer.
4. **Global Average Pooling** — Pool the gradients spatially → one importance weight per channel.
5. **Weighted combination** — Multiply each feature map by its importance weight and sum them.
6. **ReLU** — Keep only positive activations (regions that push *toward* the predicted class).
7. **Resize & overlay** — Scale the heatmap to the original image size and blend it using `cv2.applyColorMap(COLORMAP_JET)`.

**Colour interpretation:**
- 🔴 **Red / warm** — High activation → strong pneumonia signal here
- 🟡 **Yellow / green** — Moderate activation
- 🔵 **Blue / cool** — Low activation → minimal influence

---

## 🌐 API / Routes

| Method | Route | Description |
|---|---|---|
| GET | `/` | Home page with upload form |
| POST | `/predict` | Upload X-ray, run inference, show results |
| GET | `/history` | JSON array of last 20 predictions |
| GET | `/about` | About / project info page |

---

## 📸 Screenshots

> *(Add screenshots here after running the app)*

| Home Page | Result Page |
|---|---|
| ![Home](docs/home.png) | ![Result](docs/result.png) |

---

## 🚀 Future Improvements

- [ ] **VGG16 / ResNet50 transfer learning** for higher accuracy
- [ ] **Multi-class detection** (bacterial vs viral pneumonia, COVID-19)
- [ ] **PDF report generation** with patient details and heatmap
- [ ] **LIME** and **SHAP** explainability methods alongside Grad-CAM
- [ ] **Database persistence** (SQLite/PostgreSQL) for prediction history
- [ ] **User authentication** for multi-user hospital deployment
- [ ] **Docker containerisation** for one-command deployment
- [ ] **REST API** with authentication for mobile app integration
- [ ] **Grad-CAM++** for more precise localisation
- [ ] **Batch processing** — upload multiple X-rays at once

---

## ⚠️ Disclaimer

This application is built **for educational and research purposes only**.

- It is **NOT** a certified medical device.
- It must **NOT** be used for actual clinical diagnosis.
- Always consult a **qualified radiologist or physician** for medical decisions.
- The AI model's predictions can be incorrect and should never replace professional medical evaluation.

---

## 📄 License

MIT License — feel free to use, modify, and distribute with attribution.

---

<p align="center">
  Built with ❤️ using TensorFlow, Flask, and Grad-CAM
</p>
