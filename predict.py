import os, uuid, numpy as np

MODEL_PATH  = os.path.join("models", "pneumonia_model.h5")
HEATMAP_DIR = os.path.join("static", "heatmaps")
CLASS_LABELS = {0: "NORMAL", 1: "PNEUMONIA"}
THRESHOLD = 0.5
_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
        from tensorflow.keras.models import load_model as _lm
        print(f"[INFO] Loading model …")
        _model = _lm(MODEL_PATH, compile=False)
        _model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        print("[INFO] Model loaded.")
    return _model

def predict(image_path):
    result = {"prediction": None, "confidence": 0.0, "heatmap_path": None,
              "raw_score": 0.0, "error": "", "demo_mode": False}
    try:
        from preprocess import preprocess_for_prediction
        from gradcam import generate_and_save_gradcam
        img_array = preprocess_for_prediction(image_path)
        if img_array is None:
            result["error"] = "Cannot preprocess image."; return result
        model = load_model()
        raw = float(model.predict(img_array, verbose=0)[0][0])
        cls = 1 if raw >= THRESHOLD else 0
        conf = raw*100 if cls == 1 else (1-raw)*100
        result["prediction"] = CLASS_LABELS[cls]
        result["confidence"] = round(conf, 2)
        result["raw_score"]  = round(raw, 4)
        os.makedirs(HEATMAP_DIR, exist_ok=True)
        hp = os.path.join(HEATMAP_DIR, f"heatmap_{uuid.uuid4().hex[:8]}.png")
        generate_and_save_gradcam(model, img_array, image_path, hp, cls)
        result["heatmap_path"] = hp
    except Exception as e:
        import traceback; traceback.print_exc()
        result["error"] = str(e)
    return result
