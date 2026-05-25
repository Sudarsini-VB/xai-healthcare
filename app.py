import os, uuid, numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "xai-healthcare-2024")

UPLOAD_FOLDER      = os.path.join("static", "uploads")
HEATMAP_FOLDER     = os.path.join("static", "heatmaps")
ALLOWED_EXTENSIONS = {"png","jpg","jpeg","bmp","tiff","webp"}
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

for d in [UPLOAD_FOLDER, HEATMAP_FOLDER, "models", "static/images"]:
    os.makedirs(d, exist_ok=True)

prediction_history = []
MODEL_PATH  = os.path.join("models", "pneumonia_model.h5")
MODEL_READY = os.path.exists(MODEL_PATH)

if MODEL_READY:
    from predict import predict as _live_predict
    print("[INFO] LIVE mode — model found.")
else:
    print("[WARN] DEMO mode — no model found.")

def allowed_file(f):
    return "." in f and f.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

def demo_predict(image_path):
    import random, cv2, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    random.seed(os.path.getsize(image_path))
    score = random.uniform(0.15, 0.92)
    is_pneu = score >= 0.5
    heatmap_path = None
    try:
        img = cv2.imread(image_path)
        if img is None:
            from PIL import Image
            img = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        h,w = img.shape[:2]
        Y,X = np.ogrid[:h,:w]
        cx,cy = w//2+w//8, h//2
        blob = np.exp(-((X-cx)**2+(Y-cy)**2)/(2*(min(w,h)//3)**2))
        coloured = cv2.applyColorMap(np.uint8(255*blob), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, coloured, 0.4, 0)
        fig, axes = plt.subplots(1,2,figsize=(10,5))
        fig.patch.set_facecolor("#0a0e1a")
        for ax,im,t in zip(axes,[cv2.cvtColor(img,cv2.COLOR_BGR2RGB),cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)],["Original X-ray","Grad-CAM Heatmap (Demo)"]):
            ax.imshow(im); ax.set_title(t,color="white"); ax.axis("off")
        sm = plt.cm.ScalarMappable(cmap="jet",norm=plt.Normalize(0,1)); sm.set_array([])
        cbar = fig.colorbar(sm,ax=axes[1],fraction=0.046,pad=0.04)
        cbar.set_label("Activation Intensity",color="white",fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(),color="white")
        plt.suptitle("Grad-CAM Visualisation (Demo Mode)",color="white",fontsize=13)
        plt.tight_layout()
        fname = f"heatmap_{uuid.uuid4().hex[:8]}.png"
        heatmap_path = os.path.join(HEATMAP_FOLDER, fname)
        plt.savefig(heatmap_path, bbox_inches="tight", dpi=100, facecolor=fig.get_facecolor())
        plt.close()
    except Exception as e:
        print(f"[WARN] Demo heatmap error: {e}")
    return {"prediction":"PNEUMONIA" if is_pneu else "NORMAL",
            "confidence":round(score*100 if is_pneu else (1-score)*100,2),
            "raw_score":round(score,4),"heatmap_path":heatmap_path,
            "error":"","demo_mode":True}

def build_explanation(prediction, confidence, raw_score):
    if prediction == "PNEUMONIA":
        sev = "High" if confidence>=85 else "Moderate" if confidence>=65 else "Low"
        return {"summary":f"⚠️ Pneumonia Detected ({sev} Confidence)",
                "detail":f"The AI detected pneumonia patterns. Sigmoid output: {raw_score:.3f}, confidence: {confidence:.1f}%.",
                "advice":"Consult a qualified radiologist immediately. This tool does NOT replace professional medical evaluation.",
                "severity":sev}
    sev = "High" if confidence>=85 else "Moderate"
    return {"summary":f"✅ No Pneumonia Detected ({sev} Confidence)",
            "detail":f"No significant pneumonia patterns found. Sigmoid output: {raw_score:.3f}, confidence: {confidence:.1f}%.",
            "advice":"Always confirm with a certified radiologist. Follow-up recommended if symptoms persist.",
            "severity":sev}

@app.route("/")
def index():
    return render_template("index.html", model_ready=MODEL_READY)

@app.route("/predict", methods=["POST"])
def predict_route():
    if "xray" not in request.files:
        flash("No file in request.","danger"); return redirect(url_for("index"))
    file = request.files["xray"]
    if file.filename == "":
        flash("No file selected.","warning"); return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash(f"Unsupported type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}","danger")
        return redirect(url_for("index"))
    ext = file.filename.rsplit(".",1)[1].lower()
    fname = f"{uuid.uuid4().hex}.{ext}"
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(fpath)
    result = _live_predict(fpath) if MODEL_READY else demo_predict(fpath)
    if result["error"]:
        flash(f"Prediction error: {result['error']}","danger"); return redirect(url_for("index"))
    explanation = build_explanation(result["prediction"], result["confidence"], result["raw_score"])
    upload_url  = "/" + fpath.replace("\\","/")
    heatmap_url = ("/" + result["heatmap_path"].replace("\\","/")) if result.get("heatmap_path") else None
    record = {"id":len(prediction_history)+1,"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "filename":file.filename,"prediction":result["prediction"],
              "confidence":result["confidence"],"upload_url":upload_url,"heatmap_url":heatmap_url,
              "demo_mode":result.get("demo_mode",False)}
    prediction_history.append(record)
    return render_template("result.html", prediction=result["prediction"], confidence=result["confidence"],
        raw_score=result["raw_score"], upload_url=upload_url, heatmap_url=heatmap_url,
        explanation=explanation, record=record, demo_mode=result.get("demo_mode",False), model_ready=MODEL_READY)

@app.route("/history")
def history():
    return jsonify(prediction_history[-20:])

@app.route("/about")
def about():
    return render_template("about.html", model_ready=MODEL_READY)

@app.errorhandler(404)
def not_found(e): return render_template("404.html"), 404

@app.errorhandler(413)
def too_large(e):
    flash("File too large. Max 16 MB.","danger"); return redirect(url_for("index"))

@app.errorhandler(500)
def server_error(e): return render_template("500.html", error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
