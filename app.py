import os
import uuid
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException

import testing2 as t2
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_PATH = t2.MODEL_PATH

app = Flask(__name__)
app.secret_key = "replace-me"
app.config["PROPAGATE_EXCEPTIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# --- Safe model loading ---
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[FATAL] Model failed to load: {e}")
    model = None


# --- HTTP-level error handler (413, 400, 404, 405 etc.) ---
@app.errorhandler(HTTPException)
def handle_http_exception(e):
    print(f"[HTTP ERROR] {e.code} - {e.description}")
    return jsonify({
        "error": f"{e.code} {e.name}: {e.description}"
    }), e.code


# --- Global Python-level error handler ---
@app.errorhandler(Exception)
def handle_exception(e):
    tb = traceback.format_exc()
    print(f"[GLOBAL ERROR HANDLER]\n{tb}")
    return jsonify({"error": str(e), "trace": tb[-800:]}), 500


def allowed_file(fname: str) -> bool:
    return Path(fname).suffix.lower() in ALLOWED_EXT


def save_latest_figure(out_path: Path) -> bool:
    """
    Save the most recently drawn matplotlib figure to disk.
    Works with plt.draw() (used in testing2.py instead of plt.show()).
    """
    try:
        managers = matplotlib._pylab_helpers.Gcf.get_all_fig_managers()
        if not managers:
            print("[WARN] No matplotlib figures found in memory.")
            return False
        fig = managers[-1].canvas.figure
        fig.savefig(out_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"[INFO] Grad-CAM saved to: {out_path}")
        return True
    except Exception as ex:
        print(f"[WARN] save_latest_figure failed: {ex}")
        return False


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        wants_json = (
            "application/json" in request.headers.get("Accept", "")
            or request.args.get("fmt") == "json"
        )

        # --- Guard: model not loaded ---
        if model is None:
            err = "Model failed to load on startup. Check server logs."
            if wants_json:
                return jsonify({"error": err}), 500
            flash(err)
            return redirect(url_for("index"))

        # --- Validate file ---
        if "image" not in request.files:
            err = "No file part in the request."
            if wants_json:
                return jsonify({"error": err}), 400
            flash(err)
            return redirect(url_for("index"))

        f = request.files["image"]
        if f.filename == "":
            err = "Please choose an image."
            if wants_json:
                return jsonify({"error": err}), 400
            flash(err)
            return redirect(url_for("index"))

        if not allowed_file(f.filename):
            err = "Unsupported file type. Use JPG/PNG/BMP/WEBP."
            if wants_json:
                return jsonify({"error": err}), 400
            flash(err)
            return redirect(url_for("index"))

        # --- Save uploaded file ---
        safe = secure_filename(f.filename)
        stem = Path(safe).stem[:40]
        ext = Path(safe).suffix.lower()
        unique_name = f"{stem}-{uuid.uuid4().hex[:8]}{ext}"
        img_path = UPLOAD_DIR / unique_name
        f.save(str(img_path))
        print(f"[INFO] Image saved to: {img_path}")

        # --- Run inference ---
        print("[INFO] Running inference...")
        label, prob, explanation = t2.explain_image(str(img_path), model)
        print(f"[INFO] Inference done. Label={label}, Prob={prob}")

        # --- Save Grad-CAM figure ---
        gradcam_name = f"{Path(unique_name).stem}-gradcam.png"
        gradcam_path = UPLOAD_DIR / gradcam_name
        ok = save_latest_figure(gradcam_path)

        if not ok:
            print("[WARN] Grad-CAM figure not available, using placeholder.")
            fig = plt.figure(figsize=(6, 3))
            plt.axis("off")
            plt.text(0.5, 0.5, "Grad-CAM not available", ha="center", va="center",
                     fontsize=14, color="gray")
            fig.savefig(gradcam_path, bbox_inches="tight", dpi=140)
            plt.close(fig)

        if wants_json:
            return jsonify({
                "label": label,
                "probability": round(float(prob), 4),
                "gradcam_image": url_for("static", filename=f"uploads/{gradcam_name}"),
                "explanation": explanation
            })

        return render_template(
            "result.html",
            label=label,
            probability=round(float(prob), 4),
            gradcam_image=url_for("static", filename=f"uploads/{gradcam_name}"),
            explanation=explanation
        )

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] Something went wrong in /predict:\n{tb}")
        return jsonify({
            "error": str(e),
            "trace": tb[-800:]
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)