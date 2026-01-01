import json
from pathlib import Path

import faiss
import librosa
import numpy as np
import torch
from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for

from model import ContrastiveAudioTextModel

BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "static" / "sign_videos"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "outputs" / "model"
INDEX_DIR = BASE_DIR / "outputs" / "index"

app = Flask(__name__)
app.config["SECRET_KEY"] = "signconnect-secret"
UPLOAD_DIR.mkdir(exist_ok=True)


def load_index():
    index_path = INDEX_DIR / "index.faiss"
    labels_path = INDEX_DIR / "labels.json"
    if not index_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Index or labels file not found. Run build_index.py first.")
    index = faiss.read_index(str(index_path))
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return index, labels


print("Loading model from:", MODEL_DIR.resolve())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ContrastiveAudioTextModel.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()
index, labels = load_index()


def compute_audio_embedding(audio_path: Path) -> np.ndarray:
    # 1️⃣ Load audio
    waveform, _ = librosa.load(audio_path, sr=16000)

    # 2️⃣ 🔧 FIX: remove NaNs / Infs
    waveform = np.nan_to_num(
        waveform, nan=0.0, posinf=0.0, neginf=0.0
    )

    # 3️⃣ 🔧 FIX: normalize amplitude (prevents numerical explosion)
    max_abs = np.max(np.abs(waveform)) + 1e-9
    waveform = waveform / max_abs

    # 4️⃣ ensure correct dtype
    waveform = waveform.astype(np.float32)

    # 🔍 DEBUG (temporary)
    print("waveform nan?:", np.isnan(waveform).any())
    print("waveform min/max:", float(np.min(waveform)), float(np.max(waveform)))

    # 5️⃣ Convert to model input
    audio_inputs = model.audio_processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    device = next(model.parameters()).device
    audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}

    with torch.no_grad():
        embeddings = model.encode_audio(
            audio_inputs["input_values"],
            audio_inputs.get("attention_mask"),
        )

    embedding_np = embeddings.detach().cpu().numpy()

    # 🔍 DEBUG (temporary)
    print("embedding shape:", embedding_np.shape)
    print("embedding nan?:", np.isnan(embedding_np).any())

    return embedding_np



@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        flash("No file part in the request")
        return redirect(url_for("index_page"))

    file = request.files["audio"]
    print("Uploaded filename:", file.filename)

    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("index_page"))

    save_path = UPLOAD_DIR / file.filename
    file.save(save_path)

    embedding = compute_audio_embedding(save_path)
    faiss.normalize_L2(embedding)

    # safe k
    k = min(8, len(labels), index.ntotal)
    scores, indices = index.search(embedding.astype(np.float32), k=k)

    # Print top-k safely and collect valid results
    valid_results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        print(labels[idx], float(score))
        valid_results.append((int(idx), float(score)))

    if not valid_results:
        flash("No match found (index returned no valid results).")
        return redirect(url_for("index_page"))

    best_idx = valid_results[0][0]
    matched_label = labels[best_idx]
    video_filename = matched_label.lower().replace(" ", "_") + ".mp4"

    return render_template(
        "index.html",
        matched_video=video_filename,
        matched_label=matched_label,
    )


@app.route("/videos/<path:filename>")
def serve_video(filename):
    return send_from_directory(str(VIDEOS_DIR), filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
