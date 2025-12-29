import json
import os
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
model = ContrastiveAudioTextModel.from_pretrained(str(MODEL_DIR))
model.eval()
index, labels = load_index()

def compute_audio_embedding(audio_path: Path) -> np.ndarray:
    waveform, _ = librosa.load(audio_path, sr=16000)
    audio_inputs = model.audio_processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model.encode_audio(audio_inputs["input_values"], audio_inputs.get("attention_mask"))
    return embeddings.cpu().numpy()


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
    scores, indices = index.search(embedding.astype(np.float32), k=8) #changed k=1 to k=3

    for score, idx in zip(scores[0], indices[0]):
        print(labels[idx], score)

    best_idx = int(indices[0][0])
    matched_label = labels[best_idx]
    video_filename = matched_label.lower().replace(" ", "_") + ".mp4" # changed and added .lower()

    return render_template("index.html", matched_video=video_filename, matched_label=matched_label)


@app.route("/videos/<path:filename>")
def serve_video(filename):
    return send_from_directory(str(VIDEOS_DIR), filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
