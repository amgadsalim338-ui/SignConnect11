import argparse
import json
from pathlib import Path
from typing import List

import faiss
import numpy as np
import torch

from model import ContrastiveAudioTextModel


def discover_labels(video_dir: Path) -> List[str]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory {video_dir} not found")
    labels = []
    for entry in video_dir.iterdir():
        if entry.suffix.lower() == ".mp4":
            labels.append(entry.stem.replace("_", " "))
    if not labels:
        raise ValueError("No .mp4 files found in the video directory")
    return sorted(labels)


def embed_labels(labels: List[str], model: ContrastiveAudioTextModel) -> np.ndarray:
    tokenizer = model.text_tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for label in labels:
            text_inputs = tokenizer(label, return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            embeddings = model.encode_text(text_inputs["input_ids"], text_inputs["attention_mask"])
            all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype(np.float32))
    return index


def save_index(index: faiss.Index, labels: List[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.faiss"
    labels_path = output_dir / "labels.json"
    faiss.write_index(index, str(index_path))
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved index to {index_path} and labels to {labels_path}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for text labels")
    parser.add_argument("--model_dir", default="outputs/model", help="Trained model directory")
    parser.add_argument("--video_dir", default="static/sign_videos", help="Directory with .mp4 files")
    parser.add_argument("--output_dir", default="outputs/index", help="Directory to save the FAISS index")
    args = parser.parse_args()

    labels = discover_labels(Path(args.video_dir))
    model = ContrastiveAudioTextModel.from_pretrained(args.model_dir)
    embeddings = embed_labels(labels, model)
    index = build_faiss_index(embeddings)
    save_index(index, labels, Path(args.output_dir))


if __name__ == "__main__":
    main()
