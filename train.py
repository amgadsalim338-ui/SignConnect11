import argparse
import random
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import ContrastiveAudioTextModel

# This version trains on REAL audio files stored in per-label folders:
#
# data/audios/
#   excuse_me/
#     EX.wav
#     EX2.wav
#     excuse_me.wav
#   thats_a_cool_outfit/
#     TACO.wav
#     thats_a_cool_outfit.mp3
#
# Labels are discovered from videos in static/sign_videos:
#   static/sign_videos/excuse_me.mp4  -> label "excuse me"
#
# Each audio file inside data/audios/<label_folder>/ becomes a training example for that label.


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


class MultiAudioDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, Path]],
        model: ContrastiveAudioTextModel,
        sample_rate: int = 16000,
    ):
        self.samples = samples
        self.model = model
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, audio_path = self.samples[idx]
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)

        # Safety: remove NaNs/Infs and keep audio in float32
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Normalize to avoid extreme amplitudes
        max_abs = float(np.max(np.abs(audio)) + 1e-9)
        audio = audio / max_abs

        return {"label": label, "audio": audio}

    def collate_fn(self, batch):
        texts = [item["label"] for item in batch]
        audios = [item["audio"] for item in batch]

        audio_inputs = self.model.audio_processor(
            audios, sampling_rate=self.sample_rate, return_tensors="pt", padding=True
        )
        text_inputs = self.model.text_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        return audio_inputs, text_inputs


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


def prepare_multi_audio_samples(labels: List[str], audio_root: Path) -> List[Tuple[str, Path]]:
    """
    For each label (e.g. "excuse me"), expect a folder:
      audio_root / "excuse_me" / *.wav|*.mp3|...
    Collect (label, path) for every audio file inside.
    """
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root directory {audio_root} not found")

    samples: List[Tuple[str, Path]] = []
    missing = []

    for label in labels:
        folder_name = label.lower().replace(" ", "_")
        label_dir = audio_root / folder_name

        if not label_dir.exists() or not label_dir.is_dir():
            missing.append(str(label_dir))
            continue

        files = [p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
        if not files:
            missing.append(str(label_dir))
            continue

        # Add all audio files for this label
        for p in sorted(files):
            samples.append((label, p))

    if missing:
        raise FileNotFoundError(
            "Missing audio folder(s) or no audio files found for some labels.\n"
            "Expected per-label folders like data/audios/excuse_me/ with audio files inside.\n"
            "Missing/empty:\n  - " + "\n  - ".join(missing[:30]) + ("\n  ...(more)" if len(missing) > 30 else "")
        )

    return samples


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    video_dir = Path(args.video_dir)
    audio_root = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = discover_labels(video_dir)
    print(f"Discovered {len(labels)} labels from {video_dir}")

    model = ContrastiveAudioTextModel()

    # Collect ALL audio samples from per-label folders
    samples = prepare_multi_audio_samples(labels, audio_root)

    # Print per-label counts (helpful sanity check)
    counts = {}
    for lbl, _p in samples:
        counts[lbl] = counts.get(lbl, 0) + 1
    print("Audio samples per label:")
    for lbl in labels:
        print(f"  {lbl}: {counts.get(lbl, 0)}")

    print(f"Total training samples: {len(samples)}")

    dataset = MultiAudioDataset(samples, model, sample_rate=args.sample_rate)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-3,
    )

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for audio_inputs, text_inputs in progress:
            audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            outputs = model(
                audio_inputs["input_values"],
                audio_inputs.get("attention_mask"),
                text_inputs["input_ids"],
                text_inputs["attention_mask"],
            )
            loss = outputs["loss"]

            # Safety guard 1: stop if training diverges
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    "Loss became NaN/Inf. Training diverged. "
                    "Lower the learning rate and/or reduce epochs."
                )

            optimizer.zero_grad()
            loss.backward()

            # Safety guard 2: gradient clipping (prevents NaN explosions)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            epoch_loss += float(loss.item())
            progress.set_postfix({"loss": float(loss.item())})

        print(f"Epoch {epoch+1} loss: {epoch_loss / max(1, len(dataloader)):.4f}")

    model.save_pretrained(str(output_dir))
    print(f"Model saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train audio-text contrastive model (multi-audio per label)")
    parser.add_argument(
        "--video_dir",
        default="static/sign_videos",
        help="Path to directory containing .mp4 files",
    )
    parser.add_argument(
        "--audio_dir",
        default="data/audios",
        help="Audio ROOT directory containing per-label folders (e.g. data/audios/excuse_me/*.wav)",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/model",
        help="Directory to save model and processors",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
