import argparse
import random
from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import ContrastiveAudioTextModel

# This version trains on REAL audio files in data/audios (wav or mp3),
# matched by filename to videos in static/sign_videos.
# Example:
#   static/sign_videos/hello_how_are_you.mp4
#   data/audios/hello_how_are_you.wav


class AudioDataset(Dataset):
    def __init__(
        self,
        labels: List[str],
        audio_paths: List[Path],
        model: ContrastiveAudioTextModel,
        sample_rate: int = 16000,
    ):
        self.labels = labels
        self.audio_paths = audio_paths
        self.model = model
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        audio_path = self.audio_paths[idx]
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)

        # Safety: remove NaNs/Infs and keep audio in float32
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Optional safety: normalize to avoid extreme amplitudes
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


def prepare_real_audio(labels: List[str], audio_dir: Path) -> List[Path]:
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory {audio_dir} not found")

    audio_paths: List[Path] = []
    for label in labels:
        safe_name = label.replace(" ", "_")
        wav_path = audio_dir / f"{safe_name}.wav"
        mp3_path = audio_dir / f"{safe_name}.mp3"

        if wav_path.exists():
            audio_paths.append(wav_path)
        elif mp3_path.exists():
            audio_paths.append(mp3_path)
        else:
            raise FileNotFoundError(
                f"Missing audio for label '{label}'. Expected {wav_path} or {mp3_path}"
            )
    return audio_paths


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    video_dir = Path(args.video_dir)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = discover_labels(video_dir)
    print(f"Discovered {len(labels)} labels from {video_dir}")

    model = ContrastiveAudioTextModel()

    # Load real audio matching each label
    audio_paths = prepare_real_audio(labels, audio_dir)
    dataset = AudioDataset(labels, audio_paths, model, sample_rate=args.sample_rate)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn
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

            # ✅ Safety guard 1: stop if training diverges
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("Loss became NaN/Inf. Training diverged. Lower the learning rate.")

            optimizer.zero_grad()
            loss.backward()

            # ✅ Safety guard 2: gradient clipping (prevents NaN explosions)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            epoch_loss += float(loss.item())
            progress.set_postfix({"loss": float(loss.item())})

        print(f"Epoch {epoch+1} loss: {epoch_loss / max(1, len(dataloader)):.4f}")

    model.save_pretrained(str(output_dir))
    print(f"Model saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train audio-text contrastive model (real audio)")
    parser.add_argument(
        "--video_dir",
        default="static/sign_videos",
        help="Path to directory containing .mp4 files",
    )
    parser.add_argument(
        "--audio_dir",
        default="data/audios",
        help="Directory containing audio files matching video names (wav or mp3)",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/model",
        help="Directory to save model and processors",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=1e-5)  # safer default
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)  # ✅ new: gradient clipping
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
