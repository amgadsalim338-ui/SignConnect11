import argparse
import os
import random
from pathlib import Path
from typing import List

import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import ContrastiveAudioTextModel

try:
    import pyttsx3
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("pyttsx3 is required for TTS generation") from exc


class TTSDataset(Dataset):
    def __init__(self, labels: List[str], audio_paths: List[Path], model: ContrastiveAudioTextModel, sample_rate: int = 16000):
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
        return {
            "label": label,
            "audio": audio,
        }

    def collate_fn(self, batch):
        texts = [item["label"] for item in batch]
        audios = [item["audio"] for item in batch]

        audio_inputs = self.model.audio_processor(
            audios, sampling_rate=self.sample_rate, return_tensors="pt", padding=True
        )
        text_inputs = self.model.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        return audio_inputs, text_inputs


def synthesize_speech(text: str, output_path: Path, rate: int = 16000):
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)
    engine.save_to_file(text, str(output_path))
    engine.runAndWait()

    if output_path.suffix != ".wav":
        raise ValueError("pyttsx3 save_to_file should produce a .wav file")

    # pyttsx3 may default to 22k, resample for consistency
    audio, _ = librosa.load(output_path, sr=rate)
    sf.write(str(output_path), audio, rate)


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


def prepare_tts_audio(labels: List[str], tts_dir: Path) -> List[Path]:
    tts_dir.mkdir(parents=True, exist_ok=True)
    audio_paths = []
    for label in labels:
        safe_name = label.replace(" ", "_")
        output_path = tts_dir / f"{safe_name}.wav"
        if not output_path.exists():
            synthesize_speech(label, output_path)
        audio_paths.append(output_path)
    return audio_paths


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    tts_dir = output_dir / "tts_audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = discover_labels(video_dir)
    print(f"Discovered {len(labels)} labels from {video_dir}")

    model = ContrastiveAudioTextModel()

    audio_paths = prepare_tts_audio(labels, tts_dir)
    dataset = TTSDataset(labels, audio_paths, model)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=1e-3
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})
        print(f"Epoch {epoch+1} loss: {epoch_loss / len(dataloader):.4f}")

    model.save_pretrained(str(output_dir))
    print(f"Model saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train audio-text contrastive model")
    parser.add_argument("--video_dir", default="videos", help="Path to directory containing .mp4 files")
    parser.add_argument("--output_dir", default="outputs/model", help="Directory to save model and processors")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
