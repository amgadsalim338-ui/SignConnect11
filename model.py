import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor


@dataclass
class AudioTextConfig:
    audio_encoder_name: str = "facebook/wav2vec2-base-960h"
    text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    projection_dim: int = 256
    temperature: float = 0.07

    def to_dict(self):
        return {
            "audio_encoder_name": self.audio_encoder_name,
            "text_encoder_name": self.text_encoder_name,
            "projection_dim": self.projection_dim,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AudioTextConfig":
        return cls(**data)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.linear(x)
        return F.normalize(projected, dim=-1)


class ContrastiveAudioTextModel(nn.Module):
    def __init__(self, config: Optional[AudioTextConfig] = None):
        super().__init__()
        self.config = config or AudioTextConfig()

        self.audio_processor = Wav2Vec2Processor.from_pretrained(self.config.audio_encoder_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(self.config.audio_encoder_name)
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        audio_hidden = self.audio_encoder.config.hidden_size
        self.audio_proj = ProjectionHead(audio_hidden, self.config.projection_dim)

        self.text_tokenizer = AutoTokenizer.from_pretrained(self.config.text_encoder_name)
        self.text_encoder = AutoModel.from_pretrained(self.config.text_encoder_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        text_hidden = self.text_encoder.config.hidden_size
        self.text_proj = ProjectionHead(text_hidden, self.config.projection_dim)

    def encode_audio(self, audio_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.audio_encoder(audio_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        return self.audio_proj(pooled)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        masked_states = hidden_states * mask
        pooled = masked_states.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.text_proj(pooled)

    def forward(self, audio_values: torch.Tensor, audio_attention: torch.Tensor, input_ids: torch.Tensor, text_attention: torch.Tensor) -> dict:
        audio_embeddings = self.encode_audio(audio_values, audio_attention)
        text_embeddings = self.encode_text(input_ids, text_attention)
        loss = self.contrastive_loss(audio_embeddings, text_embeddings)
        return {"loss": loss, "audio_embeddings": audio_embeddings, "text_embeddings": text_embeddings}

    def contrastive_loss(self, audio_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        temperature = self.config.temperature
        logits = torch.matmul(audio_embeddings, text_embeddings.T) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_audio = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.T, labels)
        return (loss_audio + loss_text) / 2

    def save_pretrained(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "model.pt"))
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        self.audio_processor.save_pretrained(os.path.join(output_dir, "audio_processor"))
        self.text_tokenizer.save_pretrained(os.path.join(output_dir, "text_tokenizer"))

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "ContrastiveAudioTextModel":
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            config_data = json.load(f)
        config = AudioTextConfig.from_dict(config_data)
        model = cls(config)
        state = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        model.load_state_dict(state)
        model.audio_processor = Wav2Vec2Processor.from_pretrained(os.path.join(model_dir, "audio_processor"))
        model.text_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "text_tokenizer"))
        model.eval()
        return model

