
#!/usr/bin/env python3
"""
Speaker recognition baselines for comparison in a datathon.

Supported backbones:
1) ECAPA-TDNN embeddings (pretrained SpeechBrain) + linear head
2) WavLM speaker-verification backbone (pretrained Microsoft) + linear head
3) X-vector style TDNN baseline trained end-to-end

Expected manifest CSV columns:
- path: path to audio file
- label: speaker class name or integer label
Optional columns:
- split: train/valid/test
- utt_id: utterance id

Examples:
  python speaker_recognition_baselines.py --manifest data.csv --split-col split --split train --model ecapa --epochs 5
  python speaker_recognition_baselines.py --manifest data.csv --split train --model wavlm --freeze-backbone
  python speaker_recognition_baselines.py --manifest data.csv --split train --model xvector
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import accuracy_score, f1_score
except Exception:  # pragma: no cover
    accuracy_score = None
    f1_score = None

try:
    from speechbrain.pretrained import EncoderClassifier
except Exception:  # pragma: no cover
    EncoderClassifier = None

try:
    from transformers import AutoFeatureExtractor, AutoModel
except Exception:  # pragma: no cover
    AutoFeatureExtractor = None
    AutoModel = None



# Reproducibility


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Data


class SpeakerManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: Optional[str] = None,
        split_col: str = "split",
        path_col: str = "path",
        label_col: str = "label",
        target_sr: int = 16000,
        max_seconds: Optional[float] = None,
        min_seconds: Optional[float] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.split_col = split_col
        self.path_col = path_col
        self.label_col = label_col
        self.target_sr = target_sr
        self.max_seconds = max_seconds
        self.min_seconds = min_seconds

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with self.manifest_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise ValueError("Manifest is empty.")

        if self.path_col not in rows[0] or self.label_col not in rows[0]:
            raise ValueError(
                f"Manifest must contain columns '{self.path_col}' and '{self.label_col}'."
            )

        if self.split is not None and self.split_col not in rows[0]:
            raise ValueError(f"Split '{self.split}' requested, but '{self.split_col}' column is missing.")

        self.rows = []
        for row in rows:
            if self.split is not None and row.get(self.split_col) != self.split:
                continue
            path = Path(row[self.path_col])
            if not path.is_absolute():
                path = (self.manifest_path.parent / path).resolve()
            if not path.exists():
                continue
            self.rows.append(
                {
                    "path": str(path),
                    "label": row[self.label_col],
                }
            )

        if not self.rows:
            raise ValueError("No rows left after filtering by split / paths.")

        self.label_to_idx = self._build_label_map([r["label"] for r in self.rows])

    @staticmethod
    def _build_label_map(labels: Sequence[str]) -> Dict[str, int]:
        uniq = sorted({str(x) for x in labels})
        return {lab: i for i, lab in enumerate(uniq)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        waveform, sr = torchaudio.load(row["path"])

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        # Duration filtering
        if self.min_seconds is not None or self.max_seconds is not None:
            dur = waveform.shape[-1] / float(self.target_sr)
            if self.min_seconds is not None and dur < self.min_seconds:
                raise RuntimeError(f"Audio too short after loading: {row['path']}")
            if self.max_seconds is not None and dur > self.max_seconds:
                max_samples = int(self.max_seconds * self.target_sr)
                waveform = waveform[..., :max_samples]

        label_idx = self.label_to_idx[str(row["label"])]
        return {
            "waveform": waveform.squeeze(0),  # [T]
            "label": label_idx,
            "path": row["path"],
        }


def pad_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    waveforms = [torch.as_tensor(x["waveform"], dtype=torch.float32) for x in batch]
    lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)
    labels = torch.tensor([int(x["label"]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, w in enumerate(waveforms):
        padded[i, : w.shape[-1]] = w
    return {
        "waveforms": padded,
        "lengths": lengths,
        "labels": labels,
        "paths": [x["path"] for x in batch],
    }



# Augmentation


class AudioAugment(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        p_noise: float = 0.4,
        p_gain: float = 0.5,
        p_speed: float = 0.3,
        p_roll: float = 0.2,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.p_noise = p_noise
        self.p_gain = p_gain
        self.p_speed = p_speed
        self.p_roll = p_roll

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, T] or [T]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        out = wav.clone()

        if torch.rand(1).item() < self.p_gain:
            gain = torch.empty(out.shape[0], 1, device=out.device).uniform_(0.7, 1.3)
            out = out * gain

        if torch.rand(1).item() < self.p_roll:
            shift = int(torch.randint(low=0, high=max(1, out.shape[-1]), size=(1,)).item())
            out = torch.roll(out, shifts=shift, dims=-1)

        if torch.rand(1).item() < self.p_speed:
            speed = float(torch.empty(1).uniform_(0.9, 1.1).item())
            new_len = max(1, int(out.shape[-1] / speed))
            out = torch.nn.functional.interpolate(
                out.unsqueeze(1), size=new_len, mode="linear", align_corners=False
            ).squeeze(1)
            if out.shape[-1] > wav.shape[-1]:
                out = out[..., : wav.shape[-1]]
            elif out.shape[-1] < wav.shape[-1]:
                pad = wav.shape[-1] - out.shape[-1]
                out = F.pad(out, (0, pad))

        if torch.rand(1).item() < self.p_noise:
            noise = torch.randn_like(out) * torch.empty(1).uniform_(0.001, 0.01).item()
            out = out + noise

        return out.squeeze(0) if out.shape[0] == 1 else out



# Feature helpers


class LogMelExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: float = 20.0,
        f_max: Optional[float] = 7600.0,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            normalized=False,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        feat = self.mel(wav)
        feat = self.db(feat)
        return feat  # [B, M, T]


class StatsPooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        mean = x.mean(dim=-1)
        std = x.std(dim=-1).clamp_min(1e-5)
        return torch.cat([mean, std], dim=1)



# Models


class XVectorTDNN(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 192, feat_dim: int = 80) -> None:
        super().__init__()
        self.feats = LogMelExtractor(n_mels=feat_dim)
        self.tdnn = nn.Sequential(
            nn.Conv1d(feat_dim, 256, kernel_size=5, dilation=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=3, dilation=3, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        self.pool = StatsPooling()
        self.segment = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, emb_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.feats(wav)  # [B, M, T]
        x = self.tdnn(feats)
        pooled = self.pool(x)
        emb = self.segment(pooled)
        logits = self.classifier(emb)
        return logits, emb


class FrozenEmbeddingClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, emb_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes),
        )

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(wav)
        logits = self.classifier(emb)
        return logits, emb


class ECAPAEncoder(nn.Module):
    def __init__(self, model_id: str = "speechbrain/spkrec-ecapa-voxceleb") -> None:
        super().__init__()
        if EncoderClassifier is None:
            raise ImportError("speechbrain is not installed.")
        self.model = EncoderClassifier.from_hparams(source=model_id)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # SpeechBrain expects [B, T] tensor
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        emb = self.model.encode_batch(wav)
        emb = emb.squeeze(1) if emb.dim() == 3 else emb
        return emb


class WavLMEncoder(nn.Module):
    def __init__(
        self,
        model_id: str = "microsoft/wavlm-base-sv",
        use_last_n: int = 4,
        target_sr: int = 16000,
    ) -> None:
        super().__init__()
        if AutoFeatureExtractor is None or AutoModel is None:
            raise ImportError("transformers is not installed.")
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.backbone = AutoModel.from_pretrained(model_id)
        self.target_sr = target_sr
        self.use_last_n = use_last_n
        hidden = self.backbone.config.hidden_size
        self.layer_weights = nn.Parameter(torch.ones(use_last_n))
        self.proj = nn.Linear(hidden * 2, 256)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _pool(self, hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # hidden: [B, T, H]
        if mask is None:
            mean = hidden.mean(dim=1)
            std = hidden.std(dim=1).clamp_min(1e-5)
            return torch.cat([mean, std], dim=-1)
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean = (hidden * mask).sum(dim=1) / denom
        var = ((hidden - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / denom
        std = torch.sqrt(var.clamp_min(1e-5))
        return torch.cat([mean, std], dim=-1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        device = wav.device
        inputs = self.extractor(
            [x.cpu().numpy() for x in wav],
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        out = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = out.hidden_states[-self.use_last_n :]
        weights = torch.softmax(self.layer_weights, dim=0)
        stacked = torch.stack([h for h in hidden_states], dim=0)  # [L, B, T, H]
        mixed = (weights[:, None, None, None] * stacked).sum(dim=0)
        pooled = self._pool(mixed, attention_mask)
        emb = self.proj(pooled)
        return emb


# Train / Eval

@dataclasses.dataclass
class Metrics:
    loss: float
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    result = {}
    if accuracy_score is not None:
        result["accuracy"] = float(accuracy_score(y_true, y_pred))
    else:
        result["accuracy"] = float(np.mean(np.array(y_true) == np.array(y_pred)))
    if f1_score is not None:
        result["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    return result


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    augment: Optional[nn.Module] = None,
    grad_clip: float = 5.0,
) -> Metrics:
    model.train()
    if augment is not None:
        augment.train()
    total_loss = 0.0
    total = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        wav = batch["waveforms"].to(device)
        labels = batch["labels"].to(device)

        if augment is not None:
            wav = augment(wav)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(wav)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=-1).detach().cpu().tolist())

    met = compute_metrics(y_true, y_pred)
    return Metrics(loss=total_loss / max(1, total), accuracy=met.get("accuracy"), macro_f1=met.get("macro_f1"))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    total_loss = 0.0
    total = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        wav = batch["waveforms"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = model(wav)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=-1).detach().cpu().tolist())

    met = compute_metrics(y_true, y_pred)
    return Metrics(loss=total_loss / max(1, total), accuracy=met.get("accuracy"), macro_f1=met.get("macro_f1"))


def build_model(model_name: str, num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "xvector":
        return XVectorTDNN(num_classes=num_classes)

    if model_name == "ecapa":
        encoder = ECAPAEncoder()
        if freeze_backbone:
            for p in encoder.parameters():
                p.requires_grad = False

        # SpeechBrain ECAPA embeddings are commonly 192-d.
        return FrozenEmbeddingClassifier(encoder=encoder, emb_dim=192, num_classes=num_classes)

    if model_name == "wavlm":
        encoder = WavLMEncoder()
        if freeze_backbone:
            encoder.freeze_backbone()
        return FrozenEmbeddingClassifier(encoder=encoder, emb_dim=256, num_classes=num_classes)

    raise ValueError(f"Unknown model: {model_name}")


def make_loaders(
    manifest: str,
    batch_size: int,
    split_col: str,
    train_split: str,
    valid_split: Optional[str],
    target_sr: int,
    max_seconds: Optional[float],
) -> Tuple[DataLoader, Optional[DataLoader], Dict[str, int]]:
    train_ds = SpeakerManifestDataset(
        manifest_path=manifest,
        split=train_split,
        split_col=split_col,
        target_sr=target_sr,
        max_seconds=max_seconds,
    )
    label_map = train_ds.label_to_idx

    valid_loader = None
    if valid_split is not None:
        valid_ds = SpeakerManifestDataset(
            manifest_path=manifest,
            split=valid_split,
            split_col=split_col,
            target_sr=target_sr,
            max_seconds=max_seconds,
        )
        valid_ds.label_to_idx = label_map
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate, num_workers=0)
    return train_loader, valid_loader, label_map


def save_checkpoint(path: str | Path, model: nn.Module, label_map: Dict[str, int], args: argparse.Namespace) -> None:
    ckpt = {
        "model_name": args.model,
        "state_dict": model.state_dict(),
        "label_map": label_map,
        "args": vars(args),
    }
    torch.save(ckpt, str(path))


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, object]:
    return torch.load(str(path), map_location=map_location)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["ecapa", "wavlm", "xvector"], required=True)
    parser.add_argument("--split-col", type=str, default="split")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--valid-split", type=str, default="valid")
    parser.add_argument("--no-valid", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze pretrained encoder for ECAPA/WavLM.")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=True)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--out", type=str, default="checkpoint.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_split = None if args.no_valid else args.valid_split
    train_loader, valid_loader, label_map = make_loaders(
        manifest=args.manifest,
        batch_size=args.batch_size,
        split_col=args.split_col,
        train_split=args.train_split,
        valid_split=valid_split,
        target_sr=args.target_sr,
        max_seconds=args.max_seconds,
    )

    num_classes = len(label_map)
    model = build_model(args.model, num_classes=num_classes, freeze_backbone=args.freeze_backbone).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found. Check freeze_backbone or the model definition.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    augment = AudioAugment(sample_rate=args.target_sr) if args.augment else None

    best_val = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, device, augment=augment)
        row = {
            "epoch": epoch,
            "train_loss": train_m.loss,
            "train_acc": train_m.accuracy,
            "train_macro_f1": train_m.macro_f1,
        }

        if valid_loader is not None:
            val_m = evaluate(model, valid_loader, device)
            row.update(
                {
                    "val_loss": val_m.loss,
                    "val_acc": val_m.accuracy,
                    "val_macro_f1": val_m.macro_f1,
                }
            )
            score = val_m.accuracy if val_m.accuracy is not None else -1.0
            if score > best_val:
                best_val = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    if best_state is not None:
        model.load_state_dict(best_state)

    save_checkpoint(args.out, model, label_map, args)

    history_path = Path(args.out).with_suffix(".history.json")
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to: {args.out}")
    print(f"Saved history to: {history_path}")


if __name__ == "__main__":
    main()
