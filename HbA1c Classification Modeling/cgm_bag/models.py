"""Compact sequence encoders and masked patient-level multiple-instance learning.

Every segment receives a shared encoder. The patient classifier is trained only
on the pooled bag embedding; applying that head to a single segment produces a
weak, single-instance prediction rather than a separately validated diagnosis.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class CNNEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 5, padding=2),
            nn.GroupNorm(1, hidden_dim), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.GroupNorm(1, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.unsqueeze(1)).squeeze(-1)


class CausalTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = 2 * dilation
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, 3, dilation=dilation),
            nn.Conv1d(channels, channels, 3, dilation=dilation),
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(channels), nn.LayerNorm(channels)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for conv, norm in zip(self.convs, self.norms):
            x = conv(F.pad(x, (self.padding, 0)))
            x = norm(x.transpose(1, 2)).transpose(1, 2)
            x = self.dropout(F.gelu(x))
        return F.gelu(x + residual)


class TCNEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.input = nn.Conv1d(1, hidden_dim, 1)
        self.blocks = nn.Sequential(*[
            CausalTemporalBlock(hidden_dim, dilation, dropout) for dilation in (1, 2, 4, 8)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input(x.unsqueeze(1))).mean(dim=-1)


class BiGRUEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(1, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(x.unsqueeze(-1))
        return self.dropout(sequence.mean(dim=1))


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim: int, length: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(1, hidden_dim)
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        encoding = torch.zeros(length, hidden_dim)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("position_encoding", encoding.unsqueeze(0))
        heads = 4 if hidden_dim % 4 == 0 else 2
        layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x.unsqueeze(-1)) + self.position_encoding[:, :x.shape[1]]
        return self.transformer(x).mean(dim=1)


class PatientBagModel(nn.Module):
    """Fuse explicitly enabled waveform, event-feature, and clinical branches.

    Waveform models accept CGM [B,K,L]. Tabular event models retain that shape
    for compatibility but never read waveform values or instantiate a sequence
    encoder. Clinical [B,D] is shared across a patient's events; engineered
    features [B,1,D] belong to the individual event. No-CGM modes require K=1.
    """

    def __init__(self, architecture="1d_cnn", hidden_dim=32, length=64,
                 pooling="attention", dropout=0.1, metadata_dim=0, event_feature_dim=0, use_cgm=True):
        super().__init__()
        if hidden_dim < 4 or hidden_dim % 2:
            raise ValueError("hidden_dim must be an even integer >= 4.")
        if length < 4:
            raise ValueError("length must be >= 4.")
        if pooling not in {"mean", "attention"}:
            raise ValueError("pooling must be 'mean' or 'attention'.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if not isinstance(metadata_dim, int) or metadata_dim < 0:
            raise ValueError("metadata_dim must be a non-negative integer.")
        if not isinstance(event_feature_dim, int) or event_feature_dim < 0:
            raise ValueError("event_feature_dim must be a non-negative integer.")
        if not isinstance(use_cgm, bool):
            raise ValueError("use_cgm must be a boolean.")
        if not use_cgm and (architecture != "mlp" or not (metadata_dim or event_feature_dim)):
            raise ValueError("Without CGM, choose architecture='mlp' and at least one metadata or event-feature branch.")
        constructors = {"1d_cnn": CNNEncoder, "tcn": TCNEncoder, "bigru": BiGRUEncoder}
        if not use_cgm:
            pass  # No unused CGM encoder parameters or operations in tabular ablations.
        elif architecture == "transformer":
            self.encoder = TransformerEncoder(hidden_dim, length, dropout)
        elif architecture in constructors:
            self.encoder = constructors[architecture](hidden_dim, dropout)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.length = length
        self.architecture = architecture
        self.use_cgm = use_cgm
        self.pooling = pooling
        self.hidden_dim = hidden_dim
        self.metadata_dim = metadata_dim
        self.event_feature_dim = event_feature_dim
        if use_cgm:
            self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        # Do not register additional parameters in CGM-only models: existing
        # checkpoints retain their original state_dict layout.
        if metadata_dim:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(metadata_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            )
        if event_feature_dim:
            self.event_feature_encoder = nn.Sequential(
                nn.Linear(event_feature_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            )
        self.classifier = nn.Linear(hidden_dim * (use_cgm + bool(metadata_dim) + bool(event_feature_dim)), 3)

    def forward(self, values: torch.Tensor, mask: torch.Tensor,
                metadata: torch.Tensor | None = None,
                event_features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if values.ndim != 3 or values.shape[-1] != self.length:
            raise ValueError(f"Expected bags [batch, K, {self.length}].")
        if mask.shape != values.shape[:2]:
            raise ValueError("The bag mask shape must match [batch, K].")
        mask = mask.bool()
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every patient bag must contain at least one valid segment.")
        batch, k, _ = values.shape
        if not self.use_cgm and k != 1:
            raise ValueError("Tabular ablations without CGM require event mode with K=1.")
        if self.event_feature_dim:
            if k != 1:
                raise ValueError("Unified CGM event features currently require event mode with K=1.")
            if (event_features is None or event_features.shape != (batch, k, self.event_feature_dim)
                    or not torch.isfinite(event_features).all()):
                raise ValueError(f"Expected finite encoded event features [batch, 1, {self.event_feature_dim}].")
        if self.metadata_dim:
            if metadata is None:
                raise ValueError("This model requires encoded clinical metadata.")
            if metadata.shape != (batch, self.metadata_dim) or not torch.isfinite(metadata).all():
                raise ValueError(f"Expected finite metadata with shape [batch, {self.metadata_dim}].")
        pooled_parts, event_parts = [], []
        weights = mask.to(values.dtype) / mask.sum(dim=1, keepdim=True)
        if self.use_cgm:
            flat_mask = mask.reshape(-1)
            valid_embeddings = self.encoder(values.reshape(-1, self.length)[flat_mask])
            embeddings = values.new_zeros((batch * k, self.hidden_dim))
            embeddings[flat_mask] = valid_embeddings
            embeddings = embeddings.reshape(batch, k, self.hidden_dim)
            if self.pooling == "attention":
                scores = self.attention(embeddings).squeeze(-1).masked_fill(~mask, -torch.inf)
                weights = torch.softmax(scores, dim=1)
            pooled_parts.append((embeddings * weights.unsqueeze(-1)).sum(dim=1))
            event_parts.append(embeddings)
        if self.event_feature_dim:
            feature_embedding = self.event_feature_encoder(event_features)
            pooled_parts.append(feature_embedding[:, 0, :])
            event_parts.append(feature_embedding)
        if self.metadata_dim:
            clinical_embedding = self.metadata_encoder(metadata)
            pooled_parts.append(clinical_embedding)
            event_parts.append(clinical_embedding[:, None, :].expand(-1, k, -1))
        pooled = torch.cat(pooled_parts, dim=-1)
        embeddings = torch.cat(event_parts, dim=-1)
        return {
            "patient_logits": self.classifier(pooled),
            "segment_logits": self.classifier(embeddings),
            "attention": weights,
        }


def build_model(config: Any = None, **kwargs) -> PatientBagModel:
    """Build from TrainConfig, a configuration mapping, or explicit keywords."""
    names = ("architecture", "hidden_dim", "length", "pooling", "dropout", "metadata_dim", "event_feature_dim", "use_cgm")
    if config is not None:
        values = {name: config[name] for name in names if name in config} if isinstance(config, dict) else {
            name: getattr(config, name) for name in names if hasattr(config, name)
        }
        values.update(kwargs)
        kwargs = values
    return PatientBagModel(**kwargs)
