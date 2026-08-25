"""Architetture di classificazione confrontate nella tesi."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_baseline(number_of_classes: int = 4, dropout: float = 0.35) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False

    feature_size = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(feature_size, number_of_classes),
    )
    return model


class TwoBranchResNet50(nn.Module):
    """ResNet50 condivisa con fusione masked max dei due pannelli renali."""

    def __init__(self, number_of_classes: int = 4, dropout: float = 0.35) -> None:
        super().__init__()
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feature_size = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_size, number_of_classes),
        )

    def forward(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = left_image.size(0)
        combined_batch = torch.cat([left_image, right_image], dim=0)
        combined_features = self.encoder(combined_batch)
        features = torch.stack(
            [combined_features[:batch_size], combined_features[batch_size:]],
            dim=1,
        )
        valid_mask = valid_mask.to(features.device)
        masked_features = features.masked_fill(
            ~valid_mask.unsqueeze(-1),
            float("-inf"),
        )
        fused_features = masked_features.max(dim=1).values
        return self.classifier(fused_features)
