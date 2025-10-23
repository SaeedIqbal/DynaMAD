"""
ResNet-18 Backbone for Industrial Anomaly Detection.
Supports pretrained loading (ImageNet) and flexible output heads.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 backbone adapted for industrial anomaly detection.

    Attributes:
        features (nn.Sequential): Feature extractor (all layers except final FC).
        fc_in_features (int): Number of input features to the final classifier.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        freeze_backbone: bool = False
    ) -> None:
        """
        Initialize the ResNet-18 backbone.

        Args:
            pretrained (bool): If True, load ImageNet-pretrained weights.
            num_classes (int): Number of output classes (e.g., 2 for normal/anomaly).
            freeze_backbone (bool): If True, freeze all layers except the final classifier.
        """
        super().__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        # Load base ResNet-18
        if self.pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            base_model = resnet18(weights=weights)
        else:
            base_model = resnet18(weights=None)

        # Extract feature layers (all except final FC)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc_in_features = base_model.fc.in_features

        # Replace final classifier
        self.classifier = nn.Linear(self.fc_in_features, self.num_classes)

        # Optionally freeze backbone
        if self.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze all parameters in the feature extractor."""
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        # Ensure input is 4D
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got shape {x.shape}")

        # Feature extraction
        features = self.features(x)  # (B, 512, 1, 1)
        features = torch.flatten(features, 1)  # (B, 512)

        # Classification
        logits = self.classifier(features)  # (B, num_classes)
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embedding (before classifier).

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Embedding of shape (B, 512).
        """
        with torch.no_grad():
            features = self.features(x)
            return torch.flatten(features, 1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"pretrained={self.pretrained}, "
            f"num_classes={self.num_classes}, "
            f"freeze_backbone={self.freeze_backbone})"
        )


def load_resnet18_backbone(
    pretrained: bool = True,
    num_classes: int = 2,
    freeze_backbone: bool = False,
    device: Optional[Union[str, torch.device]] = None
) -> ResNet18Backbone:
    """
    Factory function to instantiate and optionally move the backbone to a device.

    Args:
        pretrained (bool): Load ImageNet weights.
        num_classes (int): Number of output classes.
        freeze_backbone (bool): Freeze feature extractor.
        device (str or torch.device, optional): Device to load model onto.

    Returns:
        ResNet18Backbone: Initialized and optionally device-placed model.
    """
    model = ResNet18Backbone(
        pretrained=pretrained,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )

    if device is not None:
        model = model.to(device)

    return model