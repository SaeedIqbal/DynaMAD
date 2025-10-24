"""
Main Continual Learning (CL) Model f_θ for Industrial Anomaly Detection.
Integrates backbone, memory transformer, value-aware policy, and drift-adaptive dynamics.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import ResNet18Backbone
from models.memory_transformer import (
    HeterogeneousDynamics,
    ValueAwarePolicy,
    DriftAdaptiveSDEPolicy
)


class ContinualLearningModel(nn.Module):
    """
    Main CL model that unifies backbone, memory transformation, and risk-aware replay.
    """

    def __init__(
        self,
        backbone: nn.Module,
        heterogeneous_dynamics: Optional[HeterogeneousDynamics] = None,
        value_aware_policy: Optional[ValueAwarePolicy] = None,
        drift_adaptive_policy: Optional[DriftAdaptiveSDEPolicy] = None,
        cvar_weight: float = 0.18,
        js_weight: float = 0.42,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.backbone = backbone
        self.heterogeneous_dynamics = heterogeneous_dynamics
        self.value_aware_policy = value_aware_policy
        self.drift_adaptive_policy = drift_adaptive_policy
        self.cvar_weight = cvar_weight
        self.js_weight = js_weight
        self.device = device
        self.to(device)

    def get_embedding(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract embedding from multi-modal input.
        Assumes 'image' is the primary modality for backbone.
        """
        if 'image' in x:
            return self.backbone.get_embedding(x['image'].to(self.device))
        elif 'vibration' in x:
            # Fallback: use vibration as 1D image
            vib = x['vibration'].to(self.device)
            vib_img = vib.unsqueeze(1).expand(-1, 3, -1)  # [B, 3, T]
            vib_img = F.interpolate(vib_img, size=(224, 224), mode='linear', align_corners=False)
            return self.backbone.get_embedding(vib_img)
        else:
            raise ValueError("No supported modality found in input")

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through backbone."""
        if 'image' in x:
            return self.backbone(x['image'].to(self.device))
        elif 'vibration' in x:
            vib = x['vibration'].to(self.device)
            vib_img = vib.unsqueeze(1).expand(-1, 3, -1)
            vib_img = F.interpolate(vib_img, size=(224, 224), mode='linear', align_corners=False)
            return self.backbone(vib_img)
        else:
            raise ValueError("No supported modality found in input")

    def transform_memory(
        self,
        memory_batch: Dict[str, torch.Tensor],
        stream_batch: Dict[str, torch.Tensor],
        memory_buffer: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Transform memory batch using heterogeneous dynamics and drift adaptation.
        """
        if self.heterogeneous_dynamics is None:
            return memory_batch

        # Get embeddings for drift adaptation
        with torch.no_grad():
            stream_emb = self.get_embedding(stream_batch)
            buffer_emb = self.get_embedding(memory_buffer)

        # Apply drift-adaptive SDE per modality
        transformed = {}
        for modality, data in memory_batch.items():
            if modality in ['image', 'vibration', 'sensor']:
                # Use drift-adaptive policy if available
                if self.drift_adaptive_policy is not None:
                    transformed[modality] = self.drift_adaptive_policy(
                        data.to(self.device),
                        stream_emb,
                        buffer_emb
                    )
                else:
                    # Fallback to heterogeneous dynamics
                    temp_input = {modality: data.to(self.device)}
                    temp_output = self.heterogeneous_dynamics(temp_input)
                    transformed[modality] = temp_output[modality]
            else:
                transformed[modality] = data.to(self.device)

        return transformed

    def compute_js_consistency(
        self,
        x_original: Dict[str, torch.Tensor],
        x_transformed: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute Jensen-Shannon consistency loss."""
        logits_orig = self(x_original)
        logits_trans = self(x_transformed)
        
        p_orig = F.softmax(logits_orig, dim=1)
        p_trans = F.softmax(logits_trans, dim=1)
        p_mean = 0.5 * (p_orig + p_trans)
        
        js_div = 0.5 * (
            F.kl_div(p_mean.log(), p_orig, reduction='batchmean') +
            F.kl_div(p_mean.log(), p_trans, reduction='batchmean')
        )
        return js_div

    def compute_cvar_loss(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        severity: torch.Tensor
    ) -> torch.Tensor:
        """Compute CVaR loss for critical anomalies."""
        if self.value_aware_policy is None:
            return torch.tensor(0.0, device=self.device)
            
        return self.value_aware_policy.compute_cvar_loss(
            x, y, severity, self
        )

    def train_step(
        self,
        stream_batch: Dict[str, torch.Tensor],
        memory_batch: Dict[str, torch.Tensor],
        memory_buffer: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Complete training step with memory transformation and risk-aware loss.
        """
        # Transform memory
        transformed_memory = self.transform_memory(
            memory_batch, stream_batch, memory_buffer
        )

        # Stream loss
        stream_loss = F.cross_entropy(
            self(stream_batch),
            stream_batch['label'].to(self.device)
        )

        # Memory loss
        memory_loss = F.cross_entropy(
            self(transformed_memory),
            memory_batch['label'].to(self.device)
        )

        # CVaR loss
        cvar_loss = self.compute_cvar_loss(
            memory_batch,
            memory_batch['label'].to(self.device),
            memory_batch['severity'].to(self.device)
        )

        # JS consistency loss
        js_loss = self.compute_js_consistency(memory_batch, transformed_memory)

        # Total loss
        total_loss = (
            stream_loss +
            memory_loss +
            self.cvar_weight * cvar_loss +
            self.js_weight * js_loss
        )

        return total_loss

    def update(self, optimizer: torch.optim.Optimizer):
        """Update model parameters."""
        optimizer.step()
        optimizer.zero_grad()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"cvar_weight={self.cvar_weight}, "
            f"js_weight={self.js_weight})"
        )