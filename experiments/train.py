"""
Full Training Pipeline for DynaMAD: Dynamic Memory Adaptation for Industrial Anomaly Detection.
Integrates heterogeneous dynamics, value-aware policy, drift-adaptive SDEs, and reservoir sampling.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from datetime import datetime

from models import (
    load_resnet18_backbone,
    ContinualLearningModel,
    HeterogeneousDynamics,
    ValueAwarePolicy,
    DriftAdaptiveSDEPolicy
)
from data.dataloaders import get_streaming_loader
from utils.reservoir import MultiModalReservoirBuffer
from utils.metrics import IndustrialMetrics
from utils.severity import get_severity_mapper


class DynaMADTrainer:
    """Main trainer class for DynaMAD."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.config = config
        self.device = device
        self.dataset_name = config['dataset']['name']
        self.checkpoint_dir = os.path.join(
            config['training']['checkpoint_dir'],
            f"{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize components
        self._setup_severity_mapper()
        self._setup_dataloaders()
        self._setup_model()
        self._setup_buffer()
        self._setup_optimizer()
        self._setup_metrics()

    def _setup_severity_mapper(self):
        """Initialize FMEA-based severity mapper."""
        self.severity_mapper = get_severity_mapper(
            dataset_name=self.dataset_name,
            config_path=self.config['dataset'].get('fmea_config')
        )

    def _setup_dataloaders(self):
        """Initialize streaming data loader."""
        self.train_loader = get_streaming_loader(
            dataset_name=self.dataset_name,
            data_root=self.config['dataset']['root'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers']
        )

    def _setup_model(self):
        """Initialize CL model with all components."""
        # Backbone
        backbone = load_resnet18_backbone(
            pretrained=True,
            num_classes=2,
            device=self.device
        )

        # Heterogeneous dynamics
        hetero_config = self.config.get('heterogeneous_dynamics', {})
        heterogeneous_dynamics = HeterogeneousDynamics(hetero_config) if hetero_config else None

        # Value-aware policy
        value_config = self.config.get('value_aware_policy', {})
        value_aware_policy = ValueAwarePolicy(**value_config) if value_config else None

        # Drift-adaptive policy
        drift_config = self.config.get('drift_adaptive_sde', {})
        if drift_config:
            # Dummy drift network (replace with learned network in practice)
            drift_net = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            ).to(self.device)
            drift_adaptive_policy = DriftAdaptiveSDEPolicy(
                drift_net=drift_net,
                base_diffusion=drift_config.get('base_diffusion', 0.25),
                drift_gain=drift_config.get('drift_gain', 1.8),
                rho=drift_config.get('rho', 0.05),
                T=drift_config.get('T', 0.03),
                steps=drift_config.get('steps', 10)
            )
        else:
            drift_adaptive_policy = None

        # CL model
        self.model = ContinualLearningModel(
            backbone=backbone,
            heterogeneous_dynamics=heterogeneous_dynamics,
            value_aware_policy=value_aware_policy,
            drift_adaptive_policy=drift_adaptive_policy,
            cvar_weight=self.config['training'].get('cvar_weight', 0.18),
            js_weight=self.config['training'].get('js_weight', 0.42),
            device=self.device
        )

    def _setup_buffer(self):
        """Initialize memory buffer."""
        buffer_config = self.config['memory_buffer']
        self.buffer = MultiModalReservoirBuffer(
            capacity=buffer_config['size'],
            required_modalities=buffer_config.get('required_modalities', ['image', 'label', 'severity'])
        )

    def _setup_optimizer(self):
        """Initialize optimizer."""
        opt_config = self.config['optimizer']
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 1e-4)
        )

    def _setup_metrics(self):
        """Initialize metric calculator."""
        self.metrics = IndustrialMetrics(
            severity_threshold=self.config['metrics'].get('severity_threshold', 0.7),
            cvar_alpha=self.config['metrics'].get('cvar_alpha', 0.95),
            dal_recovery_threshold=self.config['metrics'].get('dal_recovery_threshold', 0.9)
        )
        self.caf1_history = []
        self.drift_start_step = None

    def _assign_severity(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Assign severity scores using FMEA mapper."""
        batch_cpu = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        severities = []
        for i in range(len(batch['label'])):
            anomaly_info = {
                'type': batch.get('anomaly_type', ['unknown'] * len(batch['label']))[i],
                'size': batch.get('defect_size', [0.0] * len(batch['label']))[i]
            }
            severity = self.severity_mapper.map_severity(anomaly_info)
            severities.append(severity)
        batch['severity'] = torch.tensor(severities, dtype=torch.float32).to(self.device)
        return batch

    def train_epoch(self, epoch: int):
        """Train one epoch."""
        self.model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            # Assign severity scores
            batch = self._assign_severity(batch)

            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Add to buffer
            for i in range(len(batch['label'])):
                item = {k: v[i] for k, v in batch.items()}
                self.buffer.add(item)

            # Sample from buffer
            if len(self.buffer) > 0:
                memory_batch = self.buffer.sample(self.config['training']['batch_size'])
                memory_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in memory_batch.items()}
                memory_buffer = {
                    'x': torch.stack([self.model.backbone.get_embedding(
                        {'image': img.unsqueeze(0)} if 'image' in item else {'vibration': vib.unsqueeze(0)}
                    ).squeeze(0) for item in [self.buffer.buffer[j] for j in range(min(500, len(self.buffer))))]
                    ),
                    'y': torch.tensor([self.buffer.buffer[j]['label'] for j in range(min(500, len(self.buffer)))]).to(self.device),
                    'severity': torch.tensor([self.buffer.buffer[j]['severity'] for j in range(min(500, len(self.buffer)))]).to(self.device)
                }
            else:
                memory_batch = batch
                memory_buffer = {'x': torch.empty(0, 512), 'y': torch.empty(0), 'severity': torch.empty(0)}

            # Training step
            loss = self.model.train_step(batch, memory_batch, memory_buffer)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss.item()

            # Log metrics every 100 steps
            if step % 100 == 0:
                self._log_metrics(step, epoch_loss / (step + 1))

        return epoch_loss / len(self.train_loader)

    def _log_metrics(self, step: int, avg_loss: float):
        """Compute and log evaluation metrics."""
        if len(self.buffer) == 0:
            return

        # Sample for evaluation
        eval_batch = self.buffer.sample(min(100, len(self.buffer)))
        eval_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}

        with torch.no_grad():
            logits = self.model(eval_batch)
            preds = torch.argmax(logits, dim=1)

        # Compute metrics
        metric_values = self.metrics.from_torch(
            y_true=eval_batch['label'],
            y_pred=preds,
            severity=eval_batch['severity'],
            memory_severity=torch.tensor([item['severity'] for item in self.buffer.buffer]).to(self.device)
        )

        # Track CAF1 for DAL
        self.caf1_history.append(metric_values['CAF1'])
        if self.drift_start_step is None and len(self.caf1_history) > 100:
            # Simulate drift at step 100
            self.drift_start_step = 100

        print(f"Step {step}: Loss={avg_loss:.4f}, " + 
              f"CAF1={metric_values['CAF1']:.3f}, " +
              f"DAL={metric_values['DAL']}, " +
              f"ME={metric_values['ME']:.3f}, " +
              f"CVaR={metric_values['CVaR@0.95']:.3f}")

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)

    def train(self):
        """Full training loop."""
        num_epochs = self.config['training']['num_epochs']
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")
            
            if epoch % self.config['training']['save_every'] == 0:
                self.save_checkpoint(epoch)

        # Save final model
        self.save_checkpoint(num_epochs)
        print(f"\nTraining completed. Checkpoints saved to: {self.checkpoint_dir}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train DynaMAD')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Set dataset root
    config['dataset']['root'] = '/home/phd/datasets/'

    # Initialize trainer
    device = torch.device(args.device)
    trainer = DynaMADTrainer(config, device)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()