# experiments/evaluate.py
"""
Inference and Metric Computation for DynaMAD.
Evaluates trained models on industrial anomaly detection benchmarks.
Designed with OOP best practices: encapsulation, modularity, and type safety.
"""

import os
import argparse
import torch
import yaml
from typing import Dict, Any, Optional
from datetime import datetime

from models import ContinualLearningModel, load_resnet18_backbone
from data.dataloaders import get_streaming_loader
from utils.metrics import IndustrialMetrics
from utils.severity import get_severity_mapper


class DynaMADEvaluator:
    """Evaluator class for DynaMAD inference and metrics."""

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: str,
        device: torch.device
    ):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dataset_name = config['dataset']['name']
        self.results_dir = os.path.join(
            config['evaluation']['results_dir'],
            f"{self.dataset_name}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize components
        self._setup_severity_mapper()
        self._setup_dataloader()
        self._load_model()

    def _setup_severity_mapper(self):
        """Initialize FMEA-based severity mapper."""
        self.severity_mapper = get_severity_mapper(
            dataset_name=self.dataset_name,
            config_path=self.config['dataset'].get('fmea_config')
        )

    def _setup_dataloader(self):
        """Initialize evaluation data loader."""
        self.eval_loader = get_streaming_loader(
            dataset_name=self.dataset_name,
            data_root=self.config['dataset']['root'],
            batch_size=self.config['evaluation']['batch_size'],
            num_workers=self.config['evaluation']['num_workers'],
            shuffle=False  # Deterministic evaluation
        )

    def _load_model(self):
        """Load trained model from checkpoint."""
        # Load backbone
        backbone = load_resnet18_backbone(
            pretrained=False,  # Weights will be loaded from checkpoint
            num_classes=2,
            device=self.device
        )

        # Reconstruct model (components will be loaded from state dict)
        self.model = ContinualLearningModel(
            backbone=backbone,
            device=self.device
        )

        # Load checkpoint
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded model from {self.checkpoint_path}")

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

    def evaluate(self) -> Dict[str, float]:
        """Run full evaluation and compute metrics."""
        all_preds = []
        all_labels = []
        all_severities = []
        memory_severities = []

        print("Starting evaluation...")
        with torch.no_grad():
            for batch in self.eval_loader:
                # Assign severity
                batch = self._assign_severity(batch)
                
                # Forward pass
                logits = self.model(batch)
                preds = torch.argmax(logits, dim=1)
                
                # Collect results
                all_preds.append(preds.cpu())
                all_labels.append(batch['label'].cpu())
                all_severities.append(batch['severity'].cpu())
                
                # Simulate memory buffer for ME computation
                memory_severities.extend(batch['severity'].cpu().numpy())

        # Concatenate results
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)
        severity = torch.cat(all_severities)
        memory_severity = torch.tensor(memory_severities)

        # Compute metrics
        metrics = IndustrialMetrics(
            severity_threshold=self.config['metrics'].get('severity_threshold', 0.7),
            cvar_alpha=self.config['metrics'].get('cvar_alpha', 0.95),
            dal_recovery_threshold=self.config['metrics'].get('dal_recovery_threshold', 0.9)
        ).from_torch(
            y_true=y_true,
            y_pred=y_pred,
            severity=severity,
            memory_severity=memory_severity
        )

        # Save results
        self._save_results(metrics)
        return metrics

    def _save_results(self, metrics: Dict[str, float]):
        """Save evaluation results to file."""
        results_path = os.path.join(self.results_dir, 'evaluation_results.yaml')
        with open(results_path, 'w') as f:
            yaml.dump({
                'dataset': self.dataset_name,
                'checkpoint': self.checkpoint_path,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }, f, default_flow_style=False)
        print(f"Results saved to {results_path}")

        # Print formatted results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for metric, value in metrics.items():
            if metric == 'DAL':
                print(f"{metric}: {int(value)} steps")
            else:
                print(f"{metric}: {value:.4f}")
        print("="*50)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Evaluate DynaMAD')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Set dataset root
    config['dataset']['root'] = '/home/phd/datasets/'

    # Initialize evaluator
    device = torch.device(args.device)
    evaluator = DynaMADEvaluator(config, args.checkpoint, device)

    # Run evaluation
    metrics = evaluator.evaluate()
    print(f"\nEvaluation completed. Final CAF1: {metrics['CAF1']:.4f}")


if __name__ == '__main__':
    main()