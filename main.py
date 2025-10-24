"""
Entry point for DynaMAD: Dynamic Memory Adaptation for Industrial Anomaly Detection.
Supports training, evaluation, and data preprocessing.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class DynaMADRunner:
    """Main runner class that orchestrates training, evaluation, and preprocessing."""

    def __init__(self):
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging format."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("dynamad.log")
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _validate_dataset_path(self):
        """Ensure dataset root exists."""
        dataset_root = "/home/phd/datasets/"
        if not os.path.exists(dataset_root):
            self.logger.error(f"Dataset root not found: {dataset_root}")
            raise FileNotFoundError(f"Please ensure datasets are placed in {dataset_root}")
        self.logger.info(f"Dataset root verified: {dataset_root}")

    def run_training(self, config_path: str):
        """Run training pipeline."""
        self.logger.info("Starting training...")
        self._validate_dataset_path()
        
        try:
            from experiments.train import main as train_main
            # Patch sys.argv to simulate command-line call
            original_argv = sys.argv
            sys.argv = ['train.py', '--config', config_path]
            train_main()
            sys.argv = original_argv
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def run_evaluation(self, config_path: str, checkpoint_path: str):
        """Run evaluation pipeline."""
        self.logger.info("Starting evaluation...")
        self._validate_dataset_path()
        
        try:
            from experiments.evaluate import main as eval_main
            original_argv = sys.argv
            sys.argv = ['evaluate.py', '--config', config_path, '--checkpoint', checkpoint_path]
            eval_main()
            sys.argv = original_argv
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise

    def run_preprocessing(self, dataset_name: str):
        """Run dataset-specific preprocessing."""
        self.logger.info(f"Starting preprocessing for {dataset_name}...")
        
        if dataset_name == "cwru":
            try:
                from scripts.preprocess_cwru import main as preprocess_main
                preprocess_main()
            except Exception as e:
                self.logger.error(f"CWRU preprocessing failed: {e}")
                raise
        else:
            self.logger.warning(f"Preprocessing not implemented for {dataset_name}")

    def parse_args(self) -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(description="DynaMAD: Industrial Continual Anomaly Detection")
        subparsers = parser.add_subparsers(dest='mode', required=True)

        # Train
        train_parser = subparsers.add_parser('train', help='Train the model')
        train_parser.add_argument('--config', type=str, required=True, help='Path to config YAML')

        # Evaluate
        eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
        eval_parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
        eval_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')

        # Preprocess
        prep_parser = subparsers.add_parser('preprocess', help='Preprocess datasets')
        prep_parser.add_argument('--dataset', type=str, required=True, choices=['cwru'], help='Dataset to preprocess')

        return parser.parse_args()

    def run(self):
        """Main execution logic."""
        args = self.parse_args()
        
        try:
            if args.mode == 'train':
                self.run_training(args.config)
            elif args.mode == 'evaluate':
                self.run_evaluation(args.config, args.checkpoint)
            elif args.mode == 'preprocess':
                self.run_preprocessing(args.dataset)
            else:
                self.logger.error("Unknown mode")
                sys.exit(1)
        except Exception as e:
            self.logger.critical(f"Execution failed: {e}")
            sys.exit(1)

        self.logger.info("DynaMAD execution completed successfully.")


def main():
    """Entry point."""
    runner = DynaMADRunner()
    runner.run()


if __name__ == "__main__":
    main()