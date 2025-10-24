"""
Industrial Continual Learning Evaluation Metrics:
- Critical Anomaly F1 (CAF1)
- Drift Adaptation Latency (DAL)
- Memory Efficiency (ME)
- Conditional Value-at-Risk @ 0.95 (CVaR@0.95)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import warnings


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, **kwargs) -> float:
        """Compute the metric value."""
        pass

    def __call__(self, **kwargs) -> float:
        return self.compute(**kwargs)


class CriticalAnomalyF1(BaseMetric):
    """
    Critical Anomaly F1 (CAF1): F1-score computed only on anomalies with severity >= threshold.
    """

    def __init__(self, severity_threshold: float = 0.7):
        super().__init__("CAF1")
        self.severity_threshold = severity_threshold

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        severity: np.ndarray
    ) -> float:
        """
        Compute CAF1.

        Args:
            y_true (np.ndarray): True labels [N]
            y_pred (np.ndarray): Predicted labels [N]
            severity (np.ndarray): Severity scores [N]

        Returns:
            float: CAF1 score
        """
        # Validate inputs
        if not (y_true.shape == y_pred.shape == severity.shape):
            raise ValueError("Input arrays must have the same shape")

        # Select critical anomalies
        mask = severity >= self.severity_threshold
        if not np.any(mask):
            warnings.warn("No critical anomalies found. CAF1 = 0.0")
            return 0.0

        y_true_crit = y_true[mask]
        y_pred_crit = y_pred[mask]

        # Compute F1
        tp = np.sum((y_true_crit == 1) & (y_pred_crit == 1))
        fp = np.sum((y_true_crit == 0) & (y_pred_crit == 1))
        fn = np.sum((y_true_crit == 1) & (y_pred_crit == 0))

        if tp == 0:
            return 0.0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        return float(f1)


class DriftAdaptationLatency(BaseMetric):
    """
    Drift Adaptation Latency (DAL): Time steps to recover >90% of peak CAF1 after drift.
    """

    def __init__(self, recovery_threshold: float = 0.9):
        super().__init__("DAL")
        self.recovery_threshold = recovery_threshold

    def compute(
        self,
        caf1_history: List[float],
        drift_start_step: int,
        pre_drift_peak: Optional[float] = None
    ) -> int:
        """
        Compute DAL.

        Args:
            caf1_history (List[float]): CAF1 values over time [T]
            drift_start_step (int): Step when drift occurs
            pre_drift_peak (float, optional): Peak CAF1 before drift

        Returns:
            int: DAL in steps (returns len(caf1_history) if not recovered)
        """
        if pre_drift_peak is None:
            # Estimate peak from pre-drift window
            pre_window = caf1_history[:max(1, drift_start_step)]
            pre_drift_peak = max(pre_window) if pre_window else 0.0

        target_caf1 = self.recovery_threshold * pre_drift_peak
        post_drift = caf1_history[drift_start_step:]

        for i, caf1 in enumerate(post_drift):
            if caf1 >= target_caf1:
                return drift_start_step + i

        return len(caf1_history)  # Not recovered


class MemoryEfficiency(BaseMetric):
    """
    Memory Efficiency (ME): Fraction of high-severity anomalies in memory buffer.
    """

    def __init__(self, severity_threshold: float = 0.7):
        super().__init__("ME")
        self.severity_threshold = severity_threshold

    def compute(
        self,
        memory_severity: np.ndarray
    ) -> float:
        """
        Compute ME.

        Args:
            memory_severity (np.ndarray): Severity scores of memory samples [M]

        Returns:
            float: Memory efficiency
        """
        if len(memory_severity) == 0:
            return 0.0

        critical_count = np.sum(memory_severity >= self.severity_threshold)
        return float(critical_count / len(memory_severity))


class CVaR(BaseMetric):
    """
    Conditional Value-at-Risk (CVaR) at confidence level alpha.
    Measures expected loss in the worst (1-alpha) fraction of critical anomalies.
    """

    def __init__(self, alpha: float = 0.95, smooth: bool = True, rho: float = 20.0):
        super().__init__(f"CVaR@{alpha}")
        self.alpha = alpha
        self.smooth = smooth
        self.rho = rho

    def compute(
        self,
        losses: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute CVaR.

        Args:
            losses (np.ndarray): Loss values [N]
            weights (np.ndarray, optional): Risk weights [N]

        Returns:
            float: CVaR value
        """
        if len(losses) == 0:
            return 0.0

        if weights is None:
            weights = np.ones_like(losses)

        weighted_losses = losses * weights

        if self.smooth:
            return self._compute_smooth_cvar(weighted_losses)
        else:
            return self._compute_exact_cvar(weighted_losses)

    def _compute_exact_cvar(self, losses: np.ndarray) -> float:
        """Exact CVaR computation using quantile."""
        quantile_idx = int(self.alpha * len(losses))
        if quantile_idx >= len(losses):
            quantile_idx = len(losses) - 1

        sorted_losses = np.sort(losses)
        tau = sorted_losses[quantile_idx]
        excess_losses = np.maximum(losses - tau, 0.0)
        cvar = tau + np.mean(excess_losses) / (1.0 - self.alpha)
        return float(cvar)

    def _compute_smooth_cvar(self, losses: np.ndarray) -> float:
        """Smooth CVaR approximation using log-sum-exp."""
        tau = self._optimize_tau_smooth(losses)
        exp_term = np.exp(self.rho * (losses - tau))
        numerator = np.sum(losses * exp_term)
        denominator = np.sum(exp_term)
        cvar = tau + (numerator / denominator - tau) / (1.0 - self.alpha)
        return float(cvar)

    def _optimize_tau_smooth(self, losses: np.ndarray) -> float:
        """Optimize tau using fixed-point iteration."""
        tau = np.mean(losses)
        for _ in range(10):
            exp_term = np.exp(self.rho * (losses - tau))
            tau_new = np.sum(losses * exp_term) / np.sum(exp_term)
            if abs(tau_new - tau) < 1e-4:
                break
            tau = tau_new
        return tau


class IndustrialMetrics:
    """
    Unified metric calculator for industrial continual learning.
    """

    def __init__(
        self,
        severity_threshold: float = 0.7,
        cvar_alpha: float = 0.95,
        dal_recovery_threshold: float = 0.9
    ):
        self.caf1_metric = CriticalAnomalyF1(severity_threshold=severity_threshold)
        self.dal_metric = DriftAdaptationLatency(recovery_threshold=dal_recovery_threshold)
        self.me_metric = MemoryEfficiency(severity_threshold=severity_threshold)
        self.cvar_metric = CVaR(alpha=cvar_alpha)

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        severity: np.ndarray,
        memory_severity: np.ndarray,
        caf1_history: Optional[List[float]] = None,
        drift_start_step: Optional[int] = None,
        pre_drift_peak: Optional[float] = None,
        undetected_losses: Optional[np.ndarray] = None,
        undetected_weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics in one call.

        Returns:
            Dict[str, float]: Dictionary of metric values
        """
        metrics = {}

        # CAF1
        metrics["CAF1"] = self.caf1_metric(y_true, y_pred, severity)

        # ME
        metrics["ME"] = self.me_metric(memory_severity)

        # CVaR
        if undetected_losses is not None:
            metrics["CVaR@0.95"] = self.cvar_metric(undetected_losses, undetected_weights)
        else:
            metrics["CVaR@0.95"] = 0.0

        # DAL
        if caf1_history is not None and drift_start_step is not None:
            metrics["DAL"] = self.dal_metric(
                caf1_history, drift_start_step, pre_drift_peak
            )
        else:
            metrics["DAL"] = 0

        return metrics

    @staticmethod
    def from_torch(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        severity: torch.Tensor,
        memory_severity: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """Convert PyTorch tensors to NumPy and compute metrics."""
        return IndustrialMetrics().compute_all(
            y_true=y_true.detach().cpu().numpy(),
            y_pred=y_pred.detach().cpu().numpy(),
            severity=severity.detach().cpu().numpy(),
            memory_severity=memory_severity.detach().cpu().numpy(),
            **kwargs
        )