"""
Value-Aware Transformation Policy for Critical Anomaly Preservation.
Implements utility scoring (rarity, severity, drift sensitivity) and CVaR@0.95 risk regularization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
import math


class BaseUtilityScorer(ABC):
    """Abstract base class for utility scoring."""

    def __init__(self, beta: float = 2.8, gamma: float = 1.6, eta: float = 0.9, T0: float = 0.05):
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.T0 = T0

    @abstractmethod
    def compute_rarity(
        self,
        x: torch.Tensor,
        memory_buffer: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> torch.Tensor:
        """Compute anomaly rarity via kernel density in embedding space."""
        pass

    @abstractmethod
    def compute_drift_sensitivity(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        model: torch.nn.Module
    ) -> torch.Tensor:
        """Compute drift sensitivity via gradient norm."""
        pass

    def compute_utility(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        severity: torch.Tensor,
        memory_buffer: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> torch.Tensor:
        """
        Compute utility score: U(x) = sigmoid(beta * [log(1/rarity) + gamma * severity + eta * drift_sens])
        
        Args:
            x (torch.Tensor): Input data [B, ...]
            y (torch.Tensor): Labels [B]
            severity (torch.Tensor): Severity scores [B]
            memory_buffer (Dict): Memory buffer with keys 'x', 'y', 'severity'
            model (nn.Module): CL model
            
        Returns:
            torch.Tensor: Utility scores [B]
        """
        rarity = self.compute_rarity(x, memory_buffer, model)  # [B]
        drift_sens = self.compute_drift_sensitivity(x, y, model)  # [B]
        
        # Avoid log(0)
        rarity = torch.clamp(rarity, min=1e-8)
        log_rarity = torch.log(1.0 / rarity)
        
        utility_input = self.beta * (log_rarity + self.gamma * severity + self.eta * drift_sens)
        return torch.sigmoid(utility_input)

    def get_horizon(self, utility: torch.Tensor) -> torch.Tensor:
        """Get transformation horizon: T_i = T0 * (1 - utility)"""
        return self.T0 * (1.0 - utility)


class KernelDensityUtilityScorer(BaseUtilityScorer):
    """Utility scorer using kernel density estimation for rarity."""

    def __init__(
        self,
        beta: float = 2.8,
        gamma: float = 1.6,
        eta: float = 0.9,
        T0: float = 0.05,
        bandwidth: float = 0.15
    ):
        super().__init__(beta, gamma, eta, T0)
        self.bandwidth = bandwidth

    def compute_rarity(
        self,
        x: torch.Tensor,
        memory_buffer: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> torch.Tensor:
        """Compute rarity via Gaussian kernel density in embedding space."""
        with torch.no_grad():
            # Get embeddings
            z_x = model.get_embedding(x)  # [B, D]
            z_mem = model.get_embedding(memory_buffer['x'])  # [M, D]
            
            # Compute pairwise distances
            dists = torch.cdist(z_x, z_mem, p=2)  # [B, M]
            
            # Gaussian kernel: exp(-||z_x - z_mem||^2 / (2 * h^2))
            kernel_vals = torch.exp(-dists.pow(2) / (2 * self.bandwidth ** 2))  # [B, M]
            
            # Mean kernel density (only over anomalous samples if available)
            if 'y' in memory_buffer:
                anomaly_mask = memory_buffer['y'] == 1
                if anomaly_mask.any():
                    kernel_vals = kernel_vals[:, anomaly_mask]
            
            return kernel_vals.mean(dim=1)  # [B]

    def compute_drift_sensitivity(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        model: torch.nn.Module
    ) -> torch.Tensor:
        """Compute drift sensitivity via gradient norm of loss."""
        x.requires_grad_(True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction='none')  # [B]
        
        # Compute gradient norm for each sample
        grad_norms = []
        for i in range(loss.shape[0]):
            grad = torch.autograd.grad(
                loss[i], x, retain_graph=True, create_graph=False
            )[0]
            grad_norm = grad[i].norm(p=2)
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms) if grad_norms else torch.zeros_like(loss)
        x.requires_grad_(False)
        return grad_norms.detach()


class CVaRCalculator:
    """Conditional Value-at-Risk (CVaR) calculator for tail risk regularization."""

    def __init__(self, alpha: float = 0.95, smooth: bool = True, rho: float = 20.0):
        self.alpha = alpha
        self.smooth = smooth
        self.rho = rho

    def compute_cvar(
        self,
        losses: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute CVaR@alpha of weighted losses.
        
        Args:
            losses (torch.Tensor): Loss values [N]
            weights (torch.Tensor, optional): Risk weights [N]
            
        Returns:
            torch.Tensor: CVaR value (scalar)
        """
        if weights is None:
            weights = torch.ones_like(losses)
            
        # Weighted losses
        weighted_losses = losses * weights
        
        if self.smooth:
            # Smooth approximation of CVaR using log-sum-exp
            tau = self._optimize_tau_smooth(weighted_losses)
            cvar = tau + (1.0 / (1.0 - self.alpha)) * self._smooth_expectation(
                weighted_losses - tau, self.rho
            )
        else:
            # Exact CVaR computation
            tau = self._optimize_tau_exact(weighted_losses)
            excess_losses = torch.clamp(weighted_losses - tau, min=0.0)
            cvar = tau + (1.0 / (1.0 - self.alpha)) * excess_losses.mean()
            
        return cvar

    def _optimize_tau_exact(self, losses: torch.Tensor) -> torch.Tensor:
        """Optimize tau for exact CVaR using quantile."""
        quantile_idx = int(self.alpha * len(losses))
        sorted_losses = torch.sort(losses).values
        return sorted_losses[quantile_idx]

    def _optimize_tau_smooth(self, losses: torch.Tensor) -> torch.Tensor:
        """Optimize tau for smooth CVaR using Newton-Raphson."""
        tau = losses.mean().detach()
        for _ in range(10):  # Max iterations
            exp_term = torch.exp(self.rho * (losses - tau))
            numerator = (losses * exp_term).sum()
            denominator = exp_term.sum()
            tau_new = numerator / denominator
            if torch.abs(tau_new - tau) < 1e-4:
                break
            tau = tau_new
        return tau.detach()

    def _smooth_expectation(self, x: torch.Tensor, rho: float) -> torch.Tensor:
        """Smooth approximation of E[max(x, 0)] using log-sum-exp."""
        return (1.0 / rho) * torch.log(torch.exp(rho * x).mean() + 1e-8)


class ValueAwarePolicy:
    """
    Complete value-aware transformation policy combining utility scoring and CVaR regularization.
    """

    def __init__(
        self,
        utility_scorer: Optional[BaseUtilityScorer] = None,
        cvar_calculator: Optional[CVaRCalculator] = None,
        beta: float = 2.8,
        gamma: float = 1.6,
        eta: float = 0.9,
        T0: float = 0.05,
        bandwidth: float = 0.15,
        cvar_alpha: float = 0.95
    ):
        if utility_scorer is None:
            self.utility_scorer = KernelDensityUtilityScorer(
                beta=beta, gamma=gamma, eta=eta, T0=T0, bandwidth=bandwidth
            )
        else:
            self.utility_scorer = utility_scorer
            
        if cvar_calculator is None:
            self.cvar_calculator = CVaRCalculator(alpha=cvar_alpha)
        else:
            self.cvar_calculator = cvar_calculator

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        severity: torch.Tensor,
        memory_buffer: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute utility scores and transformation horizons.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (utilities, horizons)
        """
        utilities = self.utility_scorer.compute_utility(
            x, y, severity, memory_buffer, model
        )
        horizons = self.utility_scorer.get_horizon(utilities)
        return utilities, horizons

    def compute_cvar_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        severity: torch.Tensor,
        model: torch.nn.Module,
        mask_anomalies_only: bool = True
    ) -> torch.Tensor:
        """
        Compute CVaR loss for critical anomalies.
        
        Args:
            x (torch.Tensor): Input data
            y (torch.Tensor): Labels
            severity (torch.Tensor): Severity scores
            model (nn.Module): CL model
            mask_anomalies_only (bool): Only compute CVaR on anomalies
            
        Returns:
            torch.Tensor: CVaR loss (scalar)
        """
        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            misclassified = (pred != y).float()
            
        if mask_anomalies_only:
            anomaly_mask = (y == 1)
            if not anomaly_mask.any():
                return torch.tensor(0.0, device=x.device)
            misclassified = misclassified[anomaly_mask]
            severity = severity[anomaly_mask]
            
        if len(misclassified) == 0:
            return torch.tensor(0.0, device=x.device)
            
        # Risk-weighted losses
        risk_weights = severity * torch.log(1.0 / (torch.ones_like(severity) * 0.01 + 1e-8))
        cvar_loss = self.cvar_calculator.compute_cvar(misclassified, risk_weights)
        return cvar_loss