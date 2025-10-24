"""
Drift-Adaptive Stochastic Dynamics for Intra-Task Concept Drift.
Implements recursive Hellinger estimator and adaptive SDE with time-varying diffusion.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import math


class BaseDriftDetector(ABC):
    """Abstract base class for drift detection."""

    def __init__(self, rho: float = 0.05):
        self.rho = rho  # Exponential moving average momentum

    @abstractmethod
    def update(self, stream_stats: Dict[str, torch.Tensor], buffer_stats: Dict[str, torch.Tensor]) -> float:
        """Update drift estimate and return scalar drift magnitude."""
        pass


class RecursiveHellingerDetector(BaseDriftDetector):
    """
    Recursive Hellinger distance estimator between streaming data and memory buffer.
    Uses Gaussian approximation in embedding space for efficiency.
    """

    def __init__(self, rho: float = 0.05, eps: float = 1e-6):
        super().__init__(rho)
        self.eps = eps
        self.mu_stream = None
        self.sigma_stream = None

    def _update_statistics(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update exponential moving average of mean and covariance."""
        batch_mu = z.mean(dim=0)
        batch_sigma = torch.cov(z.T)

        if self.mu_stream is None:
            self.mu_stream = batch_mu.detach()
            self.sigma_stream = batch_sigma.detach()
        else:
            self.mu_stream = (1 - self.rho) * self.mu_stream + self.rho * batch_mu
            self.sigma_stream = (1 - self.rho) * self.sigma_stream + self.rho * batch_sigma

        return self.mu_stream.clone(), self.sigma_stream.clone()

    def _hellinger_gaussian(
        self,
        mu1: torch.Tensor,
        sigma1: torch.Tensor,
        mu2: torch.Tensor,
        sigma2: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hellinger distance between two multivariate Gaussians."""
        d = mu1.shape[0]
        sigma_avg = 0.5 * (sigma1 + sigma2)

        # Regularize to ensure invertibility
        sigma1 = sigma1 + self.eps * torch.eye(d, device=sigma1.device)
        sigma2 = sigma2 + self.eps * torch.eye(d, device=sigma2.device)
        sigma_avg = sigma_avg + self.eps * torch.eye(d, device=sigma_avg.device)

        # Log-determinant term
        logdet1 = torch.logdet(sigma1)
        logdet2 = torch.logdet(sigma2)
        logdet_avg = torch.logdet(sigma_avg)
        term1 = 0.25 * (logdet_avg - 0.5 * (logdet1 + logdet2))

        # Mahalanobis term
        diff = mu1 - mu2
        try:
            sigma_avg_inv = torch.inverse(sigma_avg)
        except:
            sigma_avg_inv = torch.pinverse(sigma_avg)
        term2 = 0.125 * torch.dot(diff, sigma_avg_inv @ diff)

        hellinger_sq = 1.0 - torch.exp(-term1 - term2)
        return torch.sqrt(torch.clamp(hellinger_sq, min=0.0, max=1.0))

    def update(
        self,
        stream_stats: Dict[str, torch.Tensor],
        buffer_stats: Dict[str, torch.Tensor]
    ) -> float:
        """
        Update drift estimate using streaming and buffer embeddings.

        Args:
            stream_stats (Dict): {'z': [B, D]}
            buffer_stats (Dict): {'z': [M, D]}

        Returns:
            float: Hellinger distance (scalar)
        """
        z_stream = stream_stats['z'].detach()
        z_buffer = buffer_stats['z'].detach()

        # Update streaming statistics
        mu_s, sigma_s = self._update_statistics(z_stream)

        # Buffer statistics (no EMA)
        mu_b = z_buffer.mean(dim=0)
        sigma_b = torch.cov(z_buffer.T)

        # Compute Hellinger distance
        h_dist = self._hellinger_gaussian(mu_s, sigma_s, mu_b, sigma_b)
        return h_dist.item()


class AdaptiveSDE(nn.Module):
    """
    Drift-adaptive stochastic differential equation with time-varying diffusion.
    Solves: dX = mu(X) dt + sigma(t) ◦ dW, where sigma(t) = sigma0 * (1 + kappa * H_t)
    """

    def __init__(
        self,
        drift_net: nn.Module,
        base_diffusion: float = 0.25,
        drift_gain: float = 1.8,
        detector: Optional[BaseDriftDetector] = None
    ):
        super().__init__()
        self.drift_net = drift_net
        self.base_diffusion = base_diffusion
        self.drift_gain = drift_gain
        self.detector = detector or RecursiveHellingerDetector()

    def _heun_step(
        self,
        x: torch.Tensor,
        dt: float,
        sigma_t: float
    ) -> torch.Tensor:
        """Reversible Heun step for Stratonovich SDE."""
        mu = self.drift_net(x)
        dW = torch.randn_like(x) * math.sqrt(dt)
        # Predictor
        x_pred = x + mu * dt + sigma_t * dW
        # Corrector
        mu_pred = self.drift_net(x_pred)
        x_corr = x + 0.5 * (mu + mu_pred) * dt + sigma_t * dW
        return x_corr

    def forward(
        self,
        x: torch.Tensor,
        stream_stats: Dict[str, torch.Tensor],
        buffer_stats: Dict[str, torch.Tensor],
        T: float = 0.03,
        steps: int = 10
    ) -> torch.Tensor:
        """
        Solve adaptive SDE with drift-modulated diffusion.

        Args:
            x (torch.Tensor): Input data [B, ...]
            stream_stats (Dict): Streaming embeddings {'z': [B, D]}
            buffer_stats (Dict): Buffer embeddings {'z': [M, D]}
            T (float): Time horizon
            steps (int): Integration steps

        Returns:
            torch.Tensor: Transformed data
        """
        # Estimate drift magnitude
        hellinger = self.detector.update(stream_stats, buffer_stats)
        sigma_t = self.base_diffusion * (1.0 + self.drift_gain * hellinger)

        # Solve SDE
        x_t = x.clone()
        dt = T / steps
        for _ in range(steps):
            x_t = self._heun_step(x_t, dt, sigma_t)
        return x_t


class DriftAdaptiveSDEPolicy:
    """
    Complete policy for drift-adaptive memory transformation.
    """

    def __init__(
        self,
        drift_net: nn.Module,
        base_diffusion: float = 0.25,
        drift_gain: float = 1.8,
        rho: float = 0.05,
        T: float = 0.03,
        steps: int = 10
    ):
        self.T = T
        self.steps = steps
        self.adaptive_sde = AdaptiveSDE(
            drift_net=drift_net,
            base_diffusion=base_diffusion,
            drift_gain=drift_gain,
            detector=RecursiveHellingerDetector(rho=rho)
        )

    def __call__(
        self,
        x: torch.Tensor,
        stream_embeddings: torch.Tensor,
        buffer_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform memory data with drift-adaptive SDE.

        Args:
            x (torch.Tensor): Memory data [B, ...]
            stream_embeddings (torch.Tensor): Current batch embeddings [B, D]
            buffer_embeddings (torch.Tensor): Memory buffer embeddings [M, D]

        Returns:
            torch.Tensor: Transformed memory data
        """
        stream_stats = {'z': stream_embeddings}
        buffer_stats = {'z': buffer_embeddings}
        return self.adaptive_sde(x, stream_stats, buffer_stats, T=self.T, steps=self.steps)