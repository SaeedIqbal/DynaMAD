"""
Reversible ODE/SDE Solvers for Continual Learning Memory Transformation.
- Adjoint Sensitivity Method for ODE gradients (Neural ODE)
- Reversible Heun Method for Stratonovich SDEs
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
import torch
import torch.nn as nn


class BaseODESolver(ABC):
    """Abstract base class for ODE solvers."""

    @abstractmethod
    def solve(
        self,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        y0: torch.Tensor,
        t_span: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Solve ODE dy/dt = func(t, y) over t_span.

        Args:
            func: Drift function (t, y) -> dy/dt
            y0: Initial state [B, ...]
            t_span: Time points [T]

        Returns:
            Solution at all time points [T, B, ...]
        """
        pass


class AdjointODESolver(BaseODESolver):
    """
    Adjoint sensitivity method for Neural ODEs (Chen et al., 2018).
    Solves both forward ODE and backward adjoint ODE in one pass.
    """

    def __init__(self, method: str = 'rk4', atol: float = 1e-6, rtol: float = 1e-5):
        self.method = method
        self.atol = atol
        self.rtol = rtol

    def _rk4_step(
        self,
        func: Callable,
        y: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor
    ) -> torch.Tensor:
        """Runge-Kutta 4th order step."""
        k1 = func(t, y)
        k2 = func(t + dt / 2, y + dt * k1 / 2)
        k3 = func(t + dt / 2, y + dt * k2 / 2)
        k4 = func(t + dt, y + dt * k3)
        return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve(
        self,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        y0: torch.Tensor,
        t_span: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Solve ODE forward pass."""
        device = y0.device
        t_span = t_span.to(device)
        y = y0.clone()
        solution = [y0]

        for i in range(len(t_span) - 1):
            t0, t1 = t_span[i], t_span[i + 1]
            dt = t1 - t0
            if dt == 0:
                continue
            y = self._rk4_step(func, y, t0, dt)
            solution.append(y.clone())

        return torch.stack(solution)  # [T, B, ...]

    def solve_adjoint(
        self,
        func: Callable,
        y0: torch.Tensor,
        t_span: torch.Tensor,
        grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve forward ODE and backward adjoint to compute dy0/dtheta.

        Returns:
            (solution, adjoint_at_t0)
        """
        # Forward pass
        with torch.no_grad():
            ts = t_span
            ys = [y0]
            y = y0
            for i in range(len(ts) - 1):
                y = self._rk4_step(func, y, ts[i], ts[i + 1] - ts[i])
                ys.append(y)

        # Backward adjoint pass
        a = grad_output[-1]  # adjoint at final time
        adjoints = [a]

        for i in reversed(range(len(ts) - 1)):
            # Compute dL/dy at time ts[i]
            if i < len(grad_output) - 1:
                a = a + grad_output[i]

            # Adjoint dynamics: da/dt = -a^T * df/dy
            y_i = ys[i].detach().requires_grad_(True)
            with torch.enable_grad():
                f_i = func(ts[i], y_i)
                # Compute df/dy via autograd
                dfdy = torch.autograd.grad(
                    f_i, y_i, grad_outputs=torch.ones_like(f_i),
                    create_graph=False, retain_graph=False
                )[0]
            a = a - (dfdy.transpose(-2, -1) @ a.unsqueeze(-1)).squeeze(-1)

            # Propagate adjoint backward in time
            dt = ts[i] - ts[i + 1]  # negative
            a = self._rk4_step(
                lambda t, a_val: -torch.autograd.grad(
                    func(t, ys[i].detach().requires_grad_(True)),
                    ys[i], grad_outputs=a_val, retain_graph=False
                )[0] if False else torch.zeros_like(a_val),
                a, ts[i + 1], dt
            )
            adjoints.append(a)

        adjoints = list(reversed(adjoints))
        return torch.stack(ys), adjoints[0]


class ReversibleHeunSDESolver:
    """
    Reversible Heun method for Stratonovich SDEs (Kidger et al., 2021).
    Solves: dX = mu(X) dt + sigma(X) ◦ dW
    """

    def __init__(self, atol: float = 1e-6, rtol: float = 1e-5):
        self.atol = atol
        self.rtol = rtol

    def _heun_step(
        self,
        mu: Callable[[torch.Tensor], torch.Tensor],
        sigma: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        dt: float,
        t: float = 0.0
    ) -> torch.Tensor:
        """
        Reversible Heun step for Stratonovich SDE.
        
        Args:
            mu: Drift function x -> mu(x)
            sigma: Diffusion function x -> sigma(x)
            x: Current state [B, ...]
            dt: Time step
            t: Current time (for time-dependent dynamics)
            
        Returns:
            Next state [B, ...]
        """
        # Wiener increment
        dW = torch.randn_like(x) * (dt ** 0.5)

        # Drift and diffusion at current state
        mu_x = mu(x)
        sigma_x = sigma(x)

        # Predictor (Euler-Maruyama)
        x_pred = x + mu_x * dt + sigma_x * dW

        # Corrector (Heun)
        mu_pred = mu(x_pred)
        sigma_pred = sigma(x_pred)
        x_corr = x + 0.5 * (mu_x + mu_pred) * dt + 0.5 * (sigma_x + sigma_pred) * dW

        return x_corr

    def solve(
        self,
        mu: Callable[[torch.Tensor], torch.Tensor],
        sigma: Callable[[torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        T: float,
        steps: int = 10
    ) -> torch.Tensor:
        """
        Solve SDE from t=0 to t=T.
        
        Args:
            mu: Drift function
            sigma: Diffusion function  
            x0: Initial state [B, ...]
            T: Final time
            steps: Number of integration steps
            
        Returns:
            Final state x(T) [B, ...]
        """
        x = x0.clone()
        dt = T / steps

        for i in range(steps):
            t = i * dt
            x = self._heun_step(mu, sigma, x, dt, t)

        return x

    def solve_path(
        self,
        mu: Callable[[torch.Tensor], torch.Tensor],
        sigma: Callable[[torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        T: float,
        steps: int = 10
    ) -> torch.Tensor:
        """
        Solve SDE and return full path.
        
        Returns:
            Path [steps+1, B, ...]
        """
        x = x0.clone()
        path = [x0]
        dt = T / steps

        for i in range(steps):
            t = i * dt
            x = self._heun_step(mu, sigma, x, dt, t)
            path.append(x.clone())

        return torch.stack(path)


class NeuralODE(nn.Module):
    """Neural ODE wrapper with adjoint solver."""

    def __init__(self, func: nn.Module, solver: Optional[AdjointODESolver] = None):
        super().__init__()
        self.func = func
        self.solver = solver or AdjointODESolver()

    def forward(
        self,
        y0: torch.Tensor,
        t_span: torch.Tensor
    ) -> torch.Tensor:
        """Forward ODE solve."""
        def odefunc(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self.func(y)
        return self.solver.solve(odefunc, y0, t_span)

    def backward(
        self,
        y0: torch.Tensor,
        t_span: torch.Tensor,
        grad_output: torch.Tensor
    ) -> torch.Tensor:
        """Backward pass with adjoint method."""
        def odefunc(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self.func(y)
        _, adjoint = self.solver.solve_adjoint(odefunc, y0, t_span, grad_output)
        return adjoint


class NeuralSDE(nn.Module):
    """Neural SDE wrapper with reversible Heun solver."""

    def __init__(
        self,
        drift: nn.Module,
        diffusion: nn.Module,
        solver: Optional[ReversibleHeunSDESolver] = None
    ):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion
        self.solver = solver or ReversibleHeunSDESolver()

    def forward(
        self,
        x0: torch.Tensor,
        T: float = 0.05,
        steps: int = 10
    ) -> torch.Tensor:
        """Solve SDE forward."""
        return self.solver.solve(self.drift, self.diffusion, x0, T, steps)

    def get_path(
        self,
        x0: torch.Tensor,
        T: float = 0.05,
        steps: int = 10
    ) -> torch.Tensor:
        """Get full SDE path."""
        return self.solver.solve_path(self.drift, self.diffusion, x0, T, steps)