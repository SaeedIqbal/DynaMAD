"""
Reservoir Sampling Memory Buffer for Industrial Continual Learning.
Supports multi-modal data, severity-aware retention, and efficient updates.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import random
import torch
import math


class BaseMemoryBuffer(ABC):
    """Abstract base class for memory buffers."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.size = 0

    @abstractmethod
    def add(self, item: Dict[str, Any]) -> None:
        """Add an item to the buffer."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch from the buffer."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return current buffer size."""
        pass

    @abstractmethod
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        pass


class ReservoirBuffer(BaseMemoryBuffer):
    """
    Reservoir sampling memory buffer with support for multi-modal data.
    Implements Algorithm R (Vitter, 1985) for uniform random sampling.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.buffer: List[Dict[str, Any]] = []

    def add(self, item: Dict[str, Any]) -> None:
        """
        Add an item using reservoir sampling.
        
        Args:
            item (Dict[str, Any]): Multi-modal data dict with keys like 'image', 'vibration', 'label', 'severity'
        """
        if self.size < self.capacity:
            # Buffer not full: add directly
            self.buffer.append(item)
        else:
            # Buffer full: replace with probability capacity / (size + 1)
            j = random.randint(0, self.size)
            if j < self.capacity:
                self.buffer[j] = item
        self.size += 1

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch uniformly at random.
        
        Args:
            batch_size (int): Number of samples to draw
            
        Returns:
            Dict[str, torch.Tensor]: Batched multi-modal data
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")
            
        batch_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = {}
        
        # Get all keys from first item
        keys = self.buffer[0].keys()
        
        for key in keys:
            values = [self.buffer[i][key] for i in indices]
            
            # Convert to tensor if possible
            if isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            elif isinstance(values[0], torch.Tensor):
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    # Handle variable-length sequences (e.g., logs)
                    batch[key] = values  # Keep as list
            else:
                batch[key] = values  # Keep as list for non-tensor data
                
        return batch

    def __len__(self) -> int:
        return len(self.buffer)

    def is_full(self) -> bool:
        return len(self.buffer) >= self.capacity

    def get_severity_stats(self) -> Dict[str, float]:
        """Get statistics about severity distribution in buffer."""
        if len(self.buffer) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            
        severities = [item.get('severity', 0.0) for item in self.buffer]
        severities = [s for s in severities if s is not None]
        
        if not severities:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            
        mean_sev = sum(severities) / len(severities)
        std_sev = math.sqrt(sum((s - mean_sev) ** 2 for s in severities) / len(severities))
        
        return {
            "mean": mean_sev,
            "std": std_sev,
            "min": min(severities),
            "max": max(severities)
        }


class SeverityAwareReservoirBuffer(ReservoirBuffer):
    """
    Reservoir buffer that prioritizes high-severity anomalies during sampling.
    Uses weighted reservoir sampling (Efraimidis & Spirakis, 2006).
    """

    def __init__(self, capacity: int, severity_weight: float = 1.0):
        super().__init__(capacity)
        self.severity_weight = severity_weight
        self.weights: List[float] = []

    def add(self, item: Dict[str, Any]) -> None:
        """Add item with severity-aware weighting."""
        severity = item.get('severity', 0.0)
        weight = (severity ** self.severity_weight) if severity > 0 else 1e-8
        
        if self.size < self.capacity:
            self.buffer.append(item)
            self.weights.append(weight)
        else:
            # Weighted reservoir sampling
            # Generate key: k = u^(1/w) where u ~ Uniform(0,1)
            u = random.random()
            key = u ** (1.0 / weight)
            
            # Find minimum key in current buffer
            min_key = float('inf')
            min_idx = -1
            for i, w in enumerate(self.weights):
                current_key = random.random() ** (1.0 / w)  # Approximate
                if current_key < min_key:
                    min_key = current_key
                    min_idx = i
                    
            # Replace if new key is larger
            if key > min_key and min_idx >= 0:
                self.buffer[min_idx] = item
                self.weights[min_idx] = weight
                
        self.size += 1

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample with probability proportional to severity."""
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")
            
        batch_size = min(batch_size, len(self.buffer))
        
        # Get weights
        weights = [item.get('severity', 0.0) for item in self.buffer]
        weights = [max(w, 1e-8) for w in weights]  # Avoid zero weights
        
        # Sample with replacement using weights
        indices = random.choices(range(len(self.buffer)), weights=weights, k=batch_size)
        batch = {}
        
        keys = self.buffer[0].keys()
        for key in keys:
            values = [self.buffer[i][key] for i in indices]
            
            if isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            elif isinstance(values[0], torch.Tensor):
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    batch[key] = values
            else:
                batch[key] = values
                
        return batch


class MultiModalReservoirBuffer(ReservoirBuffer):
    """
    Specialized buffer for multi-modal industrial data with validation.
    """

    def __init__(self, capacity: int, required_modalities: Optional[List[str]] = None):
        super().__init__(capacity)
        self.required_modalities = required_modalities or []

    def add(self, item: Dict[str, Any]) -> None:
        """Add item with modality validation."""
        # Validate required modalities
        if self.required_modalities:
            missing = set(self.required_modalities) - set(item.keys())
            if missing:
                raise ValueError(f"Missing required modalities: {missing}")
                
        # Validate tensor shapes
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    raise ValueError(f"Tensor {key} must have at least 1 dimension")
                    
        super().add(item)

    def get_modality_stats(self) -> Dict[str, int]:
        """Get count of samples per modality."""
        if len(self.buffer) == 0:
            return {}
            
        stats = {}
        for item in self.buffer:
            for key in item.keys():
                if key not in ['label', 'severity']:
                    stats[key] = stats.get(key, 0) + 1
        return stats