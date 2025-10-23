from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import torch

class BaseDataset(ABC):
    """Abstract base class for all industrial anomaly detection datasets."""

    def __init__(self, root: str, split: str = 'train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = []
        self._load_metadata()

    @abstractmethod
    def _load_metadata(self) -> None:
        """Populate self.samples with dicts containing paths and labels."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a sample dict with modality keys and label."""
        pass

    def __len__(self) -> int:
        return len(self.samples)