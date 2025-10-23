import torch
import torch.nn.functional as F
from typing import Dict, Any

class IndustrialPreprocessor:
    """
    Unified preprocessor for multi-modal industrial data.
    Applies modality-specific normalization and padding.
    """

    def __init__(
        self,
        img_size: int = 256,
        vib_length: int = 10000,
        ts_length: int = 100,
        log_length: int = 20,
        vib_mean: float = 0.0,
        vib_std: float = 1.0,
        ts_mean: float = 0.0,
        ts_std: float = 1.0
    ):
        self.img_size = img_size
        self.vib_length = vib_length
        self.ts_length = ts_length
        self.log_length = log_length
        self.vib_mean = vib_mean
        self.vib_std = vib_std
        self.ts_mean = ts_mean
        self.ts_std = ts_std

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        processed = {}
        # Image: center crop + normalize
        if 'image' in sample:
            img = sample['image']
            img = self._center_crop(img, self.img_size)
            img = (img - 0.5) / 0.5  # [-1, 1]
            processed['image'] = img

        # Vibration: pad/truncate + normalize
        if 'vibration' in sample:
            vib = sample['vibration']
            vib = self._pad_or_truncate(vib, self.vib_length, dim=-1)
            vib = (vib - self.vib_mean) / self.vib_std
            processed['vibration'] = vib

        # Telemetry/Sensor: pad/truncate + normalize
        if 'telemetry' in sample or 'sensor' in sample:
            key = 'telemetry' if 'telemetry' in sample else 'sensor'
            ts = sample[key]
            ts = self._pad_or_truncate(ts, self.ts_length, dim=-1)
            ts = (ts - self.ts_mean) / self.ts_std
            processed[key] = ts

        # Log: pad/truncate
        if 'log' in sample:
            log = sample['log']
            log = self._pad_or_truncate(log, self.log_length, dim=-1, pad_value=0)
            processed['log'] = log

        # Pass through labels
        processed['label'] = sample['label']
        processed['severity'] = sample['severity']
        return processed

    def _center_crop(self, img: torch.Tensor, size: int) -> torch.Tensor:
        _, h, w = img.shape
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        return img[:, start_h:start_h+size, start_w:start_w+size]

    def _pad_or_truncate(
        self,
        x: torch.Tensor,
        target_len: int,
        dim: int,
        pad_value: float = 0.0
    ) -> torch.Tensor:
        if x.shape[dim] > target_len:
            idx = torch.randperm(x.shape[dim])[:target_len]
            return x.index_select(dim, idx.sort().values)
        elif x.shape[dim] < target_len:
            pad_shape = list(x.shape)
            pad_shape[dim] = target_len - x.shape[dim]
            padding = torch.full(pad_shape, pad_value, dtype=x.dtype)
            return torch.cat([x, padding], dim=dim)
        return x