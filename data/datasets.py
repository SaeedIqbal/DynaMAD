import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any
from .dataset_base import BaseDataset

class MVTecVibrationDataset(BaseDataset):
    def _load_metadata(self):
        df = pd.read_csv(os.path.join(self.root, 'labels.csv'))
        self.samples = df.to_dict('records')

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        # Load image
        img = Image.open(os.path.join(self.root, item['image_path'])).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0  # [H, W, C]
        img = torch.from_numpy(img).permute(2, 0, 1)    # [C, H, W]

        # Load vibration
        vib = np.load(os.path.join(self.root, item['vibration_path']))
        vib = torch.from_numpy(vib).float().unsqueeze(0)  # [1, T]

        return {
            'image': img,
            'vibration': vib,
            'label': torch.tensor(item['label'], dtype=torch.long),
            'severity': torch.tensor(item['severity'], dtype=torch.float32)
        }

class CWRUDataset(BaseDataset):
    def _load_metadata(self):
        df = pd.read_csv(os.path.join(self.root, 'labels.csv'))
        self.samples = df.to_dict('records')

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        vib = np.load(os.path.join(self.root, item['vibration_path']))
        vib = torch.from_numpy(vib).float().unsqueeze(0)  # [1, T]
        return {
            'vibration': vib,
            'label': torch.tensor(item['label'], dtype=torch.long),
            'severity': torch.tensor(item['severity'], dtype=torch.float32)
        }

class SMAPMSLDataset(BaseDataset):
    def _load_metadata(self):
        df = pd.read_csv(os.path.join(self.root, 'labels.csv'))
        self.samples = df.to_dict('records')

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        ts = np.load(os.path.join(self.root, item['telemetry_path']))
        ts = torch.from_numpy(ts).float().T  # [C, T]
        return {
            'telemetry': ts,
            'label': torch.tensor(item['label'], dtype=torch.long),
            'severity': torch.tensor(item['severity'], dtype=torch.float32)
        }

class RealIADDataset(BaseDataset):
    def _load_metadata(self):
        df = pd.read_csv(os.path.join(self.root, 'labels.csv'))
        self.samples = df.to_dict('records')

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        # Image
        img = Image.open(os.path.join(self.root, item['image_path'])).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        # Sensor
        sensor = np.load(os.path.join(self.root, item['sensor_path']))
        sensor = torch.from_numpy(sensor).float().T  # [C, T]

        # Log (categorical sequence)
        log = np.load(os.path.join(self.root, item['log_path']))
        log = torch.from_numpy(log).long()  # [L]

        return {
            'image': img,
            'sensor': sensor,
            'log': log,
            'label': torch.tensor(item['label'], dtype=torch.long),
            'severity': torch.tensor(item['severity'], dtype=torch.float32)
        }