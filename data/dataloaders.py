from torch.utils.data import DataLoader, random_split
from typing import Dict, Any
from .datasets import MVTecVibrationDataset, CWRUDataset, SMAPMSLDataset, RealIADDataset
from .preprocess import IndustrialPreprocessor

DATASET_CLASSES = {
    'mvtec_vibration': MVTecVibrationDataset,
    'cwru': CWRUDataset,
    'smap_msl': SMAPMSLDataset,
    'real_iad': RealIADDataset
}

def get_dataset(name: str, root: str, split: str = 'train', transform=None):
    if name not in DATASET_CLASSES:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_CLASSES[name](root=os.path.join(root, name), split=split, transform=transform)

def get_streaming_loader(
    dataset_name: str,
    data_root: str = '/home/phd/datasets/',
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Returns a DataLoader for continual streaming."""
    preprocessor = IndustrialPreprocessor()
    dataset = get_dataset(dataset_name, data_root, transform=preprocessor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )