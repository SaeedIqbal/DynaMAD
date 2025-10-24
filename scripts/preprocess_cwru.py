"""
Preprocess CWRU Bearing Dataset for Industrial Continual Learning.
- Downloads raw data from official source
- Converts .mat files to .npy time-series
- Generates labels.csv with severity and drift phase
- Saves to /home/phd/datasets/cwru/

"""

import os
import sys
import urllib.request
import zipfile
import scipy.io
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path


class CWRUPreprocessor:
    """Preprocessor for CWRU Bearing Dataset."""

    def __init__(self, root_dir: str = "/home/phd/datasets"):
        self.root_dir = Path(root_dir)
        self.dataset_dir = self.root_dir / "cwru"
        self.raw_dir = self.dataset_dir / "raw"
        self.processed_dir = self.dataset_dir / "vibration"
        self.url = "https://engineering.case.edu/sites/default/files/2023-11/cwru_bearing_data.zip"
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create required directories."""
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

    def download(self) -> None:
        """Download CWRU dataset if not present."""
        zip_path = self.raw_dir / "cwru_bearing_data.zip"
        if not zip_path.exists():
            print(f"📥 Downloading CWRU dataset from {self.url}")
            urllib.request.urlretrieve(self.url, zip_path)
            print("✅ Download complete")
        else:
            print("📁 CWRU zip already exists. Skipping download.")

    def extract(self) -> None:
        """Extract zip file."""
        zip_path = self.raw_dir / "cwru_bearing_data.zip"
        extract_dir = self.raw_dir / "extracted"
        if not extract_dir.exists():
            print("📦 Extracting zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("✅ Extraction complete")
        else:
            print("📁 Already extracted. Skipping.")

    def _load_mat_file(self, mat_path: Path) -> np.ndarray:
        """Load .mat file and extract DE time-series."""
        mat = scipy.io.loadmat(str(mat_path))
        # Find key containing 'DE' (Drive End)
        for key in mat.keys():
            if 'DE' in key and not key.endswith('_'):
                return np.array(mat[key]).flatten()
        raise ValueError(f"No DE signal found in {mat_path}")

    def _get_severity(self, filename: str) -> float:
        """Map fault diameter to severity score."""
        if '007' in filename:
            return 0.6  # 0.007 inch
        elif '014' in filename:
            return 0.75  # 0.014 inch
        elif '021' in filename:
            return 0.9  # 0.021 inch
        else:
            return 0.3  # Normal or unknown

    def _get_label(self, filename: str) -> int:
        """Assign binary label: 0=normal, 1=anomaly."""
        return 0 if 'normal' in filename.lower() else 1

    def _get_drift_phase(self, idx: int, total: int) -> str:
        """Simulate drift phase for continual learning."""
        ratio = idx / total
        if ratio < 0.3:
            return "stable"
        elif ratio < 0.6:
            return "degrading"
        else:
            return "critical"

    def process(self) -> None:
        """Process all .mat files into .npy and generate labels.csv."""
        extract_dir = self.raw_dir / "extracted"
        mat_files = list(extract_dir.rglob("*.mat"))
        
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {extract_dir}")

        records = []
        print(f"⚙️  Processing {len(mat_files)} .mat files...")

        for i, mat_path in enumerate(sorted(mat_files)):
            try:
                # Load and save time-series
                signal = self._load_mat_file(mat_path)
                npy_name = f"sample_{i:05d}.npy"
                npy_path = self.processed_dir / npy_name
                np.save(npy_path, signal.astype(np.float32))

                # Generate metadata
                filename = mat_path.name
                record = {
                    "vibration_path": f"vibration/{npy_name}",
                    "label": self._get_label(filename),
                    "severity": self._get_severity(filename),
                    "drift_phase": self._get_drift_phase(i, len(mat_files)),
                    "original_file": str(mat_path.relative_to(extract_dir))
                }
                records.append(record)

            except Exception as e:
                print(f"⚠️  Skipping {mat_path}: {e}")
                continue

        # Save labels
        df = pd.DataFrame(records)
        df.to_csv(self.dataset_dir / "labels.csv", index=False)
        print(f"✅ Processed {len(df)} samples. Labels saved to {self.dataset_dir}/labels.csv")

    def run(self) -> None:
        """Run full preprocessing pipeline."""
        print("🚀 Starting CWRU preprocessing...")
        self.download()
        self.extract()
        self.process()
        print("🎉 CWRU preprocessing complete!")


def main():
    """Entry point."""
    try:
        preprocessor = CWRUPreprocessor()
        preprocessor.run()
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()