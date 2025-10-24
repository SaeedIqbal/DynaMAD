"""
FMEA-Based Severity Mapping for Industrial Anomaly Detection.
Maps anomaly types or defect characteristics to operational severity scores (0.0–1.0).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import json
import os


class BaseSeverityMapper(ABC):
    """Abstract base class for severity mappers."""

    def __init__(self, default_severity: float = 0.5):
        if not (0.0 <= default_severity <= 1.0):
            raise ValueError("Default severity must be in [0.0, 1.0]")
        self.default_severity = default_severity

    @abstractmethod
    def map_severity(self, anomaly_info: Dict[str, Any]) -> float:
        """
        Map anomaly information to a severity score.

        Args:
            anomaly_info (Dict[str, Any]): Contains keys like 'type', 'size', 'location', etc.

        Returns:
            float: Severity score in [0.0, 1.0]
        """
        pass


class FMEABasedSeverityMapper(BaseSeverityMapper):
    """
    Severity mapper based on Failure Modes and Effects Analysis (FMEA).
    Uses domain-specific lookup tables to assign severity scores.
    """

    def __init__(
        self,
        fmea_config_path: Optional[str] = None,
        default_severity: float = 0.5
    ):
        super().__init__(default_severity=default_severity)
        self.fmea_rules = self._load_fmea_rules(fmea_config_path)

    def _load_fmea_rules(self, config_path: Optional[str]) -> Dict[str, float]:
        """Load FMEA severity rules from JSON or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default FMEA rules for common industrial domains
        return {
            # Semiconductor fab
            "chamber_wall_erosion": 0.95,
            "endpoint_detection_failure": 0.90,
            "particle_contamination": 0.70,
            "micro_crack": 0.85,
            "misalignment": 0.60,
            
            # Mechanical systems (CWRU)
            "inner_race_defect": 0.95,
            "ball_fault": 0.85,
            "outer_race_defect": 0.80,
            "cage_fault": 0.70,
            "lubricant_degradation": 0.60,
            
            # Aerospace telemetry (SMAP/MSL)
            "thruster_malfunction": 0.98,
            "power_surge": 0.92,
            "sensor_glitch": 0.65,
            "communication_loss": 0.88,
            
            # Metal surface defects (MPDD)
            "crack": 0.95,
            "scratch": 0.70,
            "dent": 0.80,
            "corrosion": 0.85,
            "stain": 0.40,
            
            # Graph anomalies (DGAD)
            "high_centrality_outlier": 0.90,
            "structural_hole": 0.75,
            "feature_deviation": 0.60
        }

    def map_severity(self, anomaly_info: Dict[str, Any]) -> float:
        """
        Map anomaly to severity using FMEA rules.

        Args:
            anomaly_info (Dict): Must contain 'type' key. May contain 'size', 'location', etc.

        Returns:
            float: Severity score
        """
        if not isinstance(anomaly_info, dict):
            raise TypeError("anomaly_info must be a dictionary")

        # Primary key: anomaly type
        anomaly_type = anomaly_info.get('type')
        if anomaly_type:
            # Normalize key (lowercase, underscore)
            key = str(anomaly_type).strip().lower().replace(' ', '_')
            if key in self.fmea_rules:
                return float(self.fmea_rules[key])

        # Fallback: size-based severity for defects
        size = anomaly_info.get('size')
        if size is not None:
            try:
                size = float(size)
                # Linear mapping: size in [0, 100] -> severity in [0.3, 0.95]
                severity = min(0.95, max(0.3, 0.3 + 0.65 * (size / 100.0)))
                return float(severity)
            except (ValueError, TypeError):
                pass

        # Final fallback
        return self.default_severity


class RuleBasedSeverityMapper(BaseSeverityMapper):
    """
    Flexible rule-based mapper using custom severity functions.
    """

    def __init__(
        self,
        rules: Dict[str, Union[float, callable]],
        default_severity: float = 0.5
    ):
        super().__init__(default_severity=default_severity)
        self.rules = rules

    def map_severity(self, anomaly_info: Dict[str, Any]) -> float:
        """Apply custom rules or functions."""
        for key, rule in self.rules.items():
            if key in anomaly_info:
                value = anomaly_info[key]
                if callable(rule):
                    return float(rule(value))
                else:
                    return float(rule)
        return self.default_severity


class SeverityMapperFactory:
    """
    Factory class to instantiate severity mappers based on dataset.
    """

    @staticmethod
    def create_mapper(
        dataset_name: str,
        custom_config_path: Optional[str] = None
    ) -> BaseSeverityMapper:
        """
        Create a severity mapper appropriate for the dataset.

        Args:
            dataset_name (str): One of 'mvtec_vibration', 'cwru', 'smap_msl', 'real_iad', 'dgad', 'mpdd'
            custom_config_path (str, optional): Path to custom FMEA JSON

        Returns:
            BaseSeverityMapper: Configured severity mapper
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name in ['mvtec_vibration', 'real_iad']:
            return FMEABasedSeverityMapper(
                fmea_config_path=custom_config_path,
                default_severity=0.5
            )
        elif dataset_name == 'cwru':
            return FMEABasedSeverityMapper(
                fmea_config_path=custom_config_path,
                default_severity=0.6
            )
        elif dataset_name == 'smap_msl':
            return FMEABasedSeverityMapper(
                fmea_config_path=custom_config_path,
                default_severity=0.7
            )
        elif dataset_name in ['dgad', 'mpdd']:
            return FMEABasedSeverityMapper(
                fmea_config_path=custom_config_path,
                default_severity=0.55
            )
        else:
            # Generic mapper
            return FMEABasedSeverityMapper(
                fmea_config_path=custom_config_path,
                default_severity=0.5
            )


# Global severity mapper instance (singleton pattern)
class GlobalSeverityMapper:
    """Singleton wrapper for global severity mapping."""

    _instance = None
    _mapper = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalSeverityMapper, cls).__new__(cls)
        return cls._instance

    def initialize(self, dataset_name: str, config_path: Optional[str] = None):
        """Initialize the global mapper."""
        self._mapper = SeverityMapperFactory.create_mapper(dataset_name, config_path)

    def map(self, anomaly_info: Dict[str, Any]) -> float:
        """Map anomaly to severity."""
        if self._mapper is None:
            raise RuntimeError("GlobalSeverityMapper not initialized. Call initialize() first.")
        return self._mapper.map_severity(anomaly_info)


# Convenience function
def get_severity_mapper(
    dataset_name: str,
    config_path: Optional[str] = None
) -> BaseSeverityMapper:
    """Convenience function to get a severity mapper."""
    return SeverityMapperFactory.create_mapper(dataset_name, config_path)