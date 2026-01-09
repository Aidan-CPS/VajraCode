"""Utility functions for VajraCode."""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict


def ensure_dir(directory: str) -> None:
    """Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML or JSON file.
    
    Args:
        config: Dictionary with configuration
        config_path: Path to save configuration
    """
    path = Path(config_path)
    ensure_dir(path.parent)
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
