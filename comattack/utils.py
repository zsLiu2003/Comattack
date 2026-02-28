"""
Utility Functions
=================

Common utilities used across all modules.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import yaml


# =============================================================================
# Path Management
# =============================================================================

def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (relative to project root)
        
    Returns:
        Configuration dictionary
    """
    root = get_project_root()
    config_file = root / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Add computed paths
    config["_root"] = str(root)
    config["_data_path"] = str(root / config["paths"]["data_dir"] / config["paths"]["data_file"])
    config["_results_dir"] = str(root / config["paths"]["results_dir"])
    
    return config


# =============================================================================
# Logging
# =============================================================================

def setup_logging(debug: bool = False) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        debug: Enable debug level logging
        
    Returns:
        Logger instance
    """
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    return logging.getLogger("COMA")


def print_banner():
    """Print project banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗██╗   ██╗███████╗████████╗███████╗███╗   ███╗       ║
║   ██╔════╝╚██╗ ██╔╝██╔════╝╚══██╔══╝██╔════╝████╗ ████║       ║
║   ███████╗ ╚████╔╝ ███████╗   ██║   █████╗  ██╔████╔██║       ║
║   ╚════██║  ╚██╔╝  ╚════██║   ██║   ██╔══╝  ██║╚██╔╝██║       ║
║   ███████║   ██║   ███████║   ██║   ███████╗██║ ╚═╝ ██║       ║
║   ╚══════╝   ╚═╝   ╚══════╝   ╚═╝   ╚══════╝╚═╝     ╚═╝       ║
║                                                               ║
║          Guardrail Compression Analysis for ICML              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


# =============================================================================
# File I/O
# =============================================================================

def save_json(data: Any, path: Path, indent: int = 2):
    """Save data to JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results: Dict, phase: str, name: str, config: Dict):
    """
    Save experiment results to results directory.
    
    Args:
        results: Results dictionary
        phase: Phase name (phase1, phase2, etc.)
        name: Result name (e.g., "ppl_analysis")
        config: Configuration dictionary
    """
    results_dir = Path(config["_results_dir"]) / phase
    ensure_dir(results_dir)
    
    # Add metadata
    results["_metadata"] = {
        "phase": phase,
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "config_version": config["project"]["version"]
    }
    
    # Save JSON
    json_path = results_dir / f"{name}.json"
    save_json(results, json_path)
    
    logging.info(f"Saved results: {json_path}")
    return json_path


# =============================================================================
# Device Management
# =============================================================================

def get_device(device: str = "cuda") -> str:
    """
    Get compute device.
    
    Args:
        device: Requested device ("cuda" or "cpu")
        
    Returns:
        Available device string
    """
    import torch
    
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"Using GPU: {gpu_name}")
        return "cuda"
    else:
        logging.info("Using CPU")
        return "cpu"


# =============================================================================
# Progress Utilities
# =============================================================================

class ProgressTracker:
    """Simple progress tracker for experiments."""
    
    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        self.current += n
        self._print_progress()
    
    def _print_progress(self):
        pct = self.current / self.total * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        eta = elapsed / self.current * (self.total - self.current) if self.current > 0 else 0
        
        print(f"\r{self.desc}: {self.current}/{self.total} ({pct:.1f}%) | ETA: {eta:.0f}s", end="")
        
        if self.current >= self.total:
            print()  # New line at end




