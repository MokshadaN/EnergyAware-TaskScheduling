"""Helper utilities for random seed configuration and quick plotting."""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch

from utils.visualizer import comprehensive_plotting, plot_baselines


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def plot_comprehensive(history: Dict[str, Any], eval_results: Dict[str, Any]) -> None:
    """Delegates to comprehensive_plotting in utils.visualizer."""
    comprehensive_plotting(history, eval_results)


def plot_baseline_summary(results: Dict[str, Any]) -> None:
    """Delegates to plot_baselines in utils.visualizer."""
    plot_baselines(results)
