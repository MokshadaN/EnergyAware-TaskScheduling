"""System evaluation metrics calculations."""

from __future__ import annotations

import numpy as np

from core.environment import CloudEnvironment


def calculate_atlp(env: CloudEnvironment) -> float:
    """Calculate Average Task Loss Probability (ATLP) based on VM utilization trace."""
    if not env.resources:
        return 0.0
    p_tau = [r.u_j for r in env.resources]
    return float(np.mean(p_tau))
