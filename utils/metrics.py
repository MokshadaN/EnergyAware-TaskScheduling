import numpy as np
from core.environment import CloudEnvironment

def calculate_atlp(env: CloudEnvironment) -> float:
    if not env.resources:
        return 0.0
    P_tau = [r.u_j for r in env.resources]
    return np.mean(P_tau)
