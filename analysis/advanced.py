import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple

class ParetoAnalyzer:
    @staticmethod
    def get_pareto_front(points: np.ndarray, maximize: List[bool] = None) -> np.ndarray:
        if maximize is None:
            maximize = [False] * points.shape[1]
        
        n = points.shape[0]
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and is_pareto[i]:
                    dominates = True
                    for k in range(points.shape[1]):
                        if maximize[k]:
                            if points[j, k] < points[i, k]: dominates = False; break
                        else:
                            if points[j, k] > points[i, k]: dominates = False; break
                    if dominates:
                        is_pareto[i] = False
                        break
        return is_pareto

    @staticmethod
    def analyze_power_vs_utilization(energies: List[float], utils: List[float]) -> Dict:
        pts = np.column_stack([energies, utils])
        idx = ParetoAnalyzer.get_pareto_front(pts, maximize=[False, True])
        return {'pareto_points': pts[idx], 'ratio': len(pts[idx]) / len(pts)}

class RewardAnalyzer:
    @staticmethod
    def analyze_reward_trends(history: List[Tuple[float, float]], window=11) -> Dict:
        r1 = [h[0] for h in history]
        r2 = [h[1] for h in history]
        
        if len(r1) >= window:
            s1 = savgol_filter(r1, window, 2)
            s2 = savgol_filter(r2, window, 2)
        else:
            s1, s2 = r1, r2
            
        return {'r1_smoothed': s1, 'r2_smoothed': s2}

class MetricsAnalyzer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def get_reliability_stats(self, n_tasks: int) -> Dict:
        # Simplified simulation of reliability stats
        mtbf = 1000 + 50 * n_tasks
        fit = 1000 * np.exp(-n_tasks / 500)
        return {'mtbf': mtbf, 'fit': fit}
