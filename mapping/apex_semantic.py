import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SemanticPrior:
    target: str
    related: List[str]
    room: str
    fusion_threshold: float


class ApexSemanticHelper:
    """
    Lightweight ApexNav-style semantic helper for OneMap.
    """

    def __init__(self, prior_path: str):
        self.prior_path = Path(prior_path)
        self._priors: Dict[str, SemanticPrior] = {}
        self._load()

    def _load(self) -> None:
        if not self.prior_path.exists():
            self._priors = {}
            return

        with open(self.prior_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        priors = {}
        for k, v in raw.items():
            priors[k] = SemanticPrior(
                target=k,
                related=list(v.get("related", [])),
                room=str(v.get("room", "everywhere")),
                fusion_threshold=float(v.get("fusion_threshold", 0.55)),
            )
        self._priors = priors

    def get_prior(self, target: str) -> SemanticPrior:
        target = target.strip().lower()
        if target in self._priors:
            return self._priors[target]
        return SemanticPrior(
            target=target,
            related=[],
            room="everywhere",
            fusion_threshold=0.55,
        )

    def expand_query_classes(self, target: str, topk: int = 6) -> List[str]:
        prior = self.get_prior(target)
        classes = [prior.target] + prior.related[:topk]
        dedup = []
        seen = set()
        for c in classes:
            x = c.strip()
            if x and x not in seen:
                dedup.append(x)
                seen.add(x)
        return dedup

    @staticmethod
    def local_semantic_value(score_map: np.ndarray, pt: np.ndarray, win: int = 2) -> float:
        x, y = int(pt[0]), int(pt[1])
        h, w = score_map.shape
        x0 = max(0, x - win)
        x1 = min(h, x + win + 1)
        y0 = max(0, y - win)
        y1 = min(w, y + win + 1)
        patch = score_map[x0:x1, y0:y1]
        if patch.size == 0:
            return 0.0
        return float(np.max(patch))

    @staticmethod
    def frontier_semantic_stats(score_map: np.ndarray, frontier_points: List[np.ndarray]) -> Tuple[float, float, float, List[float]]:
        """
        Returns:
            std_dev, peak_to_mean, mean_val, per_frontier_values
        """
        values = []
        for pt in frontier_points:
            values.append(ApexSemanticHelper.local_semantic_value(score_map, pt, win=2))

        if len(values) == 0:
            return 0.0, 0.0, 0.0, []

        arr = np.asarray(values, dtype=np.float32)
        mean_val = float(np.mean(arr))
        std_dev = float(np.std(arr))
        peak_to_mean = float(np.max(arr) / (mean_val + 1e-6))
        return std_dev, peak_to_mean, mean_val, values

    @staticmethod
    def choose_mode_from_frontiers(
        score_map: np.ndarray,
        frontier_points: List[np.ndarray],
        std_thresh: float,
        peak_ratio_thresh: float,
    ) -> str:
        std_dev, peak_to_mean, _, _ = ApexSemanticHelper.frontier_semantic_stats(
            score_map, frontier_points
        )
        if std_dev >= std_thresh and peak_to_mean >= peak_ratio_thresh:
            return "semantic"
        return "geometry"