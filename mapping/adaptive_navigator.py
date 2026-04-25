from typing import List, Optional, Tuple

import numpy as np

from mapping.navigator import Navigator
from mapping.nav_goals.frontier import Frontier
from mapping.nav_goals.clustering import Cluster
from mapping.apex_semantic import ApexSemanticHelper, SemanticPrior


class AdaptiveNavigator(Navigator):
    """
    OneMap + ApexNav-style adaptive navigator.

    Key ideas ported from ApexNav:
    1) target-centric semantic fusion: target + related classes
    2) adaptive exploration: semantic mode vs geometry mode
    3) frontier reranking using semantic distribution sharpness
    """

    def __init__(self, model, detector, config):
        super().__init__(model, detector, config)

        self.use_apex_fusion = bool(getattr(config.planner, "use_apex_fusion", True))
        self.semantic_helper = ApexSemanticHelper(config.planner.semantic_prior_path)

        self.semantic_mode = "geometry"
        self.current_prior: Optional[SemanticPrior] = None
        self.current_expanded_query: List[str] = ["Other."]
        self.last_frontier_stats: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def reset(self):
        super().reset()
        self.semantic_mode = "geometry"
        self.current_prior = None
        self.current_expanded_query = ["Other."]
        self.last_frontier_stats = (0.0, 0.0, 0.0)

    def set_query(self, txt: List[str]) -> None:
        """
        Override OneMap query setting:
        - base OneMap used only the first query text for map similarity
        - we keep that behavior for planning semantics
        - but we expand detector classes with Apex-style related classes
        """
        if len(txt) == 0:
            txt = ["Other."]

        target = txt[0]
        if not self.use_apex_fusion:
            super().set_query([target])
            return

        self.current_prior = self.semantic_helper.get_prior(target)
        expanded = self.semantic_helper.expand_query_classes(
            target, topk=self.config.planner.related_classes_topk
        )
        self.current_expanded_query = expanded

        # Important:
        # Keep query_text[0] as the main target, so OneMap's similarity map
        # remains target-centric.
        self.query_text = expanded
        self.query_text_features = self.model.get_text_features(
            ["a " + expanded[0]]
        ).to(self.one_map.map_device)

        self.previous_sims = None
        self.one_map.reset_checked_map()
        self.detector.set_classes(expanded)
        self.object_detected = False

        # refresh map-dependent scores
        self.get_map(False)

    def _get_semantic_score_map(self) -> np.ndarray:
        """
        Safe accessor for the similarity map used by OneMap.
        """
        if self.previous_sims is None:
            return np.zeros_like(self.one_map.navigable_map, dtype=np.float32)

        sims = self.previous_sims
        if hasattr(sims, "detach"):
            sims = sims.detach().cpu().numpy()
        elif hasattr(sims, "cpu"):
            sims = sims.cpu().numpy()

        sims = np.asarray(sims, dtype=np.float32)
        if sims.ndim > 2:
            sims = np.squeeze(sims)

        if sims.shape != self.one_map.navigable_map.shape:
            sims = np.resize(sims, self.one_map.navigable_map.shape)

        return sims

    def _choose_semantic_mode(self) -> str:
        score_map = self._get_semantic_score_map()

        frontier_pts = []
        for goal in self.nav_goals:
            if isinstance(goal, Frontier):
                frontier_pts.append(goal.get_descr_point())

        std_dev, peak_ratio, mean_val, _ = self.semantic_helper.frontier_semantic_stats(
            score_map, frontier_pts
        )
        self.last_frontier_stats = (std_dev, peak_ratio, mean_val)

        mode = self.semantic_helper.choose_mode_from_frontiers(
            score_map=score_map,
            frontier_points=frontier_pts,
            std_thresh=self.config.planner.semantic_std_thresh,
            peak_ratio_thresh=self.config.planner.semantic_peak_ratio_thresh,
        )
        return mode

    def _rerank_nav_goals(self, start_xy: np.ndarray) -> None:
        """
        Reweights existing OneMap nav_goals in-place.
        This is the cleanest place to inject ApexNav behavior
        without rewriting OneMap path planning.
        """
        if len(self.nav_goals) == 0:
            return

        score_map = self._get_semantic_score_map()
        self.semantic_mode = self._choose_semantic_mode()

        weighted_goals = []

        for goal in self.nav_goals:
            pt = np.asarray(goal.get_descr_point()).astype(np.int32)
            base_score = float(goal.get_score())
            sem_value = self.semantic_helper.local_semantic_value(score_map, pt, win=2)
            dist = float(np.linalg.norm(start_xy - pt) + 1e-6)

            if self.semantic_mode == "semantic":
                fused_score = (
                    self.config.planner.semantic_frontier_weight * base_score
                    + sem_value
                )

                if isinstance(goal, Cluster):
                    fused_score *= self.config.planner.semantic_cluster_bonus
                    goal.cluster_score = fused_score
                elif isinstance(goal, Frontier):
                    goal.frontier_score = fused_score

                sort_key = -fused_score

            else:
                # geometry mode: closer frontiers first, but don't discard semantics fully
                geom_score = (
                    self.config.planner.geometry_frontier_weight * (1.0 / dist)
                    + 0.15 * sem_value
                )

                if isinstance(goal, Cluster):
                    goal.cluster_score = geom_score
                elif isinstance(goal, Frontier):
                    goal.frontier_score = geom_score

                sort_key = -geom_score

            weighted_goals.append((sort_key, goal))

        weighted_goals.sort(key=lambda x: x[0])
        self.nav_goals = [g for _, g in weighted_goals]

    def compute_frontiers_and_POIs(self, px, py):
        """
        Hook into OneMap's existing frontier/POI generation, then apply Apex-style reranking.
        """
        print(f"[ApexFusion] mode={self.semantic_mode}, stats={self.last_frontier_stats}")
        super().compute_frontiers_and_POIs(px, py)

        if not self.use_apex_fusion:
            return

        start_xy = np.array([px, py], dtype=np.float32)
        self._rerank_nav_goals(start_xy)

    def add_data(self, rgb, depth, transformation_matrix):
        """
        Keep OneMap mapping untouched; only update semantic mode after new evidence arrives.
        """
        obj_found = super().add_data(rgb, depth, transformation_matrix)

        if not self.use_apex_fusion:
            return obj_found

        # If object is already detected, keep OneMap's direct-object behavior.
        if self.object_detected:
            self.semantic_mode = "semantic"
            return obj_found

        # Otherwise let mode be decided from semantic frontier distribution.
        if len(self.nav_goals) > 0:
            try:
                self.semantic_mode = self._choose_semantic_mode()
            except Exception:
                self.semantic_mode = "geometry"
        
        return obj_found
    