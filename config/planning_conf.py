# from spock import spock

# @spock
# class PlanningConf:
#     percentile_exploitation: float
#     frontier_depth: float
#     no_nav_radius: float
#     yolo_confidence: float
#     filter_detections_depth: bool
#     consensus_filtering: bool
#     allow_replan: bool
#     use_frontiers: bool
#     allow_far_plan: bool
#     using_ov: bool
#     max_detect_distance: float
#     obstcl_kernel_size: float
#     min_goal_dist: float
from spock import spock


@spock
class PlanningConf:
    percentile_exploitation: float
    frontier_depth: float
    no_nav_radius: float
    yolo_confidence: float
    filter_detections_depth: bool
    consensus_filtering: bool
    allow_replan: bool
    use_frontiers: bool
    allow_far_plan: bool
    using_ov: bool
    max_detect_distance: float
    obstcl_kernel_size: float
    min_goal_dist: float

    # ---------- ApexNav fusion ----------
    use_apex_fusion: bool = True
    related_classes_topk: int = 6
    semantic_std_thresh: float = 0.08
    semantic_peak_ratio_thresh: float = 1.35
    semantic_frontier_weight: float = 1.25
    geometry_frontier_weight: float = 0.75
    semantic_cluster_bonus: float = 1.15
    related_object_weight: float = 0.35
    room_prior_weight: float = 0.15
    semantic_prior_path: str = "config/semantic_priors.json"