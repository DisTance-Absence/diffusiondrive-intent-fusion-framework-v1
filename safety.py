"""Current-environment geometry adapter for saved, ego-frame trajectories.

This is a sampled constant-velocity geometry filter, not a safety certificate,
TTC calculation, ROS/Nav2 costmap, or closed-loop test.  It deliberately reads
only current road/actor context, candidate poses, and candidate identity.

Optional future costmap adapter contract: use the same ego rear-axle frame;
represent cells outside the known query disk as UNKNOWN (never free), preserve
polygon holes as non-drivable, and use time-indexed occupancy for moving actors.
Retain the full-vehicle footprint check. Grid resolution, origin and frame
transform must accompany any raster. No raster is silently substituted here.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

try:
    from shapely.geometry import GeometryCollection, Polygon
    from shapely.ops import unary_union
    from shapely.prepared import prep
except ImportError:  # The public adapter fails closed if its dependency is absent.
    Polygon = None


DEFAULT_CONFIG = {
    "actor_motion_mode": "constant_velocity",
    "geometry_margin_m": 0.1,
    "max_translation_step_m": 0.2,
    "max_yaw_step_deg": 2.0,
    "max_time_step_s": 0.1,
    "trajectory_dt_s": 0.5,
    "expected_future_points": 8,
}

LIMITATIONS = [
    "Current NAVSIM annotations are a perception proxy, not a deployed sensor stack.",
    "Dynamic actors use their current absolute velocity in the fixed current ego frame, constant heading and constant dimensions; static categories remain fixed. This uncalibrated prediction does not model acceleration, turns or interactions.",
    "Pass means all sampled, margin-expanded footprints satisfy the chosen motion hypothesis, not collision-free future execution or TTC.",
    "Piecewise-linear XY and shortest-angle yaw interpolation do not establish tracking, vehicle dynamics, braking feasibility or a continuous swept-volume guarantee.",
    "The exported drivable polygon union does not encode all traffic rules, signal states, lane direction, right of way or route permissions.",
    "The known map domain is conservatively limited to the exported query disk; absence of actors is conditional on the supplied current annotation proxy.",
    "Localization, shape, map and perception errors are not calibrated by the fixed 0.1 m default margin.",
]

DYNAMIC_CATEGORIES = {"vehicle", "pedestrian", "bicycle", "ego"}
STATIC_CATEGORIES = {"traffic_cone", "barrier", "czone_sign", "generic_object"}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _configuration(config: Mapping | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        if not isinstance(config, Mapping):
            raise ValueError("safety config must be a mapping")
        extra = set(config) - set(cfg)
        if extra:
            raise ValueError(f"unknown safety config keys: {sorted(extra)}")
        cfg.update(config)
    for key, value in cfg.items():
        if key == "actor_motion_mode":
            if value not in ("constant_velocity", "frozen"):
                raise ValueError(f"invalid safety config {key}: {value!r}")
            continue
        if not _finite(value) or (value < 0 if key == "geometry_margin_m" else value <= 0):
            raise ValueError(f"invalid safety config {key}: {value!r}")
    if not isinstance(cfg["expected_future_points"], int):
        raise ValueError("expected_future_points must be an integer")
    return cfg


def _wrap(yaw: float) -> float:
    return (yaw + math.pi) % (2 * math.pi) - math.pi


def _rectangle(x: float, y: float, yaw: float, front: float, rear: float, width: float):
    """Rear-axle-origin rectangle, counterclockwise. Actors use front=rear=L/2."""
    c, s = math.cos(yaw), math.sin(yaw)
    local = [(-rear, -width / 2), (front, -width / 2), (front, width / 2), (-rear, width / 2)]
    return Polygon([(x + c * a - s * b, y + s * a + c * b) for a, b in local])


def _samples(poses: list, cfg: dict, max_actor_speed_mps: float = 0.0):
    """Bound ego/actor relative center motion, time and ego yaw between samples.

    The triangle-inequality bound is ego segment length + max actor speed * dt.
    This densifies crossing/approaching checks but is not continuous collision
    detection. With zero actor speed, the original frozen sampling is unchanged.
    """
    previous = (0.0, 0.0, 0.0)
    yield 0.0, 0.0, 0.0, 0.0
    dt = cfg["trajectory_dt_s"]
    for index, point in enumerate(poses):
        x0, y0, a0 = previous
        x1, y1, a1 = point
        da = _wrap(a1 - a0)
        subdivisions = max(1, math.ceil(max(
            (math.hypot(x1 - x0, y1 - y0) + max_actor_speed_mps * dt) / cfg["max_translation_step_m"],
            abs(da) / math.radians(cfg["max_yaw_step_deg"]),
            dt / cfg["max_time_step_s"],
        )))
        for step in range(1, subdivisions + 1):
            t = step / subdivisions
            yield (index + t) * dt, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, _wrap(a0 + da * t)
        previous = (x1, y1, a1)


def _xy(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(_finite(x) for x in value)


def _ring(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 3 and all(_xy(x) for x in value)


def _context(scene: Mapping, cfg: Mapping | None = None) -> tuple[dict, list[str]]:
    """Validate all current evidence once. Never repair unknown geometry as free."""
    cfg = DEFAULT_CONFIG if cfg is None else cfg
    predicting = cfg["actor_motion_mode"] == "constant_velocity"
    state: dict = {}
    reasons: list[str] = []
    road = scene.get("road_context")
    actors = scene.get("actor_context")
    if not isinstance(road, Mapping):
        return state, ["road_context_missing"]
    if road.get("context_status") != "known":
        reasons.append("road_context_not_known")
    center, radius = road.get("query_center_ego_m"), road.get("query_radius_m")
    if not _xy(center) or not _finite(radius) or radius <= 0:
        reasons.append("map_query_domain_missing_or_invalid")
    else:
        state.update(query_center=list(center), query_radius=float(radius))
    vehicle = road.get("vehicle_parameters")
    keys = ("front_length", "rear_length", "width")
    if not isinstance(vehicle, Mapping) or not all(_finite(vehicle.get(k)) and vehicle[k] > 0 for k in keys):
        reasons.append("vehicle_dimensions_missing_or_invalid")
    else:
        if "length" in vehicle and (not _finite(vehicle["length"]) or abs(vehicle["length"] - vehicle["front_length"] - vehicle["rear_length"]) > 1e-6):
            reasons.append("vehicle_length_inconsistent")
        state["vehicle"] = {key: float(vehicle[key]) for key in keys}
    raw_polygons = road.get("drivable_polygons_ego")
    if not isinstance(raw_polygons, list):
        reasons.append("drivable_polygons_missing")
    else:
        polygons = []
        for index, item in enumerate(raw_polygons):
            if not isinstance(item, Mapping) or not _ring(item.get("exterior")) or not isinstance(item.get("holes", []), list) or not all(_ring(h) for h in item.get("holes", [])):
                reasons.append(f"drivable_polygon_invalid:{index}")
                continue
            polygon = Polygon(item["exterior"], holes=item.get("holes", []))
            if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
                reasons.append(f"drivable_polygon_invalid:{index}")
            else:
                polygons.append(polygon)
        # A known explicit empty list is known non-drivable, not a missing map.
        state["road"] = unary_union(polygons) if polygons else GeometryCollection()
        state["road_prepared"] = prep(state["road"])
        state["road_polygon_count"] = len(raw_polygons)
    if not isinstance(actors, Mapping) or actors.get("actor_proxy_known") is not True or not isinstance(actors.get("actors"), list):
        reasons.append("current_actor_context_missing_or_unknown")
    else:
        obstacles = []
        for index, actor in enumerate(actors["actors"]):
            if not isinstance(actor, Mapping) or not _xy(actor.get("box_center_xy_ego_m")) or not all(_finite(actor.get(k)) for k in ("length_m", "width_m", "heading_ego_rad")) or actor["length_m"] <= 0 or actor["width_m"] <= 0:
                reasons.append(f"current_actor_box_invalid:{index}")
                continue
            x, y = actor["box_center_xy_ego_m"]
            category = actor.get("category", "unknown")
            dynamic = actor.get("is_dynamic_category")
            velocity = actor.get("velocity_xy_ego_mps")
            velocity_valid = actor.get("velocity_known") is True and _xy(velocity)
            applied_velocity = [0.0, 0.0]
            if predicting:
                if not isinstance(category, str) or category not in DYNAMIC_CATEGORIES | STATIC_CATEGORIES:
                    reasons.append(f"current_actor_category_unknown:{index}")
                elif not isinstance(dynamic, bool) or dynamic != (category in DYNAMIC_CATEGORIES):
                    reasons.append(f"current_actor_motion_class_missing_or_inconsistent:{index}")
                elif category == "ego":
                    reasons.append(f"current_actor_includes_ego:{index}")
                elif dynamic:
                    if not velocity_valid:
                        reasons.append(f"current_dynamic_actor_velocity_missing_or_invalid:{index}")
                    else:
                        applied_velocity = [float(v) for v in velocity]
            geometry = _rectangle(x, y, actor["heading_ego_rad"], actor["length_m"] / 2, actor["length_m"] / 2, actor["width_m"])
            obstacles.append({"geometry": geometry, "bounds": geometry.bounds,
                "vertices": list(geometry.exterior.coords), "index": index,
                "annotation_index": actor.get("annotation_index", index), "category": category,
                "is_dynamic_category": dynamic, "current_center_xy_ego_m": [float(x), float(y)],
                "velocity_xy_ego_mps": [float(v) for v in velocity] if velocity_valid else None,
                "applied_velocity_xy_ego_mps": applied_velocity,
                "heading_ego_rad": float(actor["heading_ego_rad"]),
                "length_m": float(actor["length_m"]), "width_m": float(actor["width_m"])})
        state["obstacles"] = obstacles
        state["obstacle_union"] = unary_union([x["geometry"] for x in obstacles]) if obstacles else GeometryCollection()
        state["obstacle_prepared"] = prep(state["obstacle_union"])
        state["actor_count"] = len(actors["actors"])
        state["dynamic_category_count"] = sum(a.get("is_dynamic_category") is True for a in actors["actors"] if isinstance(a, Mapping))
        state["max_actor_speed_mps"] = max((math.hypot(*o["applied_velocity_xy_ego_mps"]) for o in obstacles), default=0.0)
        state["moving_actor_count"] = sum(math.hypot(*o["applied_velocity_xy_ego_mps"]) > 0 for o in obstacles)
        if not _finite(road.get("current_timestamp_us")) or not _finite(actors.get("current_timestamp_us")) or road["current_timestamp_us"] != actors["current_timestamp_us"]:
            reasons.append("current_context_timestamps_missing_or_mismatched")
        if not scene.get("token") or not (scene["token"] == road.get("token") == actors.get("token")):
            reasons.append("current_context_tokens_missing_or_mismatched")
    return state, reasons


def _predicted_geometry(obstacle: dict, time_s: float):
    vx, vy = obstacle["applied_velocity_xy_ego_mps"]
    if vx == 0 and vy == 0:
        return obstacle["geometry"]
    return Polygon([(x + vx * time_s, y + vy * time_s) for x, y in obstacle["vertices"]])


def _actor_prediction(obstacle: dict, time_s: float) -> dict:
    """Serializable same-time collision evidence; indices retain source identity."""
    vx, vy = obstacle["applied_velocity_xy_ego_mps"]
    x, y = obstacle["current_center_xy_ego_m"]
    keys = ("index", "annotation_index", "category", "is_dynamic_category",
            "current_center_xy_ego_m", "velocity_xy_ego_mps", "applied_velocity_xy_ego_mps",
            "heading_ego_rad", "length_m", "width_m")
    return dict({k: obstacle[k] for k in keys},
                predicted_center_xy_ego_m=[x + vx * time_s, y + vy * time_s],
                polygon_ego_xy=[list(p) for p in _predicted_geometry(obstacle, time_s).exterior.coords],
                prediction_time_s=time_s, motion_applied=(vx != 0 or vy != 0))


def _time_aligned_actor_check(footprint, obstacles: list, time_s: float):
    """Exact rectangle queries after an AABB lower-bound distance prefilter.

    Far actors need no rebuilt polygons. The lower bound only prunes actors
    farther than the best exact distance already computed; all intersecting
    rectangles are therefore retained, including boundary contacts.
    """
    fx0, fy0, fx1, fy1 = footprint.bounds
    ordered = []
    for obstacle in obstacles:
        ax0, ay0, ax1, ay1 = obstacle["bounds"]
        vx, vy = obstacle["applied_velocity_xy_ego_mps"]
        dx, dy = vx * time_s, vy * time_s
        bound = math.hypot(max(ax0 + dx - fx1, fx0 - ax1 - dx, 0.0),
                           max(ay0 + dy - fy1, fy0 - ay1 - dy, 0.0))
        ordered.append((bound, obstacle["index"], obstacle))
    hits, clearance = [], math.inf
    for bound, _, obstacle in sorted(ordered, key=lambda item: (item[0], item[1])):
        if bound > clearance + 1e-12:
            break
        geometry = _predicted_geometry(obstacle, time_s)
        clearance = min(clearance, footprint.distance(geometry))
        if footprint.intersects(geometry):
            hits.append(obstacle)
    return sorted(hits, key=lambda o: o["index"]), clearance


def evaluate_scene(scene: Mapping, config: Mapping | None = None) -> list[dict]:
    """Return one fail-closed result per candidate, sorted by numeric anchor ID.

    Only ``candidate['row']['anchor_id']`` and ``candidate['poses']`` are read.
    Reasons never derive from model score, offline metrics or future ground truth.
    ``status='unknown'`` and ``eligible=False`` distinguish missing evidence or
    query-domain escape from a demonstrated collision/non-drivable sample.
    """
    cfg = _configuration(config)
    if not isinstance(scene, Mapping) or not isinstance(scene.get("candidates"), list):
        raise ValueError("scene must have a candidates list")
    if Polygon is None:
        state, context_reasons = {}, ["shapely_dependency_unavailable"]
    else:
        state, context_reasons = _context(scene, cfg)
    predicting = cfg["actor_motion_mode"] == "constant_velocity"
    candidates = []
    for index, candidate in enumerate(scene["candidates"]):
        row = candidate.get("row") if isinstance(candidate, Mapping) else None
        identifier = row.get("anchor_id") if isinstance(row, Mapping) else None
        valid_id = isinstance(identifier, int) and not isinstance(identifier, bool) and identifier >= 0
        candidates.append((identifier if valid_id else None, index, candidate))
    counts = Counter(identifier for identifier, _, _ in candidates if identifier is not None)
    results = []
    for identifier, original_index, candidate in sorted(candidates, key=lambda x: (x[0] is None, x[0] if x[0] is not None else x[1])):
        unknown = list(context_reasons)
        if identifier is None:
            unknown.append("candidate_id_missing_or_invalid")
        elif counts[identifier] != 1:
            unknown.append("candidate_id_duplicate")
        poses = candidate.get("poses") if isinstance(candidate, Mapping) else None
        if not isinstance(poses, list) or len(poses) != cfg["expected_future_points"] or not all(isinstance(p, (list, tuple)) and len(p) == 3 and all(_finite(v) for v in p) for p in poses):
            unknown.append("trajectory_poses_missing_or_invalid")
        diagnostic = {
            "adapter": "current_environment_prediction" if predicting else "current_environment_geometry",
            "status": "unknown" if unknown else "pass",
            "config": dict(cfg),
            "coordinate_frame": "current_ego_rear_axle_x_forward_y_left_yaw_ccw_radians",
            "scope": "drivable_union_and_time_aligned_actor_boxes" if predicting else "drivable_union_and_all_current_actor_boxes_frozen",
            "actor_motion_mode": cfg["actor_motion_mode"],
            "motion_mode": "time_aligned_constant_velocity" if predicting else "frozen_current_boxes",
            "limitations": list(LIMITATIONS) if predicting else [LIMITATIONS[0], "Every current actor box is frozen, including dynamic categories and stopped vehicles; future motion is not predicted."] + LIMITATIONS[2:],
            "source_candidate_index": original_index,
            "actor_count": state.get("actor_count"),
            "dynamic_category_count": state.get("dynamic_category_count"),
            "frozen_dynamic_category_count": 0 if predicting else state.get("dynamic_category_count"),
            "predicted_moving_actor_count": state.get("moving_actor_count"),
            "maximum_applied_actor_speed_mps": state.get("max_actor_speed_mps"),
            "actor_velocity_semantics": "Absolute object velocity in the fixed current ego frame; not relative closing velocity. Static categories always use zero applied velocity.",
            "sampling_translation_bound": "ego_segment_distance + max_applied_actor_speed * segment_duration" if predicting else "ego_segment_distance",
            "map_polygon_count": state.get("road_polygon_count"),
            "map_query_center_ego_m": state.get("query_center"),
            "map_query_radius_m": state.get("query_radius"),
            "vehicle_rear_axle_dimensions_m": state.get("vehicle"),
            "footprint_margin_definition": "Expand front, rear and each lateral side by geometry_margin_m; rectangular, not a circular buffer.",
            "samples_checked": 0,
            "total_samples": 0,
            "first_failure": None,
            "first_unknown": None,
            "minimum_sampled_road_boundary_clearance_m": None,
            "minimum_sampled_current_actor_clearance_m": None,
            "minimum_sampled_time_aligned_actor_clearance_m": None,
            "road_boundary_contact_allowed": True,
            "actor_boundary_contact_is_collision": True,
            "uses_future_ground_truth": False,
            "uses_offline_safety_metrics": False,
            "uses_model_score": False,
        }
        result = {"candidate_id": identifier, "eligible": False, "reasons": unknown, "diagnostics": diagnostic}
        if unknown:
            results.append(result)
            continue
        samples = list(_samples(poses, cfg, state["max_actor_speed_mps"] if predicting else 0.0))
        diagnostic["total_samples"] = len(samples)
        v, margin = state["vehicle"], cfg["geometry_margin_m"]
        min_road, min_actor = math.inf, math.inf
        for sample_index, (time_s, x, y, yaw) in enumerate(samples):
            footprint = _rectangle(x, y, yaw, v["front_length"] + margin, v["rear_length"] + margin, v["width"] + 2 * margin)
            diagnostic["samples_checked"] += 1
            site = {"time_s": time_s, "x": x, "y": y, "yaw_rad": yaw, "sample_index": sample_index, "footprint_ego_xy": [list(p) for p in footprint.exterior.coords]}
            cx, cy = state["query_center"]
            if any(math.hypot(px - cx, py - cy) > state["query_radius"] for px, py in footprint.exterior.coords):
                reason = "footprint_outside_known_map_query_domain"
                diagnostic["first_unknown"] = dict(site, reason=reason)
                diagnostic["status"] = "unknown"
                result["reasons"] = [reason]
                break
            road_ok = state["road_prepared"].covers(footprint)
            if predicting:
                hits, actor_clearance = _time_aligned_actor_check(footprint, state["obstacles"], time_s)
            else:
                collides = not state["obstacle_union"].is_empty and state["obstacle_prepared"].intersects(footprint)
                hits = [o for o in state["obstacles"] if footprint.intersects(o["geometry"])] if collides else []
                actor_clearance = footprint.distance(state["obstacle_union"]) if not state["obstacle_union"].is_empty else math.inf
            if not state["road"].is_empty:
                min_road = min(min_road, footprint.distance(state["road"].boundary) if road_ok else 0.0)
            min_actor = min(min_actor, actor_clearance)
            reasons = []
            if not road_ok:
                reasons.append("footprint_outside_drivable_union")
            if hits:
                reasons.append("footprint_intersects_time_aligned_actor" if predicting else "footprint_intersects_frozen_current_actor")
            if reasons:
                diagnostic["status"] = "fail"
                diagnostic["first_failure"] = dict(site, reasons=reasons, actor_indices=[o["index"] for o in hits], actor_annotation_indices=[o["annotation_index"] for o in hits], actor_categories=[o["category"] for o in hits], actor_predictions=[_actor_prediction(o, time_s) for o in hits], outside_drivable_area_m2=footprint.difference(state["road"]).area if not road_ok else 0.0)
                result["reasons"] = reasons
                break
        diagnostic["minimum_sampled_road_boundary_clearance_m"] = min_road if math.isfinite(min_road) else None
        clearance_key = "minimum_sampled_time_aligned_actor_clearance_m" if predicting else "minimum_sampled_current_actor_clearance_m"
        diagnostic[clearance_key] = min_actor if math.isfinite(min_actor) else None
        diagnostic["clearance_scope"] = "checked_samples_only_up_to_first_failure_or_unknown"
        result["eligible"] = diagnostic["status"] == "pass"
        results.append(result)
    return results
