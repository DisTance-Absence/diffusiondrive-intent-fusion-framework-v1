"""Geometry contract tests and an auditable 16-scene validation runner.

Offline DAC is read here only for descriptive disagreement counts AFTER the
adapter has evaluated the scene. It is never an input to safety.evaluate_scene.
"""

import copy
import hashlib
import json
import math
import random
from pathlib import Path
import unittest
from collections import Counter

import safety


def polygon(x0, y0, x1, y1, holes=None):
    return {"exterior": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "holes": holes or []}


def fixture():
    return {
        "scene_id": "synthetic",
        "token": "fixture",
        "road_context": {
            "token": "fixture", "current_timestamp_us": 1, "context_status": "known",
            "query_center_ego_m": [0, 0], "query_radius_m": 150,
            "vehicle_parameters": {"front_length": 3, "rear_length": 1, "width": 2, "length": 4},
            "drivable_polygons_ego": [polygon(-100, -100, 100, 100)],
        },
        "actor_context": {"token": "fixture", "current_timestamp_us": 1, "actor_proxy_known": True, "actors": []},
        "candidates": [{"row": {"anchor_id": 0}, "poses": [[i + 1, 0, 0] for i in range(8)]}],
    }


def actor(x=6, y=0, length=1, width=1, dynamic=True, velocity=(0, 0)):
    return {"annotation_index": 45, "category": "vehicle" if dynamic else "barrier", "box_center_xy_ego_m": [x, y], "length_m": length, "width_m": width, "heading_ego_rad": 0, "is_dynamic_category": dynamic, "velocity_known": True, "velocity_xy_ego_mps": list(velocity)}


class SafetyTests(unittest.TestCase):
    def first(self, scene, **config):
        return safety.evaluate_scene(scene, config or None)[0]

    def test_open_map_pass_and_all_samples_checked(self):
        r = self.first(fixture())
        self.assertTrue(r["eligible"])
        self.assertEqual(r["diagnostics"]["status"], "pass")
        self.assertEqual(r["diagnostics"]["samples_checked"], r["diagnostics"]["total_samples"])
        self.assertGreaterEqual(r["diagnostics"]["samples_checked"], 41)

    def test_full_width_and_rear_extent_not_centerline(self):
        for road in [polygon(-100, -.9, 100, .9), polygon(-.9, -100, 100, 100)]:
            s = fixture(); s["road_context"]["drivable_polygons_ego"] = [road]
            r = self.first(s)
            self.assertEqual(r["diagnostics"]["status"], "fail")
            self.assertEqual(r["diagnostics"]["first_failure"]["time_s"], 0)

    def test_polygon_hole_is_not_drivable(self):
        s = fixture()
        s["road_context"]["drivable_polygons_ego"] = [polygon(-100, -100, 100, 100, [[[5, -.2], [5.2, -.2], [5.2, .2], [5, .2]]])]
        r = self.first(s)
        self.assertIn("footprint_outside_drivable_union", r["reasons"])
        self.assertGreater(r["diagnostics"]["first_failure"]["outside_drivable_area_m2"], 0)

    def test_current_dynamic_actor_is_frozen_and_checked(self):
        s = fixture(); s["actor_context"]["actors"] = [actor()]
        r = self.first(s, actor_motion_mode="frozen")
        self.assertFalse(r["eligible"])
        self.assertIn("footprint_intersects_frozen_current_actor", r["reasons"])
        self.assertEqual(r["diagnostics"]["first_failure"]["actor_annotation_indices"], [45])
        self.assertEqual(r["diagnostics"]["frozen_dynamic_category_count"], 1)

    def test_moving_away_lead_passes_but_frozen_rejects(self):
        s = fixture(); s["actor_context"]["actors"] = [actor(velocity=(2, 0))]
        before = copy.deepcopy(s)
        self.assertTrue(self.first(s)["eligible"])
        self.assertFalse(self.first(s, actor_motion_mode="frozen")["eligible"])
        self.assertEqual(s, before)

    def test_approaching_actor_rejected_at_same_time(self):
        s = fixture(); s["actor_context"]["actors"] = [actor(x=15, velocity=(-4, 0))]
        self.assertTrue(self.first(s, actor_motion_mode="frozen")["eligible"])
        r = self.first(s)
        self.assertIn("footprint_intersects_time_aligned_actor", r["reasons"])
        f = r["diagnostics"]["first_failure"]
        prediction = f["actor_predictions"][0]
        self.assertGreater(f["time_s"], 0)
        self.assertAlmostEqual(prediction["predicted_center_xy_ego_m"][0], 15-4*f["time_s"])
        self.assertEqual(prediction["current_center_xy_ego_m"], [15, 0])
        self.assertEqual(prediction["applied_velocity_xy_ego_mps"], [-4, 0])
        self.assertEqual(prediction["prediction_time_s"], f["time_s"])
        self.assertTrue(prediction["motion_applied"])
        self.assertIsNone(r["diagnostics"]["minimum_sampled_current_actor_clearance_m"])
        self.assertEqual(r["diagnostics"]["minimum_sampled_time_aligned_actor_clearance_m"], 0)

    def test_fast_crossing_between_original_time_samples(self):
        s = fixture()
        s["road_context"]["vehicle_parameters"] = {"front_length": .1, "rear_length": .1, "width": .2}
        s["candidates"][0]["poses"] = [[0, 0, 0]] * 8
        # Crosses the stationary ego at t=.05, between the former 0,.1 samples.
        s["actor_context"]["actors"] = [actor(x=0, y=2, length=.2, width=.2, velocity=(0, -40))]
        self.assertTrue(self.first(s, actor_motion_mode="frozen", geometry_margin_m=0)["eligible"])
        r = self.first(s, geometry_margin_m=0)
        self.assertFalse(r["eligible"])
        self.assertGreater(r["diagnostics"]["first_failure"]["time_s"], 0)
        self.assertLess(r["diagnostics"]["first_failure"]["time_s"], .1)
        self.assertGreater(r["diagnostics"]["total_samples"], 41)

    def test_dynamic_velocity_unknown_or_invalid_is_unknown(self):
        for change in [{"velocity_known": False}, {"velocity_xy_ego_mps": None},
                       {"velocity_xy_ego_mps": [float("nan"), 0]},
                       {"velocity_xy_ego_mps": [1]}, {"velocity_xy_ego_mps": [True, 0]}]:
            s = fixture(); a = actor(); a.update(change); s["actor_context"]["actors"] = [a]
            r = self.first(s)
            self.assertEqual(r["diagnostics"]["status"], "unknown")
            self.assertIn("current_dynamic_actor_velocity_missing_or_invalid:0", r["reasons"])
            self.assertEqual(r["diagnostics"]["samples_checked"], 0)
            # Deliberate historical ablation needs only the current box.
            self.assertEqual(self.first(s, actor_motion_mode="frozen")["diagnostics"]["status"], "fail")

    def test_motion_classification_is_not_silently_assumed_static(self):
        for change in [{"is_dynamic_category": None}, {"is_dynamic_category": False},
                       {"category": "unknown"}, {"category": []}, {"category": "ego"}]:
            s = fixture(); a = actor(); a.update(change); s["actor_context"]["actors"] = [a]
            self.assertEqual(self.first(s)["diagnostics"]["status"], "unknown")

    def test_static_actor_ignores_nonzero_velocity(self):
        s = fixture(); s["actor_context"]["actors"] = [actor(dynamic=False, velocity=(100, 0))]
        r = self.first(s)
        self.assertFalse(r["eligible"])
        prediction = r["diagnostics"]["first_failure"]["actor_predictions"][0]
        self.assertEqual(prediction["velocity_xy_ego_mps"], [100, 0])
        self.assertEqual(prediction["applied_velocity_xy_ego_mps"], [0, 0])
        self.assertEqual(prediction["predicted_center_xy_ego_m"], [6, 0])
        self.assertFalse(prediction["motion_applied"])
        self.assertEqual(r["diagnostics"]["maximum_applied_actor_speed_mps"], 0)

    def test_zero_speed_dynamic_matches_frozen_geometry(self):
        s = fixture(); s["actor_context"]["actors"] = [actor(velocity=(0, 0))]
        cv, frozen = self.first(s), self.first(s, actor_motion_mode="frozen")
        for field in ["time_s", "sample_index", "footprint_ego_xy", "actor_indices"]:
            self.assertEqual(cv["diagnostics"]["first_failure"][field], frozen["diagnostics"]["first_failure"][field])
        self.assertEqual(cv["diagnostics"]["total_samples"], frozen["diagnostics"]["total_samples"])

    def test_motion_sampling_bounds_relative_center_distance(self):
        samples = list(safety._samples([[i+1, 0, 0] for i in range(8)], safety.DEFAULT_CONFIG, 40))
        for a, b in zip(samples, samples[1:]):
            self.assertLessEqual(math.hypot(b[1]-a[1], b[2]-a[2]) + 40*(b[0]-a[0]), .2+1e-12)

    def test_prefilter_matches_naive_geometry_for_random_actor_layouts(self):
        rng = random.Random(615)
        s = fixture()
        for _ in range(10):
            objects = []
            for i in range(12):
                a = actor(x=rng.uniform(-15, 15), y=rng.uniform(-15, 15),
                          length=rng.uniform(.2, 6), width=rng.uniform(.2, 3),
                          dynamic=(i % 3 != 0), velocity=(rng.uniform(-10, 10), rng.uniform(-10, 10)))
                a["heading_ego_rad"] = rng.uniform(-math.pi, math.pi)
                objects.append(a)
            s["actor_context"]["actors"] = objects
            state, errors = safety._context(s)
            self.assertEqual(errors, [])
            for t in [0, .15, 1.25, 4]:
                fp = safety._rectangle(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-math.pi, math.pi), 3, 1, 2)
                hits, distance = safety._time_aligned_actor_check(fp, state["obstacles"], t)
                geometries = []
                for a in objects:
                    x, y = a["box_center_xy_ego_m"]
                    vx, vy = a["velocity_xy_ego_mps"] if a["is_dynamic_category"] else [0, 0]
                    geometries.append(safety._rectangle(x+vx*t, y+vy*t, a["heading_ego_rad"], a["length_m"]/2, a["length_m"]/2, a["width_m"]))
                self.assertEqual([h["index"] for h in hits], [i for i,g in enumerate(geometries) if fp.intersects(g)])
                self.assertAlmostEqual(distance, min(fp.distance(g) for g in geometries), places=11)

    def test_frozen_reproduces_prechange_s13_s14_diagnosis(self):
        base = Path(__file__).resolve().parent
        inventory = json.loads((base.parent/"scene-choice-inventory-v3/inventory.json").read_text(encoding="utf-8"))
        saved = json.loads((base/"work/s13_s14_guard_diagnosis.json").read_text(encoding="utf-8"))
        expected = {(r["scene_id"], r["candidate_id"]): r["actual_saved_guard"] for r in saved["candidate_records"]}
        for scene in inventory["scenes"]:
            if scene["scene_id"] not in ("S13", "S14"):
                continue
            for result in safety.evaluate_scene(scene, {"actor_motion_mode": "frozen"}):
                old = expected[(scene["scene_id"], result["candidate_id"])]
                self.assertEqual(result["eligible"], old["eligible"])
                self.assertEqual(result["reasons"], old["reasons"])
                first = dict(result["diagnostics"]["first_failure"])
                first.pop("actor_predictions")
                self.assertEqual(first, old["first_failure"])

    def test_actor_touch_is_collision(self):
        s = fixture(); s["actor_context"]["actors"] = [actor(x=3.5)]
        r = self.first(s, geometry_margin_m=0)
        self.assertEqual(r["diagnostics"]["first_failure"]["time_s"], 0)

    def test_road_touch_is_allowed_and_margin_is_explicit(self):
        s = fixture(); s["road_context"]["drivable_polygons_ego"] = [polygon(-100, -1, 100, 1)]
        self.assertTrue(self.first(s, geometry_margin_m=0)["eligible"])
        self.assertEqual(self.first(s)["diagnostics"]["status"], "fail")

    def test_unknown_map_and_known_empty_map_differ(self):
        s = fixture(); del s["road_context"]["drivable_polygons_ego"]
        self.assertEqual(self.first(s)["diagnostics"]["status"], "unknown")
        s["road_context"]["drivable_polygons_ego"] = []
        self.assertEqual(self.first(s)["diagnostics"]["status"], "fail")

    def test_outside_query_is_unknown_even_when_polygon_extends(self):
        s = fixture(); s["road_context"]["query_radius_m"] = 5
        r = self.first(s)
        self.assertEqual(r["diagnostics"]["status"], "unknown")
        self.assertFalse(r["eligible"])
        self.assertIsNone(r["diagnostics"]["first_failure"])
        self.assertIsNotNone(r["diagnostics"]["first_unknown"])

    def test_unknown_actor_context_and_invalid_box_fail_closed(self):
        s = fixture(); del s["actor_context"]
        self.assertEqual(self.first(s)["diagnostics"]["status"], "unknown")
        s = fixture(); s["actor_context"]["actors"] = [actor(width=-1)]
        self.assertEqual(self.first(s)["diagnostics"]["status"], "unknown")

    def test_timestamps_and_identity_must_align(self):
        for key, value in [("current_timestamp_us", 2), ("token", "different")]:
            s = fixture(); s["actor_context"][key] = value
            self.assertEqual(self.first(s)["diagnostics"]["status"], "unknown")

    def test_invalid_polygon_is_unknown_without_silent_repair(self):
        s = fixture(); s["road_context"]["drivable_polygons_ego"] = [{"exterior": [[0, 0], [1, 1], [1, 0], [0, 1]], "holes": []}]
        self.assertEqual(self.first(s)["diagnostics"]["status"], "unknown")

    def test_missing_vehicle_and_invalid_poses_fail_closed(self):
        s = fixture(); del s["road_context"]["vehicle_parameters"]
        self.assertEqual(self.first(s)["diagnostics"]["status"], "unknown")
        for bad in [[], [[0, 0, float("nan")]] * 8, [[0, 0]] * 8]:
            s = fixture(); s["candidates"][0]["poses"] = bad
            self.assertEqual(self.first(s)["diagnostics"]["status"], "unknown")

    def test_rotation_uses_full_footprint(self):
        s = fixture(); s["road_context"]["drivable_polygons_ego"] = [polygon(-2, -1.2, 4, 1.2)]
        s["candidates"][0]["poses"] = [[0, 0, math.pi / 2]] * 8
        r = self.first(s)
        self.assertEqual(r["diagnostics"]["status"], "fail")
        self.assertGreater(r["diagnostics"]["first_failure"]["time_s"], 0)
        self.assertLess(r["diagnostics"]["first_failure"]["time_s"], .5)

    def test_interpolation_limits_and_shortest_yaw(self):
        poses = [[10, 0, math.radians(179)], [10, 0, math.radians(-179)]] + [[10, 0, math.radians(-179)]] * 6
        samples = list(safety._samples(poses, safety.DEFAULT_CONFIG))
        for a, b in zip(samples, samples[1:]):
            self.assertLessEqual(b[0] - a[0], .1 + 1e-12)
            self.assertLessEqual(math.hypot(b[1] - a[1], b[2] - a[2]), .2 + 1e-12)
            self.assertLessEqual(abs(safety._wrap(b[3] - a[3])), math.radians(2) + 1e-12)
        second = [s for s in samples if .5 < s[0] <= 1]
        self.assertEqual(len(second), 5)
        self.assertTrue(all(abs(s[3]) > math.radians(178) for s in second))
        self.assertEqual(samples[-1][0], 4)

    def test_thin_obstacle_between_saved_points(self):
        s = fixture()
        s["road_context"]["vehicle_parameters"] = {"front_length": .05, "rear_length": .05, "width": .1}
        s["candidates"][0]["poses"] = [[i + 1, 0, 0] for i in range(8)]
        s["actor_context"]["actors"] = [actor(x=.6, length=.05, width=1)]
        r = self.first(s, geometry_margin_m=0)
        self.assertEqual(r["diagnostics"]["status"], "fail")
        self.assertGreater(r["diagnostics"]["first_failure"]["time_s"], 0)
        self.assertLess(r["diagnostics"]["first_failure"]["time_s"], .5)

    def test_ids_sorted_and_duplicates_fail_closed(self):
        s = fixture(); one = copy.deepcopy(s["candidates"][0]); one["row"]["anchor_id"] = 1
        s["candidates"].insert(0, one)
        self.assertEqual([r["candidate_id"] for r in safety.evaluate_scene(s)], [0, 1])
        s["candidates"].append(copy.deepcopy(one))
        self.assertTrue(all(not r["eligible"] for r in safety.evaluate_scene(s) if r["candidate_id"] == 1))

    def test_no_candidate_metrics_or_future_fields_read(self):
        class Row(dict):
            def get(self, key, default=None):
                if key != "anchor_id":
                    raise AssertionError("forbidden candidate row access: " + key)
                return super().get(key, default)
        class Candidate(dict):
            def get(self, key, default=None):
                if key not in ("row", "poses"):
                    raise AssertionError("forbidden candidate access: " + key)
                return super().get(key, default)
        s = fixture(); old = s["candidates"][0]
        s["candidates"] = [Candidate(row=Row(anchor_id=0), poses=old["poses"], endpoint_direction=object(), future=object())]
        self.assertTrue(self.first(s)["eligible"])

    def test_bad_configuration_rejected(self):
        for config in [{"geometry_margin_m": -1}, {"max_yaw_step_deg": 0}, {"typo": 1}, {"expected_future_points": 8.5}, {"actor_motion_mode": "oracle"}, {"actor_motion_mode": None}]:
            with self.assertRaises(ValueError):
                safety.evaluate_scene(fixture(), config)


def validate_inventory(test_result):
    base = Path(__file__).resolve().parent
    source = base.parent / "scene-choice-inventory-v3" / "inventory.json"
    raw = source.read_bytes()
    inventory = json.loads(raw)
    before_hash = hashlib.sha256(raw).hexdigest()
    totals, disagreement = Counter(), Counter()
    scene_rows, all_results = [], {}
    for scene in inventory["scenes"]:
        rows = safety.evaluate_scene(scene)
        # Offline descriptive comparison occurs only here, after evaluation.
        dac_by_id = {c["row"]["anchor_id"]: c["row"].get("DAC") for c in scene["candidates"]}
        counts = Counter(r["diagnostics"]["status"] for r in rows)
        totals.update(counts)
        scene_diff = Counter()
        for r in rows:
            dac = dac_by_id[r["candidate_id"]]
            key = f"geometry_{r['diagnostics']['status']}_saved_DAC_{dac}"
            scene_diff[key] += 1
            disagreement[key] += 1
        scene_rows.append({"scene_id": scene["scene_id"], "candidate_count": len(rows), "status_counts": dict(counts), "all_current_geometry_eligible": all(r["eligible"] for r in rows), "first_reason_counts": dict(Counter(r["reasons"][0] for r in rows if r["reasons"])), "saved_DAC_comparison": dict(scene_diff)})
        all_results[scene["scene_id"]] = rows
    assert len(scene_rows) == 16 and sum(totals.values()) == 320
    assert all(len(v) == 20 and [r["candidate_id"] for r in v] == list(range(20)) for v in all_results.values())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_hash
    report = {
        "status": "PASS" if test_result.wasSuccessful() else "FAIL",
        "test_count": test_result.testsRun,
        "adapter": "current_environment_prediction",
        "source_inventory": str(source), "source_sha256": before_hash,
        "configuration": dict(safety.DEFAULT_CONFIG),
        "scene_count": len(scene_rows), "candidate_count": sum(totals.values()),
        "status_counts": dict(totals),
        "saved_DAC_comparison": dict(disagreement),
        "comparison_note": "Descriptive disagreement only: this margin-expanded time-aligned constant-velocity sampled filter has a different scope from saved DAC. DAC is not used by the adapter.",
        "limitations": list(safety.LIMITATIONS),
        "per_scene": scene_rows,
        "results_by_scene": all_results,
    }
    destination = base / "work" / "safety_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "test_count", "scene_count", "candidate_count", "status_counts", "saved_DAC_comparison")}, ensure_ascii=True))
    return report


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SafetyTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    validate_inventory(result)
