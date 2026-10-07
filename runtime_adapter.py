"""Integration boundary for real all-20 model outputs and current information.

This provides the callable fusion boundary; it does not run DiffusionDrive,
subscribe to ROS topics, publish controls, or synthesize a braking trajectory.
"""
from fusion import fuse_candidates
from safety import evaluate_scene


def select_from_model_output(candidate_poses, candidate_logits, road_context,
                             actor_context, command=None, config=None, guard_config=None):
    """Accept 20x8x3 poses and 20 original logits before model argmax/gather.

    Inputs are already detached, on CPU; arrays with tolist() are supported.
    road_context and actor_context follow the current ego rear-axle-frame
    contract demonstrated in the frozen inventory. Missing/mismatched
    contexts fail closed; the caller must check wall-clock freshness and frame
    transforms before invoking this adapter. Dynamic actors require a known
    finite absolute velocity in the same fixed ego frame; the default model
    extrapolates it at constant velocity with fixed actor yaw, up to 4 seconds.
    Real sensors/prediction or an online costmap may
    replace evaluate_scene while preserving per-candidate ID and status.
    """
    poses = candidate_poses.tolist() if hasattr(candidate_poses, 'tolist') else candidate_poses
    logits = candidate_logits.tolist() if hasattr(candidate_logits, 'tolist') else candidate_logits
    if len(poses) != 20 or len(logits) != 20:
        raise ValueError('This adapter requires all 20 original candidates in original ID order.')
    candidates = [dict(id=i, poses=p, model_logit=float(z)) for i,(p,z) in enumerate(zip(poses,logits))]
    # Validate geometry/numerics even if all current-context checks fail.
    # fuse_candidates also performs this validation before a selected output.
    scene = dict(token=road_context.get('token') if isinstance(road_context,dict) else None,
                 road_context=road_context, actor_context=actor_context,
                 candidates=[dict(row=dict(anchor_id=c['id']),poses=c['poses']) for c in candidates])
    assessments = evaluate_scene(scene, guard_config)
    result = fuse_candidates(candidates, assessments, command, config)
    result['environment_assessments'] = assessments
    motion_mode = (guard_config or {}).get('actor_motion_mode', 'constant_velocity')
    result['safety_mode'] = ('current_environment_geometry' if motion_mode == 'frozen'
                             else 'current_environment_prediction')
    result['controller_action'] = 'request_stop_controller' if result['stop_requested'] else 'track_selected_saved_trajectory'
    result['execution_performed'] = False
    return result
