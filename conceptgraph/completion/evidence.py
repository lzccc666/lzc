"""静止目标接近的确定性证据门限；不读取任何GT。"""
from __future__ import annotations


def target_evidence_decision(node, policy):
    if not policy or not policy.get('enabled', False):
        return True, 'legacy_policy'
    if node.motion_state != 'static':
        return False, 'motion_not_verified_static'
    if node.is_short_term:
        return False, 'short_term_dynamic_track'
    if int(node.metadata.get('num_detections', 0)) < int(policy.get('minimum_observations', 3)):
        return False, 'insufficient_observations'
    evidence=node.metadata.get('depth_evidence', [])
    if not evidence:
        return False, 'missing_depth_provenance'
    latest=evidence[-1]
    if (latest.get('source') != 'current_projected_lidar'
            or latest.get('units') != 'm'
            or int(latest.get('retained_pixels', 0)) < int(policy.get('minimum_lidar_pixels', 5))):
        return False, 'insufficient_measured_depth'
    if float(latest.get('support_fraction', 1.0)) < float(policy.get('minimum_depth_support_fraction', .4)):
        return False, 'ambiguous_depth_clusters'
    return True, 'verified_static_depth_supported'
