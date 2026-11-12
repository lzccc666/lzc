"""Re-read recorded grids and independently audit every accepted/reused path."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from conceptgraph.navigation.lane_constraints import load_lane_topology
from conceptgraph.scripts.lane_expansion_independent_audit import IndependentLaneAudit, continuous_collisions
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT
from conceptgraph.scripts.run_incremental_semantic_navigation import deterministic_replay_signature, write_json


class PathVerifier:
    def __init__(self, root):
        self.root, self.lanes, self.cache, self.poses = root, {}, {}, {}
        self.checks, self.failures = [], []

    def check(self, cells, worlds, occupancy_path, scene, lane, origin, frame=None, remaining_start=0):
        cells = np.asarray(cells, dtype=float).reshape(-1, 2)
        worlds = np.asarray(worlds, dtype=float).reshape(-1, 3)
        if not len(cells):
            return
        if not occupancy_path or not Path(occupancy_path).exists():
            self.failures.append({'artifact': origin, 'frame': frame, 'reason': 'missing_actual_grid'})
            return
        with np.load(occupancy_path) as saved:
            occupancy, xy, resolution = saved['occupancy'], saved['origin_xy'], float(saved['resolution'])
        expected = xy+cells[:, ::-1]*resolution
        aligned = bool(len(worlds) == len(cells) and np.allclose(worlds[:, :2], expected, rtol=0, atol=1e-6))
        remaining = cells[remaining_start:]
        key = hashlib.sha256(occupancy.tobytes()+remaining.tobytes()).hexdigest()
        if key not in self.cache:
            self.cache[key] = continuous_collisions(remaining, occupancy)
        collisions = self.cache[key]
        lane_result = None
        if lane:
            if scene not in self.lanes:
                self.lanes[scene] = IndependentLaneAudit(load_lane_topology(self.root / 'lane_maps' / scene / 'lane_constrained_bev.npz'))
            lane_key = ('lane', scene, worlds[:, :2].tobytes())
            if lane_key not in self.cache:
                self.cache[lane_key] = self.lanes[scene].validate(worlds[:, :2])
            lane_result = self.cache[lane_key]
        passed = aligned and not collisions and (not lane or lane_result['valid'])
        record = {'artifact': origin, 'scene': scene, 'frame_idx': frame, 'lane': lane, 'remaining_start_index': remaining_start, 'path_points': len(cells), 'grid_world_aligned': aligned, 'collision_cells': collisions, 'external_lane_validation': lane_result, 'passed': passed}
        self.checks.append(record)
        if not passed:
            self.failures.append(record)

    def online(self, frames):
        rows = [json.loads(line) for line in frames.open()]
        summary = json.loads((frames.parent / 'summary.json').read_text())
        lane = summary['planning_configuration']['target_selection']['mode'] == 'lane_constrained'
        scene = frames.parent.name
        if deterministic_replay_signature(rows) != summary['deterministic_content_sha256']:
            self.failures.append({'artifact': str(frames), 'reason': 'signature_mismatch'})
        for row in rows:
            audit = row['experiment_audit']
            cells, worlds = audit['path_grid'], audit['path_world']
            if not cells:
                continue
            source = Path(row['source']).parent.parent / 'poses.txt'
            if source not in self.poses:
                self.poses[source] = np.loadtxt(source).reshape(-1, 4, 4)
            vehicle = self.poses[source][row['frame_idx'], :2, 3]
            nearest = int(np.argmin(np.linalg.norm(np.asarray(worlds)[:, :2]-vehicle, axis=1)))
            start = 0 if row['replanned'] else nearest
            self.check(cells, worlds, audit['occupancy_artifact'], scene, lane, str(frames), row['frame_idx'], start)

    def static(self, result_path):
        saved = json.loads(result_path.read_text())
        if not saved['success']:
            return
        plan = json.loads(Path(saved['plan_path']).read_text())
        scene = next((part for part in result_path.parts if part.startswith('scene-') and len(part) == 10), None)
        if scene is None:
            scene = next(part[:10] for part in result_path.parts if part.startswith('scene-'))
        grid = next((p for p in (result_path.parent / 'captured_live_grid.npz', result_path.parent / 'live_grid.npz') if p.exists()), None)
        self.check(plan['path']['grid_rc'], plan['path']['world_xyz'], grid, scene, saved['mode'] == 'lane_constrained', str(result_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-name')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError('invalid run ID')
    run = DEFAULT_ROOT / 'experiments' / args.run_id
    verifier = PathVerifier(DEFAULT_ROOT)
    frames = sorted(run.rglob('frames.jsonl'))
    for path in frames:
        verifier.online(path)
    static = sorted(run.rglob('target_selection_result.json'))
    for path in static:
        verifier.static(path)
    report = {'run_id': args.run_id, 'replay_files': len(frames), 'static_results': len(static), 'accepted_or_reused_paths_checked': len(verifier.checks), 'external_geometry_library': 'Shapely 2.1.2; full closed-cell intersection and segment polygon coverage', 'checks': verifier.checks, 'failures': verifier.failures, 'passed': not verifier.failures, 'zero_paths_is_not_100_percent_safety': True, 'safety_rate': sum(c['passed'] for c in verifier.checks)/len(verifier.checks) if verifier.checks else None}
    write_json(DEFAULT_ROOT / 'path_audits' / (args.output_name or f'{args.run_id}.json'), report)
    print(json.dumps({'run': args.run_id, 'paths': len(verifier.checks), 'failures': len(verifier.failures), 'passed': report['passed']}), flush=True)
    if verifier.failures:
        raise RuntimeError('external safety audit failed; inspect report')


if __name__ == '__main__':
    main()
