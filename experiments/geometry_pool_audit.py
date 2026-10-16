#!/usr/bin/env python
"""Read-only proposal coverage audit against independently successful edits."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
import torch
from geometry_micro_edit_probe import ROOT, load_ply, unique, difference, quality_guard_variants
from models.utils.pointcloud.geometry_joint_proposals import attach_joint_geometry_guidance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', nargs='+', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for frame in a.frames:
        raw = load_ply(frame, return_color=False).astype(np.float64)
        q = unique(np.rint(raw/2.))
        origin = q.min(0)
        q -= origin
        args = SimpleNamespace(compress='G-PCC', gpcc_prequantize=True,
                               gpcc_effective_qs=2., geometry_octree_initial_total_ratio=.0025)
        exact = attach_joint_geometry_guidance(dict(full_global_voxel_coords=torch.from_numpy(q.T.copy())[None]),
                                               args, device='cpu')['ana_den6_ranked_candidate_guidance']
        pools = exact['operation_edit_units']
        one = {tuple(c['remove_coords'][0]) for c in pools['Prune'] if len(c['remove_coords']) == 1}
        any_remove = {tuple(x) for c in pools['Prune'] for x in c['remove_coords']}
        degree = cKDTree(q).query_ball_point(q, np.sqrt(3)+1e-6, workers=2, return_length=True)-1
        order = np.argsort(degree, kind='stable')
        coverage = []
        for budget in (.0015, .003):
            target = {tuple(v) for v in q[order[:round(len(q)*budget)]]}
            coverage.append(dict(budget_percent=budget*100, desired=len(target),
                                 covered_single=len(target & one), covered_any_bundle=len(target & any_remove),
                                 single_coverage_percent=100.*len(target & one)/len(target),
                                 bundle_coverage_percent=100.*len(target & any_remove)/len(target)))
        row = dict(frame=frame, voxels=len(q), pool_counts={op:len(v) for op,v in pools.items()},
                   sparse_prune_coverage=coverage,
                   boundary_extension_candidates={op:sum(any(any(v<0 for v in x) for x in c['add_coords'])
                                                         for c in pool) for op,pool in pools.items()})
        safe_coverage = []
        for name, out, count in quality_guard_variants(q,raw,origin,2.,[.0015,.003]):
            target = set(map(tuple,difference(q,out).tolist()))
            safe_coverage.append(dict(trial=name, desired=len(target),
                covered_single=len(target & one), covered_any_bundle=len(target & any_remove),
                single_coverage_percent=100.*len(target & one)/len(target),
                bundle_coverage_percent=100.*len(target & any_remove)/len(target)))
        row['quality_guard_prune_coverage'] = safe_coverage
        rows.append(row)
        print(json.dumps(row), flush=True)
        (folder/'results.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
