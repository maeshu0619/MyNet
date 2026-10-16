#!/usr/bin/env python
"""Actual bit/quality controls for changing qs; NOT micro-edit results."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import torch
from geometry_micro_edit_probe import ActualCodec, unique, difference, load_ply


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
        raw_tree = cKDTree(raw)
        baseline_physical = unique(np.rint(raw/2.)) * 2
        frame_rows = []
        for qs in (1., 2., 4., 8.):
            q_abs = unique(np.rint(raw/qs))
            physical = q_abs * qs
            codec = ActualCodec(qs, 2)
            stats = codec.encode(q_abs, verify=True)
            d1 = cKDTree(physical).query(raw, workers=2)[0]
            d2 = raw_tree.query(physical, workers=2)[0]
            removed = len(difference(baseline_physical, physical))
            added = len(difference(physical, baseline_physical))
            row = dict(frame=frame, qs=qs, **stats, voxels=len(physical),
                       chamfer_sq=float(np.mean(d1*d1) + np.mean(d2*d2)),
                       d1_max_direction_mse=float(max(np.mean(d1*d1),np.mean(d2*d2))),
                       removed_vs_qs2=removed, added_vs_qs2=added,
                       min_operation_percent_vs_qs2=100.*max(removed,added)/len(baseline_physical),
                       is_micro_edit=False)
            frame_rows.append(row)
        ref = next(r['bits'] for r in frame_rows if r['qs'] == 2.)
        for row in frame_rows:
            row['compression_percent_vs_qs2'] = 100.*(row['bits']/ref-1.)
            print(json.dumps(row), flush=True)
        rows.extend(frame_rows)
        (folder/'results.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
