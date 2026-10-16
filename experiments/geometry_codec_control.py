#!/usr/bin/env python
"""Codec-switch controls at IDENTICAL geometry, not voxel-edit achievements."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from geometry_micro_edit_probe import ActualCodec, unique, load_ply


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', nargs='+', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    settings = dict(current=[], planar_on=['--planarEnabled=1'],
                    no_qtbt=['--maxNumQtBtBeforeOt=0'], no_idcm=['--inferredDirectCodingMode=0'])
    for frame in a.frames:
        q = unique(np.rint(load_ply(frame, return_color=False).astype(np.float64) / 2.))
        baseline = None
        for name, flags in settings.items():
            stats = ActualCodec(2., 2, flags).encode(q, verify=True)
            if baseline is None:
                baseline = stats['bits']
            row = dict(frame=frame, setting=name, flags=flags, **stats,
                       compression_percent_vs_current=100.*(stats['bits']/baseline-1.),
                       physical_geometry_unchanged=True, is_voxel_edit_result=False)
            rows.append(row)
            print(json.dumps(row), flush=True)
        (folder/'results.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
