"""Exact one-way nearest neighbours for the autonomous route.

An exact CPU KD-tree only shortlists spatial neighbours. The ORIGINAL CUDA
Chamfer kernel selects distances/indices on those candidates, preserving its
FP32 arithmetic and first-index tie rule. No distances are approximated and
gradients flow through the gathered original input tensors.
"""
import numpy as np
import torch


def nearest_squared(queries, reference, chamfer, stats=None, max_candidates=256):
    if (queries.device.type == 'cuda' and queries.dtype == torch.float32
            and reference.device == queries.device and reference.dtype == torch.float32
            and queries.shape[0] == reference.shape[0] == 1
            and queries.shape[1] > 0 and reference.shape[1] > 0):
        from .autonomous_lattice_geometry import nearest_lattice
        answer = nearest_lattice(queries, reference, chamfer, _nearest_kdtree,
                                 stats, max_candidates)
        if answer is not None:
            return answer
    return _nearest_kdtree(queries, reference, chamfer, stats, max_candidates)


def _nearest_kdtree(queries, reference, chamfer, stats=None, max_candidates=256):
    if (queries.device.type != 'cuda' or queries.dtype != torch.float32
            or reference.dtype != torch.float32 or queries.shape[0] != 1
            or reference.shape[0] != 1 or reference.shape[1] == 0):
        return None
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    q = queries[0].detach().cpu().numpy()
    r = reference[0].detach().cpu().numpy()
    if not np.isfinite(q).all() or not np.isfinite(r).all() or not len(q):
        return None
    tree = cKDTree(r)
    if len(r) > 1:
        pair_distance, pair_index = tree.query(q, k=2, eps=0., workers=2)
        distance, first_index = pair_distance[:, 0], pair_index[:, 0]
        second_distance = pair_distance[:, 1]
    else:
        distance, first_index = tree.query(q, k=1, eps=0., workers=2)
        second_distance = np.full(len(q), np.inf)
    # Conservative FP32 rounding envelope: include every candidate that
    # could beat/tie the double-precision neighbour under CUDA FP32 arithmetic.
    magnitude = max(float(np.abs(q).max()), float(np.abs(r).max()), 1.)
    radius = distance * (1. + 8. * np.finfo(np.float32).eps) + 8. * np.finfo(np.float32).eps * magnitude
    unique = second_distance > radius
    # Exactly retained, uniquely matching points have zero distance AND zero
    # coordinate derivative. Do not skip duplicate/tied reference points.
    active = np.flatnonzero(~(unique & (distance == 0.)))
    output_distance = queries[0, :, 0] * 0. + reference[0, 0, 0] * 0.
    output_index = torch.from_numpy(first_index.astype(np.int64)).to(queries.device)
    if not len(active):
        if stats is not None:
            stats.append(dict(queries=len(q), reference=len(r), skipped_zero=len(q),
                              candidate_pairs=0, full_pairs=int(len(q)*len(r))))
        return output_distance.reshape(1, -1), output_index.reshape(1, -1)
    neighbors = [[int(first_index[i])] if unique[i] else None for i in active]
    ambiguous = np.flatnonzero(~unique[active])
    if len(ambiguous):
        tied = tree.query_ball_point(q[active[ambiguous]], radius[active[ambiguous]],
                                    eps=0., workers=2, return_sorted=True)
        for position, row in zip(ambiguous, tied):
            neighbors[position] = row
    lengths = np.fromiter((len(v) for v in neighbors), dtype=np.int64, count=len(active))
    width = int(lengths.max())
    if width > max_candidates or int(lengths.min()) == 0:
        return None  # never truncate candidates or replace the exact result
    indices = np.empty((len(active), width), dtype=np.int64)
    for i, row in enumerate(neighbors):
        indices[i, :len(row)] = row
        indices[i, len(row):] = row[-1]
    index_tensor = torch.from_numpy(indices).to(device=queries.device)
    active_tensor = torch.from_numpy(active).to(device=queries.device)
    distances, nearest = [], []
    # Many tiny independent batches replace Q*N brute-force comparisons.
    for start in range(0, len(active), 8192):
        ids = index_tensor[start:start+8192]
        candidates = reference[0].index_select(0, ids.reshape(-1)).reshape(-1, width, 3)
        batch_queries = queries[0].index_select(0, active_tensor[start:start+8192]).unsqueeze(1).contiguous()
        d, _, local, _ = chamfer(batch_queries, candidates.contiguous())
        distances.append(d.reshape(-1))
        nearest.append(ids.gather(1, local.long()).reshape(-1))
    if stats is not None:
        stats.append(dict(queries=len(q), reference=len(r), max_candidates=width,
                          skipped_zero=len(q)-len(active),
                          candidate_pairs=int(len(active) * width), full_pairs=int(len(q) * len(r))))
    return (output_distance.index_copy(0, active_tensor, torch.cat(distances)).reshape(1, -1),
            output_index.index_copy(0, active_tensor, torch.cat(nearest)).reshape(1, -1))


def chamfer_terms(gen, gt, weights, chamfer, stats=None):
    """Same full-cloud CD/weight surrogate formulas as utils_loss, one NN pass.

    Inputs follow utils_loss [B,3,N]. None requests its original fallback.
    """
    gen_rows, gt_rows = gen.transpose(1, 2).contiguous(), gt.transpose(1, 2).contiguous()
    forward = nearest_squared(gen_rows, gt_rows, chamfer, stats)
    backward = nearest_squared(gt_rows, gen_rows, chamfer, stats)
    if forward is None or backward is None:
        return None
    d1, d2 = forward[0], backward[0]
    hard = d1.mean() + d2.mean()
    if weights is None:
        return hard, hard, hard.detach()
    w = weights.reshape_as(d1).clamp(0., 1.).to(d1.dtype)
    weighted = (d1*w).sum() / (w.sum()+1e-12) + d2.mean()
    surrogate = (d1.detach()*w).sum() / (w.sum()+1e-12) + d2.detach().mean()
    return hard, weighted, surrogate
