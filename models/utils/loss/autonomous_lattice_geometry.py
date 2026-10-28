"""Exact integer-lattice NN with the original Chamfer distance/backward.

Every integer location within sqrt(3) is searched. If any is occupied, no
outside location can be nearer. Uncovered queries use the previous exact
KD-tree; fractional/large coordinates request the previous full path.
"""
import itertools

import torch


def nearest_lattice(queries, reference, chamfer, fallback, stats, max_candidates):
    q, r = queries[0].detach(), reference[0].detach()
    if not (torch.equal(q, q.round()) and torch.equal(r, r.round())):
        return None
    if not bool((torch.isfinite(q).all() & torch.isfinite(r).all()).item()):
        return None
    lower = torch.minimum(q.amin(0), r.amin(0))
    upper = torch.maximum(q.amax(0), r.amax(0))
    if float(torch.maximum(lower.abs(), upper.abs()).amax()) > 2**20:
        return None
    lower = lower.long()
    span = (upper.long()-lower+1).cpu().tolist()
    if span[0]*span[1]*span[2] >= 2**62:
        return None

    def keys(xyz):
        shifted = xyz.long()-lower
        return (shifted[..., 0]*span[1]+shifted[..., 1])*span[2]+shifted[..., 2]

    sorted_keys, order = torch.sort(keys(r), stable=True)

    def lookup(xyz):
        packed = keys(xyz).contiguous()
        positions = torch.searchsorted(sorted_keys, packed)
        safe = positions.clamp_max(len(r)-1)
        valid = (positions < len(r)) & (sorted_keys[safe] == packed)
        # Linear keys outside the domain could alias valid in-domain rows.
        valid &= ((xyz >= lower) & (xyz <= upper)).all(-1)
        return torch.where(valid, order[safe], torch.full_like(safe, -1))

    indices = lookup(q)
    active = torch.nonzero(indices < 0, as_tuple=False).reshape(-1)
    zero_count = len(q)-int(active.numel())
    if active.numel():
        offsets = q.new_tensor([v for v in itertools.product((-1, 0, 1), repeat=3)
                                if v != (0, 0, 0)]).long()
        squared = offsets.square().sum(-1)
        nearest = []
        for start in range(0, int(active.numel()), 8192):
            neighbors = q[active[start:start+8192]].long()[:, None, :]+offsets
            candidate_ids = lookup(neighbors)
            distances = squared.expand_as(candidate_ids).masked_fill(candidate_ids < 0, 4)
            minimum = distances.amin(1, keepdim=True)
            # Include all minimal-distance coordinates and select the first
            # original reference row, including duplicate reference locations.
            tied = (distances == minimum) & (candidate_ids >= 0)
            best = candidate_ids.masked_fill(~tied, len(r)).amin(1)
            nearest.append(best.masked_fill(best == len(r), -1))
        indices = indices.index_copy(0, active, torch.cat(nearest))
    distances = queries[0, :, 0]*0. + reference[0, 0, 0]*0.
    if active.numel():
        found = active[indices[active] >= 0]
        computed = []
        for start in range(0, int(found.numel()), 8192):
            query_ids = found[start:start+8192]
            a = queries[0].index_select(0, query_ids).unsqueeze(1).contiguous()
            b = reference[0].index_select(0, indices[query_ids]).unsqueeze(1).contiguous()
            d, _, _, _ = chamfer(a, b)
            computed.append(d.reshape(-1))
        if computed:
            distances = distances.index_copy(0, found, torch.cat(computed))
    missing = torch.nonzero(indices < 0, as_tuple=False).reshape(-1)
    if missing.numel():
        answer = fallback(queries.index_select(1, missing), reference, chamfer,
                          stats, max_candidates)
        if answer is None:
            return None
        distances = distances.index_copy(0, missing, answer[0].reshape(-1))
        indices = indices.index_copy(0, missing, answer[1].reshape(-1))
    if stats is not None:
        stats.append(dict(queries=len(q), reference=len(r), integer_lattice=True,
            skipped_zero=zero_count, lattice_queries=int(active.numel()),
            kdtree_queries=int(missing.numel()), full_pairs=len(q)*len(r)))
    return distances.reshape(1, -1), indices.reshape(1, -1)
