"""Empirical hierarchical occupancy PRIOR, never labelled as Actual bits.

Add/Prune/Adjust share one tree. Ancestor creation/removal is accounted for.
The model is frozen per input; only executed codec results supervise Actual.
"""
from collections import OrderedDict
import numpy as np


class GeometryOctreeRate:
    NEIGHBOURS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))

    def __init__(self, rows, codec='gpcc'):
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
        self.depth = max(int(rows.max()).bit_length(), 1)
        self.masks, self.costs, self.binary_costs = {}, {}, {}
        self.codec, self._cost_cache = codec, {}
        for level in range(1, self.depth + 1):
            parents, inverse = np.unique(rows // 2, axis=0, return_inverse=True)
            slot = (rows[:, 0] & 1) * 4 + (rows[:, 1] & 1) * 2 + (rows[:, 2] & 1)
            masks = np.zeros(len(parents), dtype=np.int64)
            np.bitwise_or.at(masks, inverse, 1 << slot)
            context = (parents[:, 0] & 1) * 4 + (parents[:, 1] & 1) * 2 + (parents[:, 2] & 1)
            hist = np.bincount(masks, minlength=256).astype(np.float64)
            global_p = (hist + .5) / (hist.sum() + 128.)
            for orientation in range(8):
                counts = np.bincount(masks[context == orientation], minlength=256)
                probability = (counts + 32. * global_p) / (counts.sum() + 32.)
                cost = -np.log2(probability)
                cost[0] = 0.
                self.costs[(level, orientation)] = cost
            self.masks.update({(level,) + tuple(p): int(m)
                               for p, m in zip(parents.tolist(), masks)})
            if codec == 'gpcc':
                neighbour_masks = np.array([[self.masks.get((level, p[0]+d[0], p[1]+d[1], p[2]+d[2]), 0)
                                             for d in self.NEIGHBOURS] for p in parents.tolist()])
                previous = np.zeros(len(masks), dtype=np.int64)
                probability = np.empty((8, 7, 8), dtype=np.float64)
                for slot_index in range(8):
                    neighbours = ((neighbour_masks >> slot_index) & 1).sum(1)
                    condition = neighbours * 8 + previous
                    target = (masks >> slot_index) & 1
                    totals = np.bincount(condition, minlength=56)
                    ones = np.bincount(condition, weights=target, minlength=56)
                    global_bit = (target.sum() + .5) / (len(target) + 1.)
                    probability[slot_index] = ((ones + .5 + 32. * global_bit)
                                               / (totals + 33.)).reshape(7, 8)
                    previous += target
                self.binary_costs[level] = (-np.log2(1. - probability), -np.log2(probability))
            rows = parents

    @staticmethod
    def node(point, level=1):
        return (level,) + tuple(int(v) >> level for v in point)

    def cost(self, key, mask, state=None, patch=None):
        if not mask:
            return 0.
        _, x, y, z = key
        if self.codec == 'gpcc':
            state = self.masks if state is None else state
            patch = {} if patch is None else patch
            neighbours = tuple(patch.get((key[0], x+dx, y+dy, z+dz), state.get((key[0], x+dx, y+dy, z+dz), 0))
                               for dx, dy, dz in self.NEIGHBOURS)
            cache_key = (key[0], mask) + neighbours
            cached = self._cost_cache.get(cache_key)
            if cached is not None:
                return cached
            costs = self.binary_costs[key[0]]
            previous, value = 0, 0.
            for slot in range(8):
                neighbour_count = sum((m >> slot) & 1 for m in neighbours)
                bit = (mask >> slot) & 1
                value += float(costs[bit][slot, neighbour_count, previous])
                previous += bit
            if len(self._cost_cache) < 100000:
                self._cost_cache[cache_key] = value
            return value
        return float(self.costs[(key[0], (x & 1) * 4 + (y & 1) * 2 + (z & 1))][mask])

    def patch(self, candidate, state=None):
        state = self.masks if state is None else state
        changed = {}
        for field, adding in (("remove_coords", False), ("add_coords", True)):
            for point in candidate.get(field, ()):
                point = tuple(int(v) for v in point)
                for level in range(1, self.depth + 1):
                    key = self.node(point, level)
                    child = tuple(v >> (level - 1) for v in point)
                    bit = 1 << ((child[0] & 1) * 4 + (child[1] & 1) * 2 + (child[2] & 1))
                    before = changed.get(key, state.get(key, 0))
                    after = before | bit if adding else before & ~bit
                    changed[key] = after
                    if bool(before) == bool(after):
                        break
        changed = {k: v for k, v in changed.items() if v != state.get(k, 0)}
        affected = set(changed)
        if self.codec == 'gpcc':
            for level, x, y, z in changed:
                affected.update((level, x+dx, y+dy, z+dz) for dx, dy, dz in self.NEIGHBOURS)
        gain = sum(self.cost(k, state.get(k, 0), state)
                   - self.cost(k, changed.get(k, state.get(k, 0)), state, changed)
                   for k in affected)
        return changed, float(gain)

    def dependencies(self, candidate):
        keys = {self.node(p, level) for field in ("remove_coords", "add_coords")
                for p in candidate.get(field, ()) for level in range(1, self.depth + 1)}
        if self.codec == 'gpcc':
            for _ in range(2):
                keys |= {(level, x+dx, y+dy, z+dz) for level, x, y, z in list(keys)
                         for dx, dy, dz in self.NEIGHBOURS}
        return keys


def remember_rate_model(args, signature, rows=None):
    cache = getattr(args, '_geometry_octree_rate_models', None)
    if cache is None:
        cache = OrderedDict()
        args._geometry_octree_rate_models = cache
    if signature not in cache:
        if rows is None:
            return None
        from .geometry_octree import codec_key
        use_binary = getattr(args, 'geometry_octree_rate_model', 'empirical') == 'neighbour_binary'
        cache[signature] = GeometryOctreeRate(rows, 'gpcc' if codec_key(args) == 'gpcc' and use_binary else codec_key(args) + '_categorical')
    cache.move_to_end(signature)
    while len(cache) > 2:
        cache.popitem(last=False)
    return cache[signature]
