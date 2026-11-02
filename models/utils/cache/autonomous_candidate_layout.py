"""Read-only native candidate coordinates without persistent GC list graphs."""


def immutable_candidate_coordinates(pools):
    """Preserve every coordinate, ID and order; JSON remains byte-identical.

    The outer candidate pools and their candidate dictionaries remain ordinary
    mutable list/dict objects. Only the fixed coordinate sequences become
    tuples. Once collected, their acyclic scalar tuples and candidate dicts
    can be untracked by CPython's cyclic collector, unlike nested lists.
    """
    for pool in pools.values():
        for candidate in pool:
            for field in ('remove_coords', 'add_coords'):
                coords = candidate.get(field)
                if isinstance(coords, list):
                    candidate[field] = tuple(tuple(row) for row in coords)
