"""Exact array metrics for the autonomous route's shadow teacher diagnostics."""
import numpy as np


def adjust_pair_recall(candidates, predicted_sources, predicted_targets):
    """Match the Python set-of-(source,target) recall without tuple graphs.

    Six int64 coordinates identify a pair exactly; no packed-key overflow or
    approximate spatial query is used. Duplicate pairs count once on each side.
    These are detached diagnostics and never participate in the loss graph.
    """
    rows = [row for row in candidates if row.get('remove_coords') and row.get('add_coords')]
    if not rows:
        return None
    expected = np.empty((len(rows), 6), dtype=np.int64)
    for i, row in enumerate(rows):
        expected[i, :3] = row['remove_coords'][0]
        expected[i, 3:] = row['add_coords'][0]
    predicted = np.empty((len(predicted_sources), 6), dtype=np.int64)
    predicted[:, :3] = predicted_sources
    predicted[:, 3:] = predicted_targets
    dtype = np.dtype((np.void, 6 * np.dtype(np.int64).itemsize))
    expected = np.unique(expected.view(dtype).reshape(-1))
    predicted = np.unique(predicted.view(dtype).reshape(-1))
    return float(len(np.intersect1d(expected, predicted, assume_unique=True)) / float(len(expected)))
