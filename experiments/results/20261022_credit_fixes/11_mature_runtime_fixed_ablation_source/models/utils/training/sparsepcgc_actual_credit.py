"""SparsePCGC-only matched Actual learning utilities.

Geometry codecs and inference deliberately do not enter this path. A codec
measurement remains a label for an executed complete plan, not for every point.
"""
from torch.utils.data import Dataset
import math


def sparsepcgc_credit_mode(args):
    codec = str(getattr(args, 'compress', getattr(args, 'method_com', '')))
    codec = codec.lower().replace('-', '').replace('_', '')
    return (codec == 'sparsepcgc'
            and str(getattr(args, 'heuristic_guidance_mode', '')).lower() == 'ana_den6_online')


def sparsepcgc_credit_training(args):
    return (sparsepcgc_credit_mode(args)
            and str(getattr(args, 'trainORtest', 'train')).lower() == 'train')


def stable_amount_tanh(value):
    """Same tanh forward, without subtracting 1 - rounded_tanh**2 in backward.

    At raw/scale=-9.2 FP32 tanh rounds to -1 and its standard backward is
    exactly zero. The equivalent sigmoid-tail expression retains the genuine
    derivative (~4e-8). This is not a straight-through gradient or a multiplier.
    Used only on tiny pooled Amount logits, never full point-feature tensors.
    """
    import torch
    return torch.where(value >= 0,
                       1.0 - 2.0 * torch.sigmoid(-2.0 * value),
                       2.0 * torch.sigmoid(2.0 * value) - 1.0)


def matched_frame_repeats(args):
    if not sparsepcgc_credit_training(args):
        return 1
    return max(1, min(int(getattr(args, 'sparsepcgc_matched_frame_repeats', 2)), 2))


class MatchedFrameDataset(Dataset):
    """Present anchor/variant consecutively without modifying base manifests."""
    def __init__(self, dataset, repeats):
        self.dataset = dataset
        self.repeats = int(repeats)
        self.files = [path for path in dataset.files for _ in range(self.repeats)]
        self.all_files = list(getattr(dataset, 'all_files', dataset.files))
        for name in ('active_window_cycle', 'active_window_slot',
                     'active_window_count', 'active_window_signature'):
            if hasattr(dataset, name):
                setattr(self, name, getattr(dataset, name))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.dataset[index // self.repeats]


def matched_sparsepcgc_dataset(dataset, args):
    repeats = matched_frame_repeats(args)
    return MatchedFrameDataset(dataset, repeats) if repeats > 1 else dataset


def matched_geometry_advantage(memory, *, frame_key, anchor_hash, group,
                               reused, rate, geometry, tolerance, clip=.25):
    """Quality credit only for an encoded same-anchor decision contrast.

    Never compare geometry of different frames or send an Amount contrast to
    a frozen Gate/Fine decision. None means no teacher, not an observed zero.
    """
    if group not in {'Amount','Fine','Gate'} or not anchor_hash:
        return None
    if not all(math.isfinite(float(v)) for v in (rate,geometry)):
        return None
    key=(str(frame_key),str(anchor_hash),str(group))
    if not reused:
        memory[key]=(float(rate),float(geometry))
        while len(memory)>512:
            memory.popitem(last=False)
        return None
    previous=memory.pop(key,None)
    if previous is None or float(rate)>previous[0]+max(float(tolerance),0.):
        return None
    advantage=(previous[1]-float(geometry))/max(abs(previous[1]),1e-8)
    bound=max(float(clip),0.)
    return min(max(advantage,-bound),bound) if bound else advantage
