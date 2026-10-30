"""Bound CPU parallelism for the opt-in route, honoring explicit OMP setup."""
import os


def configure_autonomous_cpu(args):
    from .sparsepcgc_autonomous_policy import autonomous_search_enabled
    if not autonomous_search_enabled(args):
        return None
    requested = int(getattr(args,'test_route_cpu_threads',2))
    if requested <= 0:
        return None
    threads = max(int(os.environ.get('OMP_NUM_THREADS',requested)),1)
    import torch
    torch.set_num_threads(threads)
    os.environ.setdefault('OMP_NUM_THREADS',str(threads))
    os.environ.setdefault('MKL_NUM_THREADS',str(threads))
    return threads
