"""Past, same-frame control variates for the conditional Where sampler.

Native coefficient z is drawn BEFORE the Where permutation a. A baseline
b(state, z), frozen before this observation, may be subtracted from the
Where score E[(cost-b)*grad log p(a|state,z)] without bias. It must NOT be
subtracted from the Native Normal score, because z is that sampler's action.
"""
import math

import torch


def conditional_where_baseline(debug, previous, args, compression, geometry_penalty,
                               family_baseline, alpha):
    from .autonomous_exploration import learned_exploration_enabled
    empty = dict(enabled=False, available=False, source='family_ema', history=None)
    if (not learned_exploration_enabled(args)
            or not bool(getattr(args, 'test_route_where_conditional_baseline', True))
            or debug.get('joint_credit_family') != 'where_sampled'
            or not torch.is_tensor(debug.get('_joint_where_position_log_prob'))
            or not torch.is_tensor(debug.get('_joint_native_sample_log_prob'))):
        return family_baseline, empty
    coefficients = debug.get('structured_where_coefficients', {})
    values = list(coefficients.values()) if isinstance(coefficients, dict) else []
    if (not values or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values)
            or any(v != values[0] for v in values)):
        return family_baseline, empty
    # This is a partition of a sampled conditioning variable, not a desired
    # sign, a codec gain estimate, or an enforced action.
    bucket = 'negative' if values[0] < 0. else 'nonnegative'
    history = dict(previous.get('where_native_condition_ema', {})) if previous else {}
    old = history.get(bucket)
    baseline = (float(old['c']), float(old['g'])) if old is not None else family_baseline
    c, g = float(compression), float(geometry_penalty)
    # Read before update: this observation never labels its own baseline.
    history[bucket] = dict(c=(1.-alpha)*baseline[0]+alpha*c if old is not None else c,
                           g=(1.-alpha)*baseline[1]+alpha*g if old is not None else g)
    return baseline, dict(enabled=True, available=old is not None,
                          source='past_same_frame_native_condition_ema' if old is not None else 'family_ema',
                          bucket=bucket, history=history)
