"""Actual-trained exploration widths for the opt-in SparsePCGC policy.

There is no Episode target, preferred Amount, or stored expert action here.
Sampling and its likelihood must use the very same widths. Evaluation still
uses the existing greedy policy and ignores all exploration parameters.
"""
import math
import warnings

import torch

PARAMETER = 'exploration_log_scales'


def learned_exploration_enabled(args):
    from .sparsepcgc_autonomous_policy import autonomous_search_enabled
    return autonomous_search_enabled(args) and bool(getattr(args, 'test_route_learned_exploration', False))


def install_learned_exploration(actor, args):
    if not learned_exploration_enabled(args):
        return
    epsilon = min(max(float(getattr(args, 'test_route_amount_uniform_fraction', .25)), 1e-6), 1.-1e-6)
    initial = torch.tensor([0., 0., 0., 0., math.log(epsilon/(1.-epsilon))])
    actor.register_parameter(PARAMETER, torch.nn.Parameter(initial))

    def load_legacy(module, state, prefix, *unused):
        key = prefix + PARAMETER
        if key not in state:
            state[key] = getattr(module, PARAMETER).detach().clone()
            warnings.warn('Legacy autonomous checkpoint has no learned exploration widths; '
                          'initial broad exploration is used on training resume. '
                          'Greedy evaluation is unchanged.', RuntimeWarning)
    actor._register_load_state_dict_pre_hook(load_legacy, with_module=True)


def exploration_controls(actor, args):
    if not learned_exploration_enabled(args):
        return None
    value = getattr(actor, PARAMETER)
    native_base = max(float(getattr(args, 'test_route_structured_where_noise', 4.)), 0.)
    fine_base = max(float(getattr(args, 'sparsepcgc_amount_exploration_log_sigma', .3)), 0.)
    # Positive numerical floors bound score-function sensitivity; they do not
    # set an operation amount or terminate Actual policy learning.
    return dict(native_std=(native_base*value[0].clamp(-8., 4.).exp()).clamp(.05, 8.) if native_base else value[0]*0.,
        where_temperature=value[1].clamp(-8., 4.).exp().clamp(.05, 2.),
        amount_temperature=(2.*value[2].clamp(-8., 4.).exp()).clamp(.1, 4.),
        fine_std=(fine_base*value[3].clamp(-8., 4.).exp()).clamp(.02, .6) if fine_base else value[3]*0.,
        uniform_fraction=value[4].sigmoid().clamp(1e-6, 1.-1e-6)
            if float(getattr(args, 'test_route_amount_uniform_fraction', .25)) > 0. else value[4]*0.)


def exploration_parameter(model):
    base = model.module if hasattr(model, 'module') else model
    actor = getattr(getattr(base, 'actuator', None), 'den6_candidate_actor', None)
    return getattr(actor, PARAMETER, None)
