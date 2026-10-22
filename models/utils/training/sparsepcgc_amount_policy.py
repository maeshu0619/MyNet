"""Small, checkpoint-compatible Amount policy primitives (SparsePCGC only)."""
import torch
import torch.nn.functional as F

from .sparsepcgc_actual_credit import sparsepcgc_credit_mode

OPERATIONS = (('Add', 'add_amount_head'), ('Prune', 'drop_amount_head'),
              ('Adjust', 'move_amount_head'))


def amount_learning_enabled(args):
    return (sparsepcgc_credit_mode(args)
            and bool(getattr(args, 'sparsepcgc_amount_exploration', False))
            and bool(getattr(args, 'repair_learn_operation_amounts', True)))


def calibrated_log_residual(raw, reference_raw, span):
    """True bounded residual, not a surrogate/STE derivative.

    Equal learned/reference heads give zero for every input. At that point
    d residual/d raw=1 even for a legacy raw logit of -55. The reference is a
    frozen checkpoint buffer, NOT the current raw output detached each step.
    """
    span = max(float(span), 1e-6)
    return span * F.softsign((raw - reference_raw) / span)


def register_amount_reference(owner):
    if not sparsepcgc_credit_mode(owner.args):
        return
    owner.register_buffer('sparse_amount_policy_version', torch.tensor(1, dtype=torch.long))
    for operation, name in OPERATIONS:
        head = getattr(owner, name)
        owner.register_buffer('sparse_amount_reference_' + operation.lower() + '_weight', head.weight.detach().clone())
        owner.register_buffer('sparse_amount_reference_' + operation.lower() + '_bias', head.bias.detach().clone())


def migrate_amount_reference(owner, state_dict, prefix):
    """Migrate a legacy state IN MEMORY; never rewrite its checkpoint file."""
    version = prefix + 'sparse_amount_policy_version'
    if not hasattr(owner, 'sparse_amount_policy_version') or version in state_dict:
        return
    state_dict[version] = owner.sparse_amount_policy_version.detach().clone()
    for operation, name in OPERATIONS:
        for part in ('weight', 'bias'):
            target = prefix + 'sparse_amount_reference_' + operation.lower() + '_' + part
            source = prefix + name + '.' + part
            # Missing original heads still trigger normal strict-load errors.
            value = state_dict.get(source, getattr(getattr(owner, name), part))
            state_dict[target] = value.detach().clone()


def reference_raw(owner, operation, pooled):
    weight = getattr(owner, 'sparse_amount_reference_' + operation.lower() + '_weight')
    bias = getattr(owner, 'sparse_amount_reference_' + operation.lower() + '_bias')
    return F.conv1d(pooled, weight.to(pooled), bias.to(pooled))


def sampled_bin_log_probs(logits, gumbel_scale):
    """argmax(logits + scale*Gumbel) samples softmax(logits/scale)."""
    return torch.log_softmax(logits / max(float(gumbel_scale), 1e-6), dim=0)
