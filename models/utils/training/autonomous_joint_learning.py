"""Compatibility import. Implementation: :mod:`.joint_voxel_policy`.

Keep old imports and module-level monkeypatches connected to the same module.
New code should import the descriptive implementation directly.
"""
import sys as _sys
from . import joint_voxel_policy as _implementation

_sys.modules[__name__] = _implementation
