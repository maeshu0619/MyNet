"""Compatibility import. Implementation: :mod:`.reference_voxel_plan`.

Keep old imports and module-level monkeypatches connected to the same module.
New code should import the descriptive implementation directly.
"""
import sys as _sys
from . import reference_voxel_plan as _implementation

_sys.modules[__name__] = _implementation
