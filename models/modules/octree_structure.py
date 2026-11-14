"""Compatibility import. Implementation: :mod:`.voxel_structure_features`.

Keep old imports and module-level monkeypatches connected to the same module.
New code should import the descriptive implementation directly.
"""
import sys as _sys
from . import voxel_structure_features as _implementation

_sys.modules[__name__] = _implementation
