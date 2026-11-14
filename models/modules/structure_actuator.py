"""Compatibility import. Implementation: :mod:`.voxel_edit_controller`.

Keep old imports and module-level monkeypatches connected to the same module.
New code should import the descriptive implementation directly.
"""
import sys as _sys
from . import voxel_edit_controller as _implementation

_sys.modules[__name__] = _implementation
