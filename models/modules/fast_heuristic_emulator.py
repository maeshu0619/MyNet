"""den6の判断をPoolなしで近似する高速Emulator。"""

from __future__ import annotations

from .single_plan_student import SinglePlanStudentPolicy


class FastHeuristicEmulator(SinglePlanStudentPolicy):
    """訓練時はExact planを教師とし、推論時は単独で1 planを生成する。

    実装本体はtrain/test共通のSinglePlanStudentPolicyを再利用する。
    これにより座標membership、方向選択、衝突解消を重複実装しない。
    """

