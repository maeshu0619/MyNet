# 名称・読解性の整理と訓練互換性の検証

2026-11-13。編集開始時のHEADは `d2970729913711545c696892774b96c0ea2c16ac`。
開始時のGit差分は空だった。既存の資料・args・学習済みcheckpoint・実行中processは編集していない。

**3 Stepで、実行中ログの損失表示・Actual圧縮率・Geometry・操作数を再現した。
ただし勾配・更新後パラメータ・実ストリームのバイト列の完全一致は確認できていない。**
この制約と、長期訓練を再検証していないことを含めて報告する。

## 1. 何を整理したか

新しい読解入口は [CODE_MAP.md](../CODE_MAP.md)。
ファイル名・関数名・主要ローカル変数・docstring・接続説明を整理した。
計算式、演算順序、乱数の呼び出し、detach境界、分岐条件、既存係数は維持した。

### 実装ファイルとクラス

| 以前 | 現在 | 内容 |
|---|---|---|
| `octree_structure.py`: `OctreeStructureAnalysis` | [voxel_structure_features.py](../models/modules/voxel_structure_features.py): `VoxelStructureFeatureExtractor` | 40chの占有構造観測。SparsePCGC自体のOctree符号化器という誤解を避ける |
| `structure_actuator.py`: `StructureRepairActuator` | [voxel_edit_controller.py](../models/modules/voxel_edit_controller.py): `VoxelEditController` | Where / Amount、候補評価、競合解決、共同編集 |
| `ana_den6_online.py` | [voxel_candidate_guidance.py](../models/utils/pointcloud/voxel_candidate_guidance.py) | 合法候補reservoir、固定観測、既存参照経路との分岐 |
| `ana_den6_reference.py` | [reference_voxel_plan.py](../models/utils/pointcloud/reference_voxel_plan.py) | 従来の参照plan接続 |
| `autonomous_joint_learning.py` | [joint_voxel_policy.py](../models/utils/training/joint_voxel_policy.py) | Amount / Where sampling、Actual共同方策loss、役割別clip |

旧5ファイルは互換用の短いmodule入口として残した。
旧importと新importは同じmodule objectを共有し、cacheやprefetch状態を二重化しない。
旧クラス名・旧関数名もaliasで利用できる。

### 関数とモジュールの見方

主な対応:

| 以前 | 現在 |
|---|---|
| `_build_fast_unordered_teacher` | `build_legal_candidate_reservoir` |
| `_build_exact_den6_residual_plan` | `build_joint_voxel_edit_plan` |
| `_den6_actor_critic_candidate_values` | `_evaluate_voxel_edit_candidates` |
| `_exact_den6_candidate_scores` | `_gather_candidate_head_scores` |
| `_select_den6_factorized_exploration_group` | `_select_policy_credit_group` |
| `_complete_den6_pending_matched_group` | `_complete_pending_matched_comparison` |
| `joint_actual_policy_loss` | `compute_joint_voxel_policy_loss` |
| `_capture_den6_policy_only_gradients` | `capture_joint_policy_gradients` |
| `_restore_den6_policy_only_gradients` | `restore_joint_policy_gradients` |
| `_guard_den6_online_edited_actual_encode` | `_guard_joint_plan_actual_encode` |

他にも候補の座標変更、合法性、競合、実測品質、参照特徴構築を表す補助関数へ改名した。
全対応は監査成果物の `name_map.json` に記録した。

[network.py](../models/network.py)の冒頭・初期化・forwardに接続説明を追加し、
`geometry_encoder`、`voxel_structure_extractor`、`voxel_edit_controller`を
読み取り用propertyとして導入した。Controllerにも `prune_location_head`、
`adjust_direction_head`、`candidate_actor`、`plan_value_estimator`等のpropertyを追加した。
既存Moduleへの参照であり、新しいNNを登録しない。

### 目的関数

[train.py](../train.py)の合成scalarを `L_total` とし、合成場所をコメントで明示した。
[compression_primary_loss.py](../models/utils/training/compression_primary_loss.py)では、
圧縮主項・Geometry forward項・Geometry backward調整項・補助項を説明的な変数名にした。
[joint_voxel_policy.py](../models/utils/training/joint_voxel_policy.py)では
`L_compression_policy`、`L_geometry_policy`、`L_plan_value`、`L_policy_entropy`、
`L_edit_efficiency`、`L_measured_plan_replay`、`L_joint_policy`に整理した。

係数はloss自体と区別するため `compression_loss_weight` / `geometry_loss_weight`等と記述した。
CLIの `w_com` / `cp_lambda_geom`、表示ログの `L` / `L_com` 等の既存形式は維持した。
表示目的と勾配専用方策項は異なるため、その違いもdocstringで明示した。

## 2. 互換性のため残したもの

- `args.py`、引数値 `ana_den6_online`、既存設定の値。
- `encoder` / `structure_analyzer` / `actuator`、`den6_candidate_actor`等の登録名とstate_dictキー。
- debug辞書、CSV、cache schema、runtime状態のキー。
- 他Codec、False経路、独立実験から利用される関数と計算。
- 過去の報告書と既存ログ。過去報告書の行番号は当時のsnapshotのものである。

したがって、文字列 `den6` が完全に消えたわけではない。
成果物との互換契約まで一括置換すると、読解性だけの変更にならなくなるためである。

既存テストの5ファイルは、差し替える関数の名前・patch対象を新名へ更新した。
テストの期待値・勾配の非ゼロ判定・比較条件は弱めていない。
3つの独立実験のhookも、新旧どちらのsnapshotでも実際の呼び出し先を捕捉するよう調整した。

## 3. 未使用コードの扱い

削除したのは `models/encoder/point_trans.py` の未使用import `math` と `gumbel_softmax`だけ。
名前参照を確認し、計算・関数の削除は行わなかった。
現在のTrue経路で使われないことだけを、リポジトリ全体で未使用である証拠とはしていない。

## 4. 検証条件

監査成果物:

`/data/maejima/log/independent_experiments/20261113_readability_refactor_xttnsakf/`

編集前167 Pythonファイル、Git状態、名前対応、各実験の引数、初期・更新後の重み、
最終勾配、各Stepの指標、実streamのbit数・SHA256を保存した。

参照する実行中ログ:

`/data/maejima/log/20261113/MyNetwork_train/SparsePCGC/150727.txt`

既存processを継続したまま、独立したlog・cache・checkpoint・Surrogate registryを持つ
`experiments/autonomous_speed_probe.py`で、実際の `train.main()` を実行した。
訓練本体を模したダミーのforward/backwardではない。
seed21、MVUBの元のフレーム順、1 Episode 100frames、既定の15個の非ゼロAmount bin、
連続した訓練乱数、GPU Actual encoderを使用した。

実測のためtiming/profileを有効にし、重いgradient auditを無効にした。
診断値の捕捉・パラメータ保存・cacheの温冷・共有GPU状況があるため、通常訓練の速度比較には使わない。
出力先だけでなくcache先も実験専用にし、既存の学習processへsignalを送っていない。

主な実行:

- `before`: 編集開始前の実コードを3 Step。
- `before_repeat`: 編集前snapshotを同条件で再度3 Step。名前整理以外の再現性の対照。
- `after_final`: 最終コードを3 Step。編集前再実行と同じtiming hookを捕捉する。
- 途中の `after` / `after_matched_hooks` も保存しているが、最終比較は上記を使用する。

## 5. 実圧縮を含む3 Stepの結果

下表は**実行中ログ・編集前・編集後で同じ**だった値。
Actualは `current_compression_value` の百分率で、`StepLoss`中の複合scalar `L_com` と混同しない。

| Step | Actual圧縮率 [%] | Geometry | Add / Prune / Adjust | StepLossの全体L（表示値） |
|---|---:|---:|---:|---:|
| 1 | -5.193014621734619 | 0.051547255367040634 | 6074 / 383 / 3406 | -7.8334 |
| 2 | -4.990842342376709 | 0.0073324693366885185 | 34 / 1308 / 319 | -13.8727 |
| 3 | +0.7582515478134155 | 0.016971543431282043 | 72 / 1330 / 1590 | +3.9808 |

Actual・Geometry・操作割合はJSONの数値で等値。
`StepLoss` 3行は、圧縮・幾何・補助項の表示全体を実行中ログと文字列比較して一致した。
日時・計算時間・出力先は当然異なる。

`before_repeat`対`after_final`では、3 Stepとも以下も一致した。

- 初期モデルのTensor名・順序・値（初期状態627,324要素）。
- `plan`記録全体（最終Voxel集合hash、Amount bin、割合、操作数、候補pool数）。
- `policy`記録全体。
- Native指標（bit、点数、node、single、入力point hash）。
- 実streamのbit数: **16,496 / 16,592 / 18,064 bits**。

ここでencoderが返した `metrics.bit` は16,504 / 16,600 / 18,072で、
この実験では上記streamより各8 bits大きい。名称整理でこの既存の数え方を変更していない。

## 6. 勾配・更新について一致しなかったもの

GPUでのbit単位一致は未達である。
最終勾配82,108要素の比較結果を以下に示す。
RMSは勾配差そのもののRMSであり、元の勾配RMSではない。

| Step | 編集前再実行→最終編集後: 勾配差max abs | 同RMS | 編集前コード同士の再実行: 勾配差max abs | 更新量L2: 編集前→編集後 |
|---|---:|---:|---:|---:|
| 1 | 2.422603e-7 | 5.135040e-9 | 1.713634e-7 | 0.1029202100 → 0.1029200242 |
| 2 | 1.955032e-5 | 3.117006e-7 | 3.710389e-6 | 0.0888128116 → 0.0888127956 |
| 3 | 9.164214e-7 | 1.487242e-8 | 9.872764e-5 | 0.0795748299 → 0.0795747541 |

更新後パラメータの最大絶対差は約7.284e-5。
初回の最大差は `actuator.subtree_move_source_head.2.weight` にあり、
対応する勾配差maxは2.423e-7だった。
小さな勾配差がAdam更新にどう反映されるかも、bit単位一致と同等とはみなしていない。

実stream SHA256も3 Stepとも不一致だった。
**編集前コード同士の再実行でも**stream SHA256は3 Stepとも不一致で、
3 Step目は最終Voxel集合hashまで異なったが、圧縮率・Geometry・操作数・表示損失は同じだった。
現在の `deterministic=False` 条件では、編集の有無と独立に再実行差があることを確認した。
これだけから全差分の発生源を特定したとは主張しない。

このため結論は「確認した3 Stepの主要指標と最終選択を維持し、計算式の変更はない」であり、
「全勾配・全バイト・長期訓練曲線が完全に同一」ではない。
再現性設定やCodecの計算を変更して見かけの一致を作ることは行っていない。

## 7. 静的・単体・互換性検証

- 変更した本体18ファイルで、旧→新の名前を対応付けたAST比較が一致。
  追加docstring・互換alias・読み取りproperty・未使用import削除を識別したうえで、
  式、枝、演算順序、乱数呼出しが一致することを確認した。
  Pythonの動的な全挙動の形式証明ではない。
- CPUの解析用fixtureで、共同目的関数の値・勾配が編集前と完全一致。
  seed7 / 21 / 321でAmount sampling・Where順序・勾配・乱数状態も完全一致。
  これは解析的な単体検証であり、codec実測と区別する。
- 全体unit test **474件中469件成功・CUDA用5件skip**。
  その後、CUDAを有効にして該当2ファイル **11件すべて成功**し、skipした5件も検証した。
- 新規 `tests/test_readable_voxel_api.py` の4件で、旧新module object共有、
  重複パラメータ登録の不在、旧API、strict checkpoint roundtripを検証。
- False経路: 3datasetの引数、9pool、乱数状態、12 native worker command/fingerprintが一致。
  False経路の追加Actual訓練を実行したという意味ではない。
- `models` / `tests` / `experiments` / `train.py` の420 Pythonファイルをbytecode生成なしで構文確認。
- `git diff --check`成功。
- Native SparsePCGCのコード・既存checkpoint・既存ログを編集していない。
  開始時から存在した2つのtrain processが継続していることを確認した。

## 8. 成果物と再集計

監査ディレクトリ内の主要ファイル:

| ファイル | 内容 |
|---|---|
| `before_source/manifest.json` | 編集前167ファイルのsnapshot manifest |
| `name_map.json` | 旧新の名前対応 |
| `ast_equivalence.json`, `check_ast.py` | AST比較結果と監査用script |
| `cpu_analytic_parity.json` | CPUでの目的関数・sampling・勾配等の一致 |
| `live_first_three.json`, `live_step_loss_lines.json` | 実行中ログから読み取った比較対象 |
| `final_comparison.json` | 初期重み・勾配・更新量・選択・Native評価の比較 |
| `all_tests_final_v3.stdout` | 全体テスト |
| `gpu_speed_tests.stdout`, `gpu_exact_tests.stdout` | CUDAの追加テスト |
| `false_compatibility_final.json` | False経路の互換性 |
| `before/`, `before_repeat/`, `after_final/` | 実圧縮を含む訓練記録・一時checkpoint |

新規 [compare_readability_runs.py](../experiments/compare_readability_runs.py) は
保存済み結果を読むだけで、訓練・Codecを起動せず再集計する。
出力先が既に存在する場合は停止する。

```bash
AUDIT=/data/maejima/log/independent_experiments/20261113_readability_refactor_xttnsakf
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/maejima/miniconda3/envs/myOA/bin/python -B experiments/compare_readability_runs.py \
  --before "$AUDIT/before_repeat" --after "$AUDIT/after_final" \
  --unchanged-repeat "$AUDIT/before" --live-steps "$AUDIT/live_first_three.json" \
  --out "$AUDIT/comparison_new.json"
```

本検証は先頭3 Stepであり、学習済みcheckpointからの長時間再開、382 Episodeの曲線、
圧縮性能の長期収束を再実験したものではない。
今回の変更を性能改善として扱わず、名称と読解性の整理として扱う。
