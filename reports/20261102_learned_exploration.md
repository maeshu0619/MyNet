# Actual RDから探索幅を学習するTrue経路

2026-11-02。既存の未コミット監査成果物を保持し、開始時のGit状態と全train/modelsのsnapshotを保存した。本変更はSparsePCGCの`test_route=True`が対象。G-PCC / OctAttention、codecソース、既存checkpoint・ログ・学習processは変更していない。

## 目的と変更

前回の監査で、95.5 Episodeで探索が完全停止し、新しいActual結果からActor / Where / Amountへの方策勾配がゼロになることを確認した。今回は序盤に悪い行動も実行する探索を維持し、**探索の広がりも学習対象にした**。損失値、表示曲線、正解Amount、目標圧縮率を補正する処理は追加していない。

実行の流れは次のとおり。

```text
GT構造・Native観測 → NetworkのWhere / Amount分布
 → 学習可能な探索幅で3操作の共同行動をサンプル
 → 実点群編集 → Actual encode・Geometry
 → 同一frameの過去EMAとの差による方策勾配
 → Where / Amount / Actor / Proposalと探索幅を更新
```

Actual方策が使う目的は従来と同じ `3*C + 150*max(G-tau,0)`。CはActualの圧縮変化率[%]、Gは既存Geometry損失。main損失の計算、重み、backward倍率、圧縮器、操作の安全上限・探索候補は維持した。

### 学習する探索幅

既存Actor配下に5要素のparameter `exploration_log_scales` を追加した。Actorの既存optimizer groupへ入り、同じLRで更新される。dataset名やden6の最適割合から値を設定しない。

| 学習対象 | 初期値 | 数値上の範囲 |
|---|---:|---:|
| Native-bit選好係数のNormal標準偏差 | 4 | 0.05～8 |
| WhereのGumbel / Plackett–Luce温度 | 1 | 0.05～2 |
| Amountカテゴリ温度 | 2 | 0.1～4 |
| Fine Amountの対数Normal標準偏差 | 0.3 | 0.02～0.6 |
| Amount一様混合率 | 0.25 | 1e−6～1−1e−6 |

明示的な初期noise=0、一様成分=0はhelperで無効化を維持する。範囲は数値安定性と暴走防止のためで、特定のAmountへ収束させる条件ではない。探索幅は実測結果により増減し、必ず減少する処理はない。初期標準設定の大きな正負の探索を維持し、負のNative選好も禁止しない。

Samplingとlog likelihoodは同じ温度・標準偏差・混合率を使う。Normalの実サンプルはlikelihood評価時にdetachし、温度・分散への勾配も含めたscore-functionを使う。一様成分はカテゴリとFineの混合密度として扱う。

訓練時は旧Episode noise envelopeによる停止を受けず、後半もsampling・Actual方策の更新が続く。evalは従来どおりgreedyで、追加した探索幅を使わず、探索の乱数も引かない。離散binや順位の変化があるため、すべての損失変化が滑らかになる保証はない。

### 教師・勾配

- 従来の「Whereを固定してAmountを試す」試行と、「Whereを探索する」試行を維持した。どちらもAdd / Prune / Adjustは同時に実行する。前者はAmount、後者はWhereへ因果関係のある方策creditを与える。3操作を順番に最適化する方式ではない。
- この2試行は損失分布が違うため、同一GT/CodecのEMAをcredit family別に保持する。各familyの初回は0のcontrol variateを使い、実測した比較教師とは呼ばない。global EMAと過去の実測bestも保存する。
- 過去bestの自己模倣は、現Stepの実測RDより教師が良いときに使用する。現在行動の方が良ければ古い行動へ戻す損失をゼロにし、良い現在行動を保存できる。比較は現在の探索行動に対するものであり、未計測のgreedy行動より教師が良いことを保証するものではない。
- 5つの探索幅の勾配は各座標で既存上限を超えた分だけclipする。長いWhere prefixの勾配が他の探索幅や既存Actorを一括縮小しないように分離した。小さい勾配を増幅したり、normを同じ値に揃えたりしない。
- 加工点群のActual encodeは従来と同じ1回/Step。初回GT encode・既存Validationは別。新しい比較用の追加encodeや入力の間引きは導入していない。

## 設定と互換性

[args.py](/home/maejima/MasterEx/myNet/models/utils/config/args.py:3624) に `--test_route_learned_exploration` を追加した。既定Trueだが、SparsePCGCのTrue経路だけで有効になる。

```bash
# 今回の学習型探索
python train.py --test_route true --test_route_learned_exploration true

# 従来のTrue経路：前半25%でnoiseを終了する比較用
python train.py --test_route true --test_route_learned_exploration false

# 従来のden6経路
python train.py --test_route false
```

新規の探索幅parameterを含む同形式checkpointとAdam stateの保存・復元をテストした。他の形状・設定が互換な旧True checkpointは、欠けた探索幅だけ初期化して警告を出す。**旧checkpointからの再開では探索幅の学習履歴は存在しないため、広い探索から再開する。** greedy評価はこの幅を使用しない。旧optimizer stateへの新parameterの自動追加は実装しておらず、今回の旧checkpoint試験は新しいAdamによるwarm startである。

ログの`AutonomousExploration: mode=actual_learned`と、`AutonomousLearningAudit.learned_exploration`の5つの値を参照する。従来の`ExplorationSchedule`は残るが、新方式の行動noiseを終了させる時計ではない。`exploration=1`はsamplingが有効なことを表し、実際のnoise幅は5つの学習値で決まる。

## 実測と限界

成果物root：`/data/maejima/log/independent_experiments/20261102_learned_exploration_2hxw8h2m`。

環境はmyOA / SparsePCGC用Python、RTX A6000、CPU threads=2。点群自体のdownsamplingは行っていない。True診断は4系列各1フレームの4-frame Episode、Falseは本番と同じ100-frame Episodeの最初の数Step。短期True診断の4-frame設定を本番argsへ反映していない。GT cacheはfingerprintを検証する既存処理を用い、独立出力先へコピーした。MINEのActualは再encodeした。

### 序盤の悪い探索は維持

変更前の初期probeと最終実装の`after_initial_final`で最初の4 StepのActual・Geometry・選択planが一致した。Cは **+97.742950 / +8.041809 / +5.636327 / +0.403956%**。第1 StepのG=0.02321627。大きく悪化する行動を事前排除していない。初期8 Stepの中間実装ではgreedy Validationも+24.390919%だったが、この8 Stepの結果を最終実装の長期評価とは扱わない。

### 学習済み重み、探索終了条件でのActual勾配

同じ既存100.pthとruntimeを読み込み、旧方式ならsamplingが止まるphaseを与えて各5 Step実行した。新方式はその時計から独立している。seed22、同じ重み・データ順、両方とも新しいAdam。旧方式で更新があっても、主に自己模倣等であり、新規Actual方策の勾配とは区別した。

| 既存モジュール | 旧方式：Actual方策L2平均 | 新方式：Actual方策L2平均 | 新方式：非ゼロprobe数 |
|---|---:|---:|---:|
| Add Where | 0 | 2147.753 | 2/5 |
| Prune Where | 0 | 823.309 | 2/5 |
| Adjust Where | 0 | 3509.101 | 2/5 |
| Add Amount | 0 | 4053.384 | 3/5 |
| Prune Amount | 0 | 1030.206 | 3/5 |
| Adjust Amount | 0 | 1656.674 | 3/5 |
| Coarse Amount | 0 | 4.02336 | 3/5 |
| Proposal | 0 | 593.621 | 2/5 |

clip前のsource別L2。2/5と3/5の違いはWhere / Amountへのcreditを分離した試行の違い。新Actorは探索幅parameterも含むので、Actorのnormだけを学習成功の証拠にはせず、既存headとProposalへ流れることを確認した。

新方式でのclip後勾配RMS平均、実更新率`||delta_theta||/||theta||`平均は次のとおり。

| モジュール | clip後RMS | 実更新率 |
|---|---:|---:|
| Add Where | 0.00162952 | 0.00042587 |
| Prune Where | 0.00077918 | 0.00048308 |
| Adjust Where | 0.00248810 | 0.00050355 |
| Add Amount | 0.07557351 | 0.06616510 |
| Prune Amount | 0.04485553 | 0.03749748 |
| Adjust Amount | 0.03981874 | 0.07796164 |
| Coarse Amount | 0.01168668 | 0.00006441 |

既存Fine Amountのparameter normは小さいため、相対更新率だけで他headより大きすぎると判断できない。絶対更新L2も`updates.jsonl`に保存した。上記は短期の動作確認であり、normの増大を圧縮性能の改善とは扱わない。

### Actual圧縮とGeometry

探索なしの固定Validationを4系列各1フレームで測定した。元の長期ログにあったlongdressだけの5フレームとは別の集合であり、それらの平均値と直接比較しない。

| 条件 | Actual C [%] | Geometry G | 固定RD目的 |
|---|---:|---:|---:|
| 旧方式、100.pthから5 Step、seed22 | −5.711310 | 0.00919336 | −15.754927 |
| 新方式、同じ重みから5 Step、seed22 | −5.707324 | 0.00959326 | −15.682981 |
| 新方式、同じ重みから5 Step、seed21 | −5.704642 | 0.00886649 | −15.783953 |

新方式seed22の探索中Cは **−4.4932 / −3.2850 / −5.5766 / +16.6736 / +71.6823%**。悪い探索を続けながら、greedyでは−5%級の行動を出せることをこの短期条件で確認した。

ただし、**新方式による圧縮性能向上・Geometry非劣化は未証明**。seed22ではGが約4.35%悪化し、seed21では平均Gが改善したがCが少し悪化した。全frameでC/Gが両方非劣化という既存判定は、両seedとも不合格だった。新モデルを既存モデルへ昇格させていない。

5 Stepで探索幅が十分に学習されたとも言えない。seed22ではNative std=4.000217、Fine std=0.299896、seed21ではNative std=3.999548、Fine std=0.300048だった。増減をActualから学ぶ経路は機能しているが、**新規重みからの緩やかな低下、382 Episode後の−5到達、長期のGeometry維持は未検証**である。離散選択による段差や、局所的な悪化がなくなることも保証しない。ここでのGeometryは訓練と同じ損失であり、D1/D2 PSNRやBD-rateを測定した結果ではない。

従来の品質gateや固定Validationの選定は今回変更していない。前回監査で見つけた初期no-opのG=0を相対基準にする問題、production Validationがlongdressだけになる問題も残る。これらを含む長期のモデル選択まで解決済みとはしない。

### Falseとテスト

変更前source snapshotと変更後をseed21・100-frame Episodeで各4 Step比較した。初期stateはNaN sentinel bufferのbyteも含め一致。全4 plan、編集点群hash、Native bit数・点数・統計、Actual Cが一致した。

`C = -3.25950694 / -3.97022343 / -3.77741671 / -3.39051461 [%]`

GPU bitstreamのhashは一致しなかった。勾配・更新もbitwise同一ではなく、比較したparameter最大絶対差は約2.98e−5、gradient最大差は約3.48e−6。Falseの完全なbitwise再現を主張しない。Falseの計算分岐・学習対象へ今回の探索parameterは入らず、codec実装も変更していない。

全編集後にも`false_final`を2 Step実行し、変更前の初期state byte、2 plan、Actual C・Native bit数等の一致を再確認した。`false_final_comparison.json`に保存した。

全435テスト合格。Sampler / likelihood整合、温度のscore期待値、探索幅の勾配、既存headの勾配、evalの乱数不使用、FalseのRNGとparameter集合不変、legacy model読み込み、新形式model/Adam state復元、自己模倣条件、clip分離を確認した。toy二乗costでの分散学習テストは数学的な接続確認であり、Actual codec性能とは扱わない。

## ファイルと再現記録

- [autonomous_exploration.py](/home/maejima/MasterEx/myNet/models/utils/training/autonomous_exploration.py)：5つの探索制御と旧model state移行。
- [structure_actuator.py](/home/maejima/MasterEx/myNet/models/modules/structure_actuator.py)：Trueのサンプリング・likelihoodへ学習制御を接続。
- [autonomous_joint_learning.py](/home/maejima/MasterEx/myNet/models/utils/training/autonomous_joint_learning.py)：温度・分散への方策勾配、family別EMA、探索制御のclip。
- [autonomous_actual_replay.py](/home/maejima/MasterEx/myNet/models/utils/training/autonomous_actual_replay.py)：現在の実測RDとの優劣による自己模倣。
- train.pyはTrueの説明ログ追加のみ。args.pyは新設定とhelp。autonomous_audit.pyは5つの学習値を記録。
- `experiments/learned_exploration_probe.py` / `summarize_learned_exploration.py`：独立実訓練・更新観測と集計。各runのinvocation / effective_args / conditions / integrity / source SHAを保存。
- `tests/test_autonomous_learned_exploration.py`：追加検証。全編集後の`regression_verified.txt`で435件合格。
- 成果物rootの`summary.json` / `initial_final_comparison.json` / `validation_non_regression.json`：各runの完了・不完了、初期比較、frame別の非劣化判定。`before/` / `after/`のsource snapshot、`final_source_audit.json`で変更対象を確認できる。意図した既存6 source以外は内容一致。

最初の`before_initial`は独立診断を11更新後に中断した。32 Step完走やValidation成功とは数えない。`before_mature`は4更新が完了したが、独立wrapperがEpisode末の検証入力を取得できず失敗した。`before_mature_v2`を5更新で再実行し、Validationまで完走した。失敗ログ・途中結果は保持し、成功runへ混ぜていない。初期8 Stepの`after_initial`はfamily EMA追加前の中間実装である。

今回確認したのは「序盤の悪い探索を残す」「後半も新しいActualから既存headと探索幅を更新できる」「既存の−5%級greedy行動を短期で失わない」範囲。長期収束・汎化・品質維持を証明するには、初期重みを揃えた複数seedの長期比較、4系列以上の固定Validation、および現在のgreedy行動と過去teacherの同一frame比較が必要である。
