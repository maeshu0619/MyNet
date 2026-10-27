# SparsePCGCのden6非依存探索：実装と非採用判定

2026-10-27。

**Amount/Whereをden6の表・順位から切り離す実験経路を実装した。しかし、Actual圧縮率の非劣化要件は未達である。通常設定を自律探索へ切り替えていない。必ず−3.9%へ収束する、と報告できる結果ではない。**

## 1. 再監査で確認した固定化

既存の`ana_den6_online`主経路は以下の制約を持っていた。

- `ana_den6_online.py`のExact候補poolはden6由来のcompact pool。
- fast poolにも`_FAST_PROFILES`というdataset/m別のAmount・操作配分表がある。
- `structure_actuator.py`はpriorのAmountに有界乗法残差を加え、coarse Amount logitsにもpriorを混ぜる。
- Whereの既定はHeuristic順位と有界なActor残差の和。Proposal subsetにもHeuristic coreを保持する。
- Gateもpriorの操作比率を基準にする。

したがって、探索noiseが存在しても、データセット別の基準値や候補品質への依存が残る。単純なnoise倍率増加を解決策にしなかった。

## 2. 追加した設計

`--sparsepcgc_autonomous_search true`で有効になるSparsePCGC専用の実験経路。
モード名・一部属性名は既存のActual Actor/Critic経路との互換性のため`ana_den6_online`/`den6_*`を残しているが、候補生成でden6表・Exact候補JSON・den6順位を参照しない。
既存で対応している8i/MVUB/UVGに同じ生成式を使う。未知データセット全般のローダ対応まで実装したという意味ではない。

```text
GT → canonical Voxel集合
   → 現在の点群から合法な近傍候補を空間的に抽出（順位・dataset別Amount表なし）
   → neural node行と構造特徴の行を一致させる
   → Proposal / Actor / Critic
   → 絶対coarse Amount + 3操作のFine Amount / Gate
   → Add・Prune・Adjustの複合plan、衝突処理・既存の変更量上限
   → 実Codec encode + 既存Geometry損失
   → 同一frame・同一anchorのActual比較 / 方策勾配 / Critic更新
   → 探索なし固定Validation → 非劣化を確認できたモデルだけbestへ昇格
```

### 候補とAmount

- `_FAST_PROFILES`を使わず、各操作のpool容量を総操作上限から決める。旧操作配分にpool容量を縛らない。
- Whereは入力全体を層化して抽出し、既存voxelのPruneと、26近傍の空voxelへのAdd/Adjustを提案する。
- bboxを保護する。全ての連続座標・任意の大域patchを探索する方式ではない。
- poolを2周分保持して比較教師を成立させ、その後に候補の抽出位置・方向を更新する。
- coarse AmountはNo-opと5つの正値。上限0.0099なら、操作率のbinは
  `0, 0.0002475, 0.0007425, 0.00198, 0.00495, 0.0099`。
  百分率では`0, 0.02475, 0.07425, 0.198, 0.495, 0.99%`。
- これは**選択bin**。実操作数はFine、Gate、衝突・安全上限で変わる。
- Where順位、Amount logits、Gate logitsへden6 priorを加えない。
- Fine headは既存の学習可能なhead−frozen reference経路を使う。全datasetで同じ初期化・探索式を使う。
- Actorへの既存Proxy蒸留を自律モードでは止める。既存の局所Proxy/Entropy補助が全て消えるわけではなく、Actual-only学習とは主張しない。
- Networkのparameter数・checkpointのtensor構造は増やしていない。

探索noiseは既存の前半25%の減衰を維持する。評価ではnoiseを切り、pool位相とProposal coverageを固定する。
従来はcoverage巡回カウンタがevalでも進む箇所があったため、自律モードでは評価中に進めないよう修正した。

## 3. 構造特徴の行対応

通常経路の実モデル評価で、構造metadataのvoxel行とneural node行の一致率は4フレームで
`0 / 0 / 0.0000012809 / 0`だった。座標集合は同じでも順序が異なる。

自律モードでは最小full-cloud contextをneural nodeの座標順に揃え、占有構造をその順番から計算する。
既存の構造cacheと混在させないnamespaceを付けた。未対応の点対応Morton metadataを含むcontextを無検証で部分置換しない。

実モデルの自律モード評価では対応率は1に一致するか、FP32平均の丸めで`0.9999999404`だった。
小規模テストではchild slotの行対応を直接検証した。
通常den6経路は比較対象として保持し、今回その行順序を変更していない。

## 4. 実験条件

- 8i：訓練はlongdress 1051、loot 1000、redandblack 1450、soldier 0536。
- 開発用固定Validation：longdress 1151/1152、loot 1100/1101。訓練フレームとは非重複。
- 短時間診断だけ`train_fixed_episode_frames=4`を明示した。通常のデータセットごと100フレーム＝100 Step設定は維持。
- 各フレームは1回/Step。2-viewによるStep数倍増はない。比較メモリは次の訪問へ持ち越す。
- 同じseed・同じデータ順序・同じCodec設定で比較。baselineと自律モードは中立初期化へ変更したheadの初期biasが異なり、完全に同一parameterからの一因子ablationではない。
- 自律初版と行対応修正版のseed 21は初期state hashが同一。
- SparsePCGC：dense_lossy、AE=0/SR=2/m=8、実encode。Native teacherはCPU、NetworkはGPU。
- Cは既存教師APIのActual圧縮率。Gは既存の編集後点群に対する訓練Geometry損失。
  最終復号点群のD1/D2やBD-rateを新たに測定した実験ではない。
- 通常比較2本、自律初版1本、修正版2本、ガード追加後の実訓練1本、計80 Stepのoptimizer更新を実行。
- Codec・既存checkpoint・既存ログへ書き込まず、Surrogate registry、cache、checkpoint、ログを実験専用領域へ分離。

## 5. 固定Validationの実測

全て探索なしの評価。負のCが圧縮改善。

| 方策 | seed | 更新Step | Actual C [%] | Geometry G |
|---|---:|---:|---:|---:|
| 通常den6経路 | 21 | 8 | −3.846976506 | 0.03243634384 |
| 通常den6経路 | 22 | 8 | −3.846976506 | 0.03240326326 |
| 自律初版 | 21 | 8 | +0.740791536 | 0.03114520758 |
| 自律・行対応修正版 | 21 | 24 | +0.339660491 | 0.002648951660 |
| 自律・行対応修正版 | 22 | 24 | +0.625845523 | 0.003123525879 |
| ガード追加後の自律実訓練 | 21 | 8 | +0.652503679 | 0.002663208405 |

修正版の学習前→24 Stepは次の通り。

- seed 21：C `+1.145151824 → +0.339660491`、G `0.003030170163 → 0.002648951660`。
- seed 22：C `+0.977518446 → +0.625845523`、G `0.002733445726 → 0.003123525879`。

seed 21では両方改善したが、seed 22ではGが増加した。**安定な非劣化学習・収束を確認できたとは言えない。**
比較対象の通常経路は8 Step、自律修正版は24 Stepであり、学習長が同じ比較ではない。
24 Stepの不達から382 Episode後の到達不能を証明することもできない。

### 実行された探索

2 seedの24 Stepで上記6種類のcoarse Amount binが全て実選択された。
seed 21ではAdd 58～2150、Prune 33～5862、Adjust 72～1505操作を実行。
seed 22ではNo-opも選択され、各操作0件のStepがある。
フレームの点数が異なるため、この件数範囲をそのまま操作率範囲とは解釈しない。
編集後点群hashの変化と、実Codecに渡した編集後点群・実符号量を記録した。

strict Actual比較はseed 21で11/24、seed 22で12/24。
Where・Amount・Gate・Fineを因果比較するfactorized group方式のため、全headに毎Step非ゼロのActual勾配が出るわけではない。

## 6. 勾配・実更新

修正版seed 21のStep 5/6/7/8/21/22/23/24を計測した最大値。
Actual列は**ActualとGeometry等を含むRD方策損失群**由来であり、Codecへのpathwise微分ではない。
clip後は合成勾配。各最大値が同じStepとは限らない。

| モジュール | Actual-RD grad L2 | RMS | clip後合成grad L2 | 実更新Δθ L2最大 |
|---|---:|---:|---:|---:|
| Candidate Actor | 0.37631 | 0.022136 | 0.37631 | 0.00081526 |
| Candidate Critic | 0.16788 | 0.0093554 | 0.16788 | 0.00090394 |
| Proposal | 0.094855 | 0.0055797 | 0.094855 | 0.0014930 |
| Add Where | 0.13020 | 0.00098787 | 0.13020 | 0.0033510 |
| Prune Where | 0.16533 | 0.0018633 | 0.16533 | 0.0028860 |
| Adjust Where | 0.19067 | 0.0014466 | 0.19067 | 0.0045121 |
| coarse Amount | 175.89 | 2.7680 | 1.0000 | 0.00024169 |
| Add Fine Amount | 767.87 | 101.71 | 0.72518 | 0.000065534 |
| Prune Fine Amount | 355.89 | 47.139 | 0.66912 | 0.000065534 |
| Adjust Fine Amount | 1305.2 | 172.88 | 0.83976 | 0.000065452 |

Where/Amountは24回すべて実更新された。Criticは23回、Proposalは3回。
WhereのActual非ゼロは8 probe中2回、coarse Amountは1回、Fine各headは3回だった。
他Stepの更新には補助損失・Adam momentum等も含まれる。24回更新を24個の新しいActual教師と同一視しない。
初期parameter normが0の更新率は未定義として集計から除き、巨大な架空の相対更新率を報告しない。

勾配断線・完全なAmount固定だけで今回の性能不達を説明することはできない。
勾配が流れることと、必要な圧縮率へ到達することは別である。

## 7. 採用ガード

自律探索を有効にしても、以下が揃わなければ正式`best.pth`へ昇格させない。

1. 比較基準の実測JSONがある。
2. 固定Validationが必要件数分すべて成功している。
3. 同じframe、入力SHA、Codec fingerprint、Geometry定義で比較できる。
4. **各フレームでCとGの両方が非劣化**。平均改善で悪化frameを相殺しない。

基準がない、検証無効・失敗、条件不一致、CまたはGの悪化なら、周期checkpointと`candidate_loss_*.pth`を保存する。
既存の`best.pth`を置き換えない。実験の合成lossが下がったことだけでは昇格させない。
比較基準はden6に限定しない。既存の検証済みNetworkでもよく、探索行動のpriorや蒸留教師としては使わない。

実際の4フレーム比較では、修正版seed 21のCは基準より
`+3.870501 / +4.350979 / +4.224451 / +4.300617` percentage points悪化。
Gは全て改善したが、**ガードは非採用と判定した**。
このガードは測定した固定集合の条件であり、未観測フレームの非劣化保証ではない。

また、従来の固定検証scalarは`C + λG`で、Actor教師の圧縮係数`w_com=3`と異なっていた。
自律モードだけは圧縮係数をActorに合わせて`3C + λG`へ整合した。
通常経路の選択基準は今回変更していない。
従来scalarと新scalarを直接比較して性能改善とは主張しない。採否はraw C/Gの対応比較を使う。

## 8. 再検証と残る問題

- 通常経路の同じcheckpointを最終コードで再評価し、C/G、操作件数、編集後点群hashが修正前と完全一致。
- 自律checkpointを同じ入力で連続2回Actual評価し、C/G・4件の編集後点群hashが一致。
- ガード追加後にも8 Stepの実訓練を完了。未検証モデルで`best.pth`を作らず、候補・周期checkpointを保存。
- 最後の評価counter固定とValidation lifecycleの修正は、回帰テストおよび実checkpoint再評価で確認。
- 関連199テスト合格、構文検査・`train.py --help`成功。通常/自律checkpointのstate keyとtensor shapeが一致することも確認。
- 一部の実験は同時実行した。Step時間は競合を含むため、厳密な計算速度比較には使わない。
- 4つの開発用Validationフレーム、2 seed、最大24 Stepである。独立した大規模test・長期収束・学習済み最終checkpointでの汎化は未検証。

**残る原因の候補**：den6の候補品質を外した一方、新poolの近傍・密度Proxyはnative Codecの符号化確率ではない。
多数の候補編集に対し、教師は基本的に複合plan全体のActual scalarであり、位置の価値を学ぶ情報が疎である。
今回、Candidate Criticの予測と候補別Actual寄与の相関を測定していないので、教師不足を唯一の原因とは断定しない。
短い最適化期間、入力特徴、Proxyとの競合等の寄与も分離できていない。

次に優先するのは、noiseやloss倍率の追加ではなく、**native SOPA等の符号化contextを観測として与え、同一frameの共同編集に対するActual Δbits教師を増やして候補価値精度を検証すること**。
観測された局所NLLを周辺context込みのActual改善と同一視しない。
その経路を完成させた、−3.9%への到達や必然的収束を保証した、とは報告しない。

## 9. ファイルと成果物

- 新規 `models/utils/training/sparsepcgc_autonomous_policy.py`：mode設定・pool phase・行対応・評価coverage・非劣化判定。
- 既存分岐の変更：`structure_actuator.py`、`heuristic_guidance.py`、`octree_structure.py`、`network.py`、`ana_den6_online.py`、`args.py`、`train.py`、`train_runtime.py`、`checkpointing.py`。
- `experiments/autonomous_search_probe.py`：既存の勾配監査runnerを再利用した独立実験。
- `experiments/summarize_autonomous_search.py`：実測集計。保存済み測定値を補正しない。
- `tests/test_sparsepcgc_autonomous_search.py`：独立性、合法性、座標対応、cache更新、評価再現性、best保護等。

資料root：`/data/maejima/log/independent_experiments/20261027_autonomous_rhxhfebe/`

- `snapshot/`：開始前のGit差分・113 source snapshot。
- 各runの`effective_args.json`, `source_sha256.json`, `integrity.json`, `decisions.jsonl`, `teachers.jsonl`, `gradients.jsonl`, `updates.jsonl`, `encodes.jsonl`。
- `analysis_final/summary.json`：表の根拠。
- `default_replay_final/`：通常経路の再評価・実測reference JSON。
- `autonomous_repeat_evaluation/`：再現性確認、frame別のActual/Geometry、`promotion_status.json`。
- `verification.json`：通常出力の同値性、自律出力の繰返し同値性、採用不可の確認。
- `session_changes.patch`, `modified_sources.json`：今回の差分。

標準は`--sparsepcgc_autonomous_search false`。
実験有効化は`--sparsepcgc_autonomous_search true`。
正式bestの候補にするには、同条件の実測referenceを`--sparsepcgc_autonomous_reference_json`へ指定する。
**今回の候補はその性能条件を通過しておらず、標準設定を置き換えることは推奨しない。**
