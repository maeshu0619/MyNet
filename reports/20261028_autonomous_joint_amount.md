# True専用の同時Amount / Where探索：実装と短時間検証

2026-10-28。標準の `--test_route false` を維持した独立実験ルート。
速度・16候補化の追加検証は `20261028_autonomous_amount_speed.md` を参照。

## 実装した行動空間

- Add / Prune / Adjustが、**同じStepでそれぞれ独立に**No-opまたは絶対Amountを選ぶ。共有total Amountを3操作に正規化して配る以前の方式をTrueで使わない。
- 各操作の上限は既定3%。Fine headが選択binへ連続的な倍率を与える。最大でAdd3%、Prune3%、Adjust3%を同時に要求できる。競合・bbox保護・合法候補数により実施数は減る場合がある。
- 初期Amount、操作配分、Where順位にdataset別den6の事前検証値を与えない。Gateは各操作独立で、3操作間の正規化を行わない。
- Whereは現在の点群から生成した合法な空間候補に対するActor + Proposalの分布。Heuristic上位core、Actorへのrank入力、学習済みProposal shortlistによる先行制限をTrueで使わない。
- 合法候補poolは残っている。全点の任意の連続座標から直接操作を生成する方式ではない。候補の各方向は26近傍から生成し、reservoirを既存探索期間内で更新する。
- AmountのGumbel / FineのNormal / WhereのGumbelは既存探索スケジュールに従い減衰する。探索終了後は候補reservoirの暗黙の回転も停止する。評価では探索を無効にする。

## 損失と更新

Actual codecは非微分であるため、その整数符号量を通るpathwise勾配があるとは主張しない。実行した**完全な共同plan**のActual圧縮値Cと幾何損失Gから、score-function勾配を各操作のAmount / Where、Actor / Proposalへ返す。重みは既存の設定を使い、単に勾配を増やすloss倍率は追加していない。

初回frameのcontrol variateは0。同じ入力hash・codec・幾何条件の再訪時だけ、そのframeのEMAを使う。**初回の0を実encodeした比較baselineと呼ばない。** 別frameの圧縮差をAmountの教師値にしない。

Whereのlikelihoodは、競合で棄却した候補も含む実際に試した順序prefixのPlackett–Luce確率。未訪問suffixは周辺化する。No-op Amountでは使用しないFineの確率も周辺化する。

Criticは共同planのRDを学ぶ。Candidate Qの要約からPlan Criticへ学習信号を流すが、**全planの改善を各候補の実測改善として複製しない**。個別候補のActual寄与を測定した教師ではない。

更新直前の勾配置換によりTrueの意思決定headをActual方策由来に限定し、従来の不完全なglobal clipをTrue専用の重複しないrole別clipへ置き換えた。Where、Actor/Proposal、coarse Amount、Fine Amount、Gate、Critic、backboneを分ける。小さい勾配を目標ノルムへ増幅しない。

## 実際の短時間学習

成果物：
`/data/maejima/log/independent_experiments/20261028_joint_amount_b41rad7f/`

6候補版の `true_joint_final_s21` で8更新を実施した。選択された実操作率の範囲はAdd **0.0824–1.9490%**、Prune **0.1485–3.0000%**、Adjust **0–3.0000%**。別seedの `true_final_smoke_s22` の2更新目ではAddも実際に約3%実行された。理論上の上限だけの確認ではない。

最初のStepで、noiseなしの同一score上位集合から入れ替わった選択候補の割合はAdd **98.734%**、Prune **52.058%**、Adjust **83.386%**。この値はこのStepの観測であり、全期間の保証ではない。

以下は8更新中の最大値。C/GのL2は個別source backward時の生勾配、clip後は実更新に用いた勾配。互いに異なるStepの最大値を一つの勾配分解として足し合わせてはいけない。

| head | Actual圧縮方策L2最大 | 幾何方策L2最大 | clip後L2最大 | Δθ L2最大 |
|---|---:|---:|---:|---:|
| Add Where | 530.234 | 57.985 | 0.8884 | 0.005901 |
| Prune Where | 728.083 | 284.107 | 0.8832 | 0.005915 |
| Adjust Where | 702.519 | 143.628 | 0.7139 | 0.009197 |
| Add Amount | 25578.12 | 7172.09 | 0.9586 | 0.00006568 |
| Prune Amount | 9603.71 | 1618.51 | 0.9592 | 0.00006550 |
| Adjust Amount | 12593.76 | 3531.28 | 0.6000 | 0.00006558 |
| Actor | 485.705 | 136.187 | 0.7674 | 0.001500 |
| Proposal | 496.467 | 139.200 | 0.7607 | 0.001499 |
| Candidate Critic | 0.00001707 | 0 | 0.00001707 | 0.0004879 |

丸め前の値・RMSは `final_gradients.json` と各runの `gradients.jsonl` / `updates.jsonl` に保存。Candidate Criticは最初の更新が0、以降7/8更新が非ゼロ。初期Plan Critic最終層が0であるため初回の下流特徴への微分が0になることと整合する。

## 解決済みとしない性能上の制約

8更新後の4フレーム固定Validationでは**No-opが選ばれ、C=0%、G=0**だった。den6の **C=−3.8469765%、G=0.03243634** を上回っていない。探索・勾配・更新が成立したことと、圧縮性能改善の学習に成功したことを区別する。長期収束、複数seed・独立評価集合での優位性は未検証。

参照Actual / geometry測定がないTrueのcheckpointは候補として保存し、正式bestへ無条件昇格させない既存のguardを維持した。No-opをden6より優れた結果として保存していない。

`false_regression_s21` は修正前Falseと同じ8更新・4フレームValidationを実行し、16回の訓練/評価編集点群hash・操作数が一致。上記C/Gも一致。標準データセット100フレーム/Episodeは維持し、独立診断runだけ4フレームを指定した。既存checkpoint、既存ログ、codecソース、実行中の既存訓練processは操作していない。

## 主要ファイル

- `models/utils/training/autonomous_joint_learning.py`：同時Amount/Whereの確率、共同Actual方策損失、role別clip。
- `models/utils/training/sparsepcgc_autonomous_policy.py`：True設定、検証guard、独立poolのphase。
- `models/utils/pointcloud/autonomous_proposals.py`：dataset別事前Amountに依存しない合法候補。
- `structure_actuator.py` / `network.py` / `train.py` / `train_runtime.py` / `optim_amp.py`：Trueの分岐で接続。
- `experiments/autonomous_search_probe.py`：既存の勾配計測runnerを私有cache・出力先で使用。

この段階の結論は「独立同時探索と更新を実測した」であり、「den6と同等以上の収束性能を保証した」ではない。
