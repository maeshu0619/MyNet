# den6標準ルートとAmount/Where自律探索ルートの分離

2026-10-28。作業開始時のmyNet commit: `70948429961eaae4f013db993486bf0f3492916b`。

現在の復元済みden6設計を標準に保ち、14:07の編集履歴から自律探索ルートを復元した。**標準は `test_route = False`。自律探索の性能改善を達成した変更ではない。**

## 切替方法

`models/utils/config/args.py` 冒頭の `test_route` を設定する。CLI指定が優先される。

```bash
# 復元済みden6ルート（省略時もこちら）
python train.py --test_route false

# Amount/Whereの自律探索実験ルート
python train.py --test_route true
```

その他の起動オプションは従来どおり指定する。起動ログの `TrainingRoute: test_route=...` でも選択を確認できる。TrueはSparsePCGC / ana_den6_online専用で、他Codec・他guidance modeとの組合せは明示的なエラーにする。過去の `sparsepcgc_autonomous_search` フラグは切替に使わない。

| 項目 | False：現在のden6 | True：履歴からの実験ルート |
|---|---|---|
| 候補生成 | 現在のExact den6 / 設定済みpool | 現在の点群から合法な26近傍候補を空間抽出 |
| Amount | 現在のdataset別priorと既存残差 | No-op＋絶対Amountの5 bin、既存Fine head残差 |
| Where | 現在の順位・Actor混合・Proposal core | Actorによる順位、Heuristic融合・coreなし |
| 操作配分Gate | 現在のpriorを保持 | dataset別配分priorを足さない |
| 構造特徴 | 現在の行順序・cache | neural node行順序へ対応、別cache namespace |
| 探索減衰 | 現在の設定 | 履歴どおり既存の前半25%の減衰を再利用 |
| 評価 | 現在の評価 | noiseなし、pool位相・coverage固定 |
| best保存 | 現在の保存条件 | 実測参照と同一条件の固定Validationで非劣化確認が必要 |

TrueのAmount binは既定上限0.0099に対し、割合で `0 / 0.0002475 / 0.0007425 / 0.00198 / 0.00495 / 0.0099`。百分率では `0 / 0.02475 / 0.07425 / 0.198 / 0.495 / 0.99%`。Fine・Gate・衝突処理後の実操作割合が必ずこの値になるわけではない。bbox・操作上限などの合法性制約も残る。

Trueで参照JSONを指定しない場合は、周期checkpointと `candidate_loss_*.pth` を保存し、未検証モデルを `best.pth` に昇格させない。指定する場合の引数は `--sparsepcgc_autonomous_reference_json`。同一入力SHA・Codec・Geometry定義・全固定frameでActualとGeometryの両方が非劣化であることを要求する。これは履歴にあった採用ガードの復元であり、未観測データでの性能保証ではない。

## 原設計を保持した確認

- 作業開始時の113訓練ソースは、14:07編集前の保存snapshotとSHA256が全て一致していた。現在の復元済みソースを別の `before/` snapshotにも保存した。
- 保存済みpatchとCodex編集履歴を読み、旧実験の運用コード12ファイルを当時のfinal SHA256と一致する状態まで復元。その後、切替を `test_route` に一本化し、候補生成本体を別ファイルへ分離した。
- Falseでは追加設定の上書きを行わない。8i / MVUB / UVGについて、全既存args値が編集前と一致することを確認した。
- Falseのfast pool本体・返却schemaは原形を保持。Trueのpool本体は `autonomous_proposals.py` に独立させた。両ルート×3 dataset×3時点の18比較で、対応する旧poolと時間計測欄以外が一致し、Torch RNGも消費しなかった。
- 両ルートのstate_dictは510項目すべてでキー・形状が一致した。Networkの学習parameter構造、Codec、loss本体、既存checkpointを変更していない。
- datasetあたり100 frame / Episodeという本設定は変更していない。以下の診断だけ明示的に4 frame / Episodeへ縮小した。

## 実訓練・Actual評価

NetworkはGPU、native SparsePCGCはCPUで実行。8iの4訓練frame、非重複の固定Validation 4 frame、seed 21。Actualはdense_lossy / AE=0 / SR=2 / m=8の実encode値。Gは既存訓練Geometry損失であり、復号後D1/D2 PSNRやBD-rateではない。

| 実行 | 更新数 | 固定Validation Actual [%] | Geometry G |
|---|---:|---:|---:|
| 過去のden6実行 | 8 | −3.846976506393 | 0.0324363438413 |
| 今回のFalse | 8 | −3.846976506393 | 0.0324363438413 |
| 過去の自律探索実行 | 8 | +0.652503678798 | 0.00266320840456 |
| 過去の同一自律checkpointを今回のTrueで再評価 | 0 | +0.652503678798 | 0.00266320840456 |
| 今回のTrueを同seedで再訓練 | 8 | +0.642648862731 | 0.00266259256750 |

Falseは8 Step全ての訓練Actual / Gが過去実行と一致し、訓練8件＋初期/最終Validation各4件、計16件の編集後点群hashと操作件数も一致した。さらに編集前ソースを読み込む独立対照を4 Step実行し、そのActual / G・編集結果も今回のFalseの対応Stepと一致した。

Trueは同一の過去checkpointをstrict loadして4 frame再評価し、編集後点群hash・操作件数・Actual・Gが過去版と完全一致した。新規訓練では初期評価と8 StepのActual・操作件数が過去版と一致した一方、途中の一部選択位置・Geometry・最終重みには差が出た。監査runnerは既存の `deterministic=False` であり、再訓練のbit単位の再現保証とはしない。**同一重み・runtimeでのルート再現と、再訓練の結果の完全一致を区別する。**

Trueの短期性能は依然den6を上回っていない。今回の目的は分岐と原設計の保存であり、長期収束や−3.9%への到達を新たに確認したものではない。

## 勾配・更新と保存の確認

TrueのStep 5–8で計測したActual-RD損失群由来L2と、8更新での最大parameter更新L2。Actual-RDはGeometry等も含み、native Codecに直接autogradを通した値ではない。各最大値は同じStepとは限らない。

| モジュール | Actual-RD grad L2最大 | 実更新Δθ L2最大 |
|---|---:|---:|
| Candidate Actor | 0.339315 | 0.000816438 |
| Candidate Critic | 0.146654 | 0.000899867 |
| Proposal | 0.0859956 | 0.00148727 |
| Add Where | 0.0476439 | 0.00336283 |
| Prune Where | 0.139249 | 0.00288572 |
| Adjust Where | 0.197411 | 0.00452305 |
| coarse Amount | 175.894 | 0.000241694 |
| Add Fine Amount | 767.642 | 0.0000655339 |
| Prune Fine Amount | 355.841 | 0.0000655339 |
| Adjust Fine Amount | 1304.88 | 0.0000654522 |

RMS・clip後L2・更新回数は成果物の `true_gradients_summary.json` に保存した。既存clipを変えて数値を増やしたものではない。非ゼロ勾配・更新は確認したが、それ自体を圧縮性能改善の証明とはしない。参照未指定のTrue実訓練で `best.pth` が作られず、周期・候補checkpointが保存されることも確認した。

回帰テスト201件合格。4本の独立実行すべてで実行中の訓練ソース変更0件、保護対象の既存checkpoint SHA256も一致。既存training process・既存ログ・G-PCC / OctAttention / Codecソースを変更していない。

## ファイルと再現記録

- `args.py`：冒頭の既定値、CLI、Trueだけの設定適用。
- `sparsepcgc_autonomous_policy.py`：実験設定・pool位相・行対応・評価固定・採用ガード。
- `autonomous_proposals.py`：True専用の合法候補生成。
- `structure_actuator.py` / `heuristic_guidance.py` / `octree_structure.py` / `network.py`：Trueだけの選択・構造特徴分岐。
- `train.py` / `train_runtime.py` / `checkpointing.py`：ルート表示、True専用の検証・保存分岐。
- `experiments/autonomous_search_probe.py`：私有cache / registry / log / checkpointでの実訓練・snapshot対照実行。
- `experiments/verify_test_route.py`：編集前/履歴snapshotとの設定・pool・RNG比較。
- `tests/test_sparsepcgc_autonomous_search.py`：合法性・分岐・評価固定・checkpoint保護の回帰テスト。

成果物root：`/data/maejima/log/independent_experiments/20261028_test_route_gpd4fmkb/`。
`before/`、`history/`、`history_integrity.json`、`route_parity.json`、`verification.json`、`tests_final.txt`、各runのraw記録を保存した。
