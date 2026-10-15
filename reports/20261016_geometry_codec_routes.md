# G-PCC / OctAttention 学習経路の分離と検証（2026-10-16）

## 現行コードでSparsePCGCに依存していた箇所

| 箇所 | 依存・問題 | 今回の対応 |
|---|---|---|
| args.py | codec名だけ変更してもbackendはsparsepcgc_surrogate。後段の旧設定は非SparsePCGCでActualを無効化 | `--compression`を`--compress`のaliasにし、選択codecのbackendを自動選択。別codecの明示backendは拒否 |
| train_runtimeのfull-cloud context | 無条件にSparsePCGC二段round量子化 | 幾何Octree専用context builderへ分岐 |
| OctreeStructureAnalysis | 全体がSparseTensor専用というわけではない。occupancy・sibling・近傍・局所幾何の40ch計算は通常Octreeでも使えるが、canonical contextとcacheがSparsePCGC前提 | 別ファイルのGeometryOctreeStructureAnalysisで幾何codecの座標・cacheへ分離。共通40ch記述子のみ継承 |
| NetworkのPoint Transformer入力 | Sparse-grid生成時に別の量子化を重ねる可能性。特徴cacheにcodec・qs・dtypeの区別が不足 | canonical幾何ノードから独立のencoder予算縮約を実行。幾何codec専用cache namespace |
| ana_den6_online候補生成 | SparsePCGC GT symbol特徴・Exact pool cache・worker・AE/SR/mの識別 | 幾何Octree専用の合法Add/Prune/Adjust pool生成。SparsePCGC workerやcacheを参照しない |
| Amount selector | SparsePCGC正規化でだけbin数を合わせるため、他codecはselector=8 / action bins=6で停止 | 幾何codecは共通online Amount binから専用tupleを作り、headと実行actionを一致 |
| GT Actual bit cache | fingerprintにcodecがなく、他方式でもSparsePCGCのGT bitを読み得た | codec・qs・入力・encoder/config/model・算術符号化有無で分離。保存先もgeometry_actual_gt_cacheへ分離 |
| 公開出力と幾何損失 | 他codecでは編集前ノードを公開出力に残し、Actualだけ編集後集合になる旧設定 | 幾何・Surrogate・Actualが同じ編集後occupied集合を見る設定を有効化 |
| 固定full-cloud validation | SparsePCGC以外は無条件にスキップ | G-PCC/OctAttentionも同じ固定検証経路へ分岐 |
| Actual encoder | G-PCCとOctAttentionのラッパーは既に存在 | 同梱tmc3、OctAttention checkpointと算術符号器を利用。SparsePCGC encoderは変更しない |

`sparse_tensor.py` はMinkowskiEngineのSparseTensorそのものではなく、torchによる疎グリッド風の特徴準備を含む。今回の幾何codecはSparsePCGC量子化を避ける一方、codec非依存のvoxel予算縮約・整数座標unique等の低水準補助関数は共有する。

## 新しい実行フロー

選択codecの量子化 → canonical通常Octree → 40ch構造記述子＋Point Transformer特徴 → 3操作の合法候補pool → 既存Proposal subset / Actor / Critic → 既存のannealed探索 → 共通Amount / Gate / Fine → 1つの編集後occupied集合 → 選択codecのActual符号化 → 共通Surrogate・Actual方策損失 → optimizer更新。

- G-PCC: 現在のGPCCGeometryEncoderと同じ、絶対座標round後の整数bbox移動。`gpcc_prequantize=False`なら単位グリッド。
- OctAttention: 同梱dataPrepareと同じ、raw minimumを引いてからround。float64への昇格タイミングも既存のin-memory量子化へ合わせた。
- qs>1で複数raw点が1ノードへ統合される場合、構造解析・候補写像の行順をuniqueノードへ合わせる。raw点対応を切り詰める処理は使わない。
- 幾何用候補の局所occupancy gainは明示的にproxyであり、G-PCCやOctAttentionの実bit予測とは呼ばない。最終planの成否は選択codecのActual結果で学習する。
- 操作数上限、1 Step 1 edited Actual、候補別Actual=0、WhereAll同時探索、探索減衰、共通head・損失重みを維持。SparsePCGCの完成planやsymbol featureを他codecへ転用しない。
- OctAttentionは既定で`octattention_actualcode=True`とし、確率NLL推定だけではなく同梱numpyAcの算術符号化stream bitを使う。明示的にFalseを指定した場合は従来の推定bitモードであり、Actual bitstreamとは区別する。
- retained policy-gradient captureはGradScalerの前に実行される。非SparsePCGC旧既定のFP16では実際にWhereがunderflowしたため、この経路はSparsePCGCと同じFP32を既定・必須とした。

## 新規ファイル

- `models/utils/pointcloud/geometry_octree.py`: codec別量子化、context、復元、encoder縮約、cache identity。
- `models/utils/pointcloud/geometry_octree_proposals.py`: 幾何Octree専用の3操作pool。
- `models/modules/geometry_octree_structure.py`: 通常Octree構造分析のadapter。
- `models/utils/config/geometry_codec.py`: 幾何codecの設定正規化。
- `tests/test_geometry_codec_routes.py`: codec分岐・量子化一致・pool合法性・cache分離等の回帰テスト。

既存ファイルの変更は主としてcodec判定、adapter呼び出し、幾何候補sourceの許可、cache/validation入口の分岐に限定した。SparsePCGC側の候補生成・解析式・encoder・損失計算式・既定重みは変更していない。

## 最終版の実訓練

myOA環境で、8iの4系列それぞれの先頭1フレームを2 Episode反復。各codecで8 Step。探索減衰基準は382 Episodeのまま。初期Networkは新規、Surrogate保存先もcodec・runごとに分離。検証用に訓練と重ならないlongdress_vox10_1052.plyを使用した。

| 指標 | G-PCC | OctAttention |
|---|---:|---:|
| 訓練Step / optimizer更新成功 | 8 / 8 | 8 / 8 |
| 同一フレームActual比較成立 | 4 | 4 |
| edited Actual encode / train Step | 1 | 1 |
| 候補別Actual encode / train Step | 0 | 0 |
| 勾配NaN/Inf | 0 | 0 |
| E1平均Actual差分[%] | +3.277571350 | +2.509777158 |
| E2平均Actual差分[%] | +2.756794751 | +2.899644315 |
| E1平均幾何損失 | 2.184874415 | 2.185217679 |
| E2平均幾何損失 | 2.184417844 | 2.185605228 |
| E2固定検証Actual差分[%] | +3.964059197 | +2.975562402 |

Actual差分は100×(編集後bit−未編集bit)/未編集bit。正値は悪化。両方式で学習・比較・更新が可能になったことは確認したが、8 Step時点では圧縮改善に成功していない。値を負に見せるsign変更、baseline差し替え、良いフレームだけの選別は行っていない。

G-PCCは`compress/octree/G-PCC/build/tmc3/tmc3`と既存geometry encoder.cfg、OctAttentionは`compress/octree/OctAttention/modelsave/obj/encoder_epoch_00800093.pth`と同ディレクトリのnumpyAcを実際に呼び出した。OctAttention検証では`--octattention_teacher_device cuda`を明示。無指定時は既存のbalanced設定に従いCPU teacherになる。

Actual比較がない初回と、比較marginを満たした再訪を区別した。Proposalは両codecで再訪Stepに非ゼロ勾配・更新が入った。

| optimizer直前パラメータ勾配L2最大値 | G-PCC | OctAttention |
|---|---:|---:|
| Where合算 | 0.0871602 | 0.0770496 |
| Coarse Amount | 1.000000 | 1.000000 |
| Fine Amount合算 | 1.000000 | 1.000000 |
| Gate | 1.000000 | 1.000000 |
| Actor | 0.0782161 | 0.0681947 |
| Proposal | 0.0659491 | 0.0727206 |
| Candidate Critic | 0.284030 | 0.269222 |
| Plan Critic | 2.29880 | 2.25304 |

勾配値は実optimizer直前の測定で、更新差分も別途全テンソル記録した。Proposalは8 Step中1 StepでActual教師が成立し、他のゼロStepを勾配遮断とは扱わない。

幾何はqs=2の量子化そのものによる歪みも含むfull Chamfer経路であり、SparsePCGC native qs=1のexact-edit差分計算と絶対値を直接比較しない。このfallbackでは追加点plane-fit項が`disabled_without_addition_provenance`となることもログに明示される。幾何の評価対象集合が編集後Actual入力と一致する点は揃えたが、他codecの品質・重み最適性まで同一と主張しない。

## SparsePCGC回帰

同じseed・4入力・初期Surrogateで、変更前後各4 Stepを実行。

| Episode平均 | 変更前 | 変更後 |
|---|---:|---:|
| Loss | -6.023672462 | -6.023672462 |
| Actual圧縮差分[%] | -3.728889942 | -3.728889942 |
| 幾何損失 | 0.03441948071 | 0.03441948071 |

CSV記録精度で一致。133件のテストも成功し、構文・diff whitespaceチェックも成功。全学習期間や別設定でのbitwise一致を保証するものではない。

## 実行方法

`myNet`ディレクトリ、myOA環境で:

```bash
python train.py --compression G-PCC
python train.py --compression OctAttention --octattention_teacher_device cuda
```

従来の`--compress`も使用可能。SparsePCGCは従来どおり無指定または`--compression SparsePCGC`。G-PCC/OctAttentionで現在対応を検証した学習modeは既定の`ana_den6_online`（名称は共通方策システムとの互換用）である。

## 証拠・限界

最終訓練ログ:

- `/tmp/mynet-multicodec-Sh0mBh/gpcc-final8/log/20261016/MyNetwork_train/G-PCC/013506.txt`
- `/tmp/mynet-multicodec-Sh0mBh/octattention-final8/log/20261016/MyNetwork_train/OctAttention/013507.txt`
- 同runの`gradients.jsonl`: 全optimizer対象の勾配・更新量。
- `/tmp/mynet-multicodec-Sh0mBh/{sparse-before,sparse-after}/log/`: SparsePCGC比較。
- `/tmp/mynet-multicodec-Sh0mBh/tests.txt`: 133テスト結果。

途中の`gpcc-v*`、`octattention-v*`等は不整合の特定用試行。特に旧GT cacheで約+300〜400%となった試行は異なるcodecのbaselineを読んだ無効な比較であり、最終性能値には採用していない。

50%前後の改善は未実証であり、今回の短期値はむしろ悪化を示す。既存論文の削減率をこの候補空間・編集上限・データ・歪み条件へそのまま適用する根拠にはならない。方式内で同じcodec設定を用いた長期学習と、品質を揃えたRD評価が別途必要。既存の外部訓練プロセスを停止・再開したり、元checkpointを上書きしたりしていない。
