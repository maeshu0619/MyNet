# Trueルートの追加高速化：Step全体の再計測

2026-10-31。前回の変更を含む現在のコードを出発点とした。開始時に存在した12ファイルのstaged差分を保存し、保持した。reset・commit・既存processの停止は行っていない。

## 結果

**同条件の独立訓練24 Stepで、前処理からcleanupまでの平均は3.801→2.753秒、27.57%短縮した。ログ表示のStep時間も3.728→2.683秒で、約1.05秒短くなった。1秒/Stepは未達である。**

| 測定 | 変更前 | 最終版 |
|---|---:|---:|
| Step全体の平均：data load後～cleanup完了 | 3.801084秒 | 2.753215秒 |
| 同中央値 | 3.704955秒 | 2.660185秒 |
| 既存ログに表示するStep時間の平均 | 3.728489秒 | 2.683063秒 |
| main processの最大CUDA allocated | 11,685.85 MiB | 11,687.20 MiB |

平均・中央値は最初の5 Stepを除いた6～24 Stepの19件。既存Step表示はcleanup前で計時を終えるため、`*_memory_diagnostics.csv`の`step_after_data_load`と`step_after_cleanup`のunix time差も使用した。計算区間を付け替えて高速化に見せたものではない。data load自体とEpisode末のplot等はこの区間の外にある。

条件はseed 22、4系列各1フレームの反復、同じ15 Amount bins、FP32、同じcodec設定・データ順。診断用の全Tensor保存、追加の強制CUDA同期、cProfileは最終の速度比較では外した。実行中のユーザー訓練とGPUを共有したため、測定時刻による負荷変動を含む。本番の100-frame Episode全体や、独立GPU環境でこの改善率を保証するものではない。

## 1. 最新ログと原因の再特定

調査開始時の最新ログは`20261031/MyNetwork_train/SparsePCGC/134351.txt`。直近20 Stepの`step_before_audit`は平均約4.30秒、Actualは約0.77秒。既存のPID 954131とnative worker 954879を継続させたまま調査した。

前回はforward短縮後の損失・診断側に残ったGCを十分に分離できていなかった。今回は親processのStepを次のforwardまで追跡し、native workerの内部にも独立wrapperでprofileを追加した。

主な発見は次の通り。

| 箇所 | 実測した問題 | 対応 |
|---|---|---|
| shadow蒸留の診断指標 | Adjustの`set((source tuple, target tuple))`生成中に約0.7秒の世代2 GC。Step9では蒸留全体0.785秒 | 同じ集合照合を6列int64配列に変更。TrueだけGCをまとめる |
| GTのcanonical hash | 毎Step、同じGTをCPUでunique/sortしていた。Step9で0.244秒 | 全内容・行順のSHA256で照合してcanonical hashを再利用 |
| 静的入力cache | 保存容量不足と分かる前にGPU→CPUコピーし、結局保存しない。約0.100秒 | Trueだけ、clone後と同じbyte数をコピー前に計算してadmission判定 |
| native loader | 入力処理0.2～0.4秒の一部で、int32座標をfloat64へ拡張し、精度1で除算・roundしてintへ戻していた | 数学的に恒等となる条件だけ元と同じinteger dtypeへ変換 |
| 固定候補の座標構造 | 137,850候補の入れ子listがGC対象として残り、大きなheap走査を繰り返した | Trueのnative候補の固定座標だけtupleへ変更 |

nativeの固定poolingは約0.007秒程度であり、今回そこは変更していない。nativeは既にCPU統計とGPU encodeを並列化していた。

## 2. 採用した変更の内容と不変条件

### 診断指標を整数配列で照合

`autonomous_shadow_metrics.adjust_pair_recall`は、source xyzとtarget xyzを6列のint64として保持し、重複を除いた集合の積集合率を求める。近似距離や省略した候補を使わない。巨大なPython tuple/setを作らず、元と同じrecallを返す。

`SinglePlanDistillationLoss.forward`に既定Falseの`compact_metrics`を追加し、NetworkからTrueルートの場合だけ有効化した。変更するのはdetach済み診断指標であり、蒸留の損失式・重み・学習対象は維持した。単体テストでは通常版と配列版の損失、全metric、全parameterの勾配が一致した。中間profileの同条件Step9では蒸留全体0.785→0.103秒だった。

### 内容照合付きcanonical hash cache

元の座標値をint64の連続配列として、shapeと全行の全byteをSHA256へ入力する。filename、pointer、点数だけを信用しない。ヒットした場合だけ、以前求めたcanonical unique/sort後のhashを再利用する。

格納するのはdigestだけで、点群配列自体は保持しない。上限256件とし、100-frame Episodeで複数の座標順が使われても保持しやすくした。入力の変更・並べ替え・view・重複・空集合・上限をテストした。中間profileでは0.244→0.019秒だった。

### 保存されないcacheコピーを実行しない

`static_clone_bytes`は、同じTensor objectのaliasを1回だけ数え、別Tensor viewは別cloneとして数える。元storageの共有だけで数えるとclone後の容量を過小評価するため、その方法は使わない。

Trueだけ、従来と同じ容量・working set判定をコピー前に行う。保存可能な場合は従来のコピー経路を使う。Falseは従来経路を維持する。view/aliasを含む実cloneのbyte数一致と、拒否される場合にcloneを呼ばないことをテストした。中間profileでは約0.100秒のコピーが不要になった。

### native loaderの恒等量子化を省略

既存のTrue専用`--exact-unique-loader`のworker内局所関数だけを拡張した。条件は整数array、要素幅4 byte以下、precision=1、round/floor、offset不要の場合だけ。int32/uint32はfloat64で正確に表せるため、元のfloat64除算・round/floorは値を変えない。返却dtypeも元と同じ`int`にする。

float、int64、非単位precision、offset要求等は元の関数へ戻す。reader、unique後の座標順、SparseTensor構築、codecのNN、算術coder、点数、品質設定は維持した。Falseではこのworker hookを有効にしない。

### 固定座標を読み取り専用tupleにする

Trueのnative候補だけ、`remove_coords`/`add_coords`の入れ子listをtupleへ変換した。候補pool自体はlist、候補はdictのまま。座標値・順序・candidate ID・score・候補数は変えず、JSONはbyte単位で同じになる。

固定座標を読む現行consumerでlistへの追加・書換えやlist限定判定がないことを確認した。数値だけのtupleはGCがuntrackでき、その後candidate dictもuntrackできる。実点群の独立試験では、full collection後のtracked candidate数は137,850→0だった。ただし最初のfull collection自体はtuple版0.375秒、list版0.275秒であり、その1回が速くなったとは主張しない。後続走査の対象を減らすことが狙いである。

この表現変更も含め、独立した最終24 Stepを実行した。mutableな座標を書き換えることを前提とした外部コードのAPI互換まで保証するものではなく、現行Trueの固定候補内部に限定した変更である。

## 3. 計算結果と学習設計の確認

最終24 Stepの比較は以下。

| 項目 | 結果 |
|---|---:|
| 操作数一致 | 24/24 |
| 候補pool数一致 | 24/24 |
| 編集hash等を含むplan一致 | 15/24 |
| Actual圧縮率完全一致 | 18/24 |
| Actual圧縮率の最大差 | 0.0398327 percentage point |
| Actual圧縮率の平均差：後−前 | −0.0031698 percentage point |
| Geometry完全一致 | 19/24 |
| Geometry最大差 | 1.8887222e−6 |

同じ変更前コードのprofile付き/なしの2実行でも、先頭16 StepでActual最大0.0139294 percentage pointの差が出た。最終比較でActualが違う6 Stepは、入力点群hashも異なる。**単に同じ点群のbitstreamだけが違ったとは扱わない。** GPU経路の再実行差を含むため、選択やTensorの完全一致を主張しない。

数学的な損失・dtype・勾配式・探索noise・Amount bins・操作予算・Where候補・Actual回数は変更していない。CPU上の同一fixtureでは、tuple座標も含めて蒸留loss/metric/全勾配が一致した。しかし、長期収束後の精度維持や−4%への到達を今回の短時間試験で保証したものではない。上記の平均差を学習性能改善の証拠にも数えない。

## 4. Falseの実訓練

開始snapshotのコードと最終コードで各4 Stepを実行した。seed 21、通常の連続RNG、productionと同じ100-frame Episodeの先頭longdress 1051～1054、同じ初期weightsを使った。

| Step | Actual圧縮率：前→後 [%] | Geometry：前→後（ログ精度） |
|---:|---:|---:|
| 1 | −3.259507→−3.259507 | 0.0330→0.0330 |
| 2 | −3.970223→−3.970223 | 0.0309→0.0309 |
| 3 | −3.777417→−3.777417 | 0.0345→0.0345 |
| 4 | −3.390515→−3.390515 | 0.0336→0.0336 |

全4 Stepでplan・編集voxel hash・GT/MINE bits・操作数が一致した。Geometryは4桁ログの比較であり、フル精度一致とは主張しない。勾配/Tensorのbit完全一致も主張しない。

Falseのargs（3 dataset、既定/明示False）、9 poolとRNG、12組のnative worker command/fingerprintも開始snapshotと一致した。Full regressionは**423テスト合格**、新規・関連の絞った検証は85テスト合格。

## 5. 不採用案

残るfull GCを毎回、またはGC countに応じてcodec待ちへ移す案も16 Stepずつ実験した。しかし、生きている候補graphを走査するCPU時間が0.4～0.9秒へ増え、十分な追加改善を得なかった。最終コードでは採用せず、`autonomous_pool_ram.py`は作業開始時の内容へ戻した。前回から存在するbounded先読み自体は保持している。

採用案は候補・診断用の不要なPython object graphと転送を減らすものであり、GCを恒久無効化するものではない。点群のdownsample、loss倍率変更、Actual頻度削減、proxyへの置換、精度低下は行っていない。

## 6. 成果物・再現

新規source:

- `models/utils/loss/autonomous_shadow_metrics.py`
- `models/utils/cache/autonomous_static_admission.py`
- `models/utils/cache/autonomous_candidate_layout.py`

既存sourceへの追加は、Networkの上記True分岐、SinglePlanDistillationの既定Falseの診断経路、True専用canonical hash、native loader、native候補構築のみに限定した。`train.py`、`args.py`、`actual_encoder.py`、native worker本体、G-PCC/OctAttention、codecソース、既存checkpointは変更していない。

実験root: `/data/maejima/log/independent_experiments/20261031_step_hotspots_v2_16lmq02k/`。

`summary.json`に全結果、`false_comparison.json`にFalseのTensor/勾配比較、`false_static.json`に設定等の保全確認を保存。`before/`、開始時index/working-tree差分、各runの引数・log・source integrity・native profile・GC記録・中間失敗のテストlogも保持した。

最終速度比較は`light_before`と`light_after_immutable`。beforeは`--source-snapshot before`に加え、childのnative loaderも`--native-source-snapshot before`で旧関数を読み込ませた。最終速度比較ではcProfileを使っていない。profile/GC方針の中間案は最終版の速度値へ混ぜていない。

再集計は`experiments/summarize_step_hotspots.py --root <root> --out <新規JSON>`。既存出力が存在すると停止する。CPU thread数2、既存myOA/sparsepcgc環境、`-B`と`PYTHONDONTWRITEBYTECODE=1`で実行した。

既存訓練processへ変更を注入していないため、最適化は次回起動から有効。Actual encode、Network、backwardの実処理は残り、現在の同等計算で1秒/Stepを達成したとは報告しない。
