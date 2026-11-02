# 実行中のTrue訓練が15～20秒/Stepになる原因

2026-11-03、JST 01時台。対象は実行中の`train.py --test_route true`（PID 766550）、Native worker PID 767918、最新ログ`/data/maejima/log/20261102/MyNetwork_train/SparsePCGC/221459.txt`。読み取り調査のみで、コード・設定・既存ログ・checkpoint・processは変更していない。追加GPU実験も実行していない。

## 確認できた時間

集計基準を固定するためglobal_step 0～456の457件を保存し、末尾50件を集計した。

| 区間 | 平均 |
|---|---:|
| ログの`step_before_audit` | **18.3301秒** |
| 上記に含まれるActual encode処理 | **8.7065秒** |
| 上記の残り（Network・各loss・backward等） | **9.6237秒** |
| data load後～cleanup完了のCSV区間 | 18.6125秒 |
| 前Stepのcleanup後～次Stepのdata load後（末尾100件、Episode境界除外99件） | **2.5388秒** |

最後の2.54秒は表示Step時間の外側。単純に隣接区間を足すと通常の実処理間隔は約21秒になるが、区間の集計集合は完全同一ではなく、Episode末のValidation・plotはさらに別である。直近50件にGT再encodeはなく、Actualの約8.71秒は毎Step必要な加工点群側の処理である。

## GPUの共有競合

計測時のA6000は7 processで共有されていた。うち本訓練はmainとNativeの2つ。`nvidia-smi pmon -s um -c 20 -d 1`で20秒間記録した。

本訓練以外の2 jobだけで、SM使用率の計測値合計は**平均80.84%、最小56%、最大99%**だった。両processが数値で取得できた19 sampleのうち9件が80%以上。その他のjobもGPUを使用している。したがって、GPU全体100%を本訓練が使えていると解釈できない。

これはSM activityの観測値であり、予約されたGPU容量や、競合だけで何秒遅延したかを表す数値ではない。**実圧縮・Network双方が競合する状態は確認したが、競合による遅延の秒数はGPU専有時との比較なしには確定できない。**

参考として、10/31の独立試験ではActual約0.77秒・Step全体約2.75秒だった。ただし4フレーム反復・診断OFFなど、今回の100フレーム訓練とは条件が違う。以前の改善率や2.75秒を今回の本番時間として保証できない。前回の探索幅学習変更前の独立試験でも、共有負荷下では20秒級のStepが記録されており、5つの新しいparameterだけを今回の長時間化の原因とする根拠はない。

GPU空きは約30 GiB、SM clock 1875 MHz、温度75℃、throttle reason=0。観測時点でVRAM不足による入場待ちやthermal throttlingの証拠はなかった。CPU全体も約8割idle。swapは使用済みだがmain/workerのVmSwapは0、vmstatのsi/soも0であり、swap満杯だけからこの遅延を説明しない。

## コード側に残る処理

「1行動を選ぶ」だけの処理ではない。全点群に対する候補score・選択、実点群編集、Geometry、SparsePCGCの実圧縮、方策勾配の保存用backward、mainのbackward、shadow蒸留等がある。

`train.py`のActual方策勾配は一度backwardして保存し、その後の全体backward後にdecision勾配を復元する。さらにsource別勾配診断が既定ON、10 Stepごとに追加のbackwardを行う。末尾200件では、診断なし180件の平均18.6589秒に対し、診断あり20件は20.5800秒。Actual部分は8.7564対8.7628秒でほぼ同じだった。

差は約1.92秒/診断Stepで、平均寄与は概ね0.2秒/Step相当。ただしframe・行動条件も違う観測比較であり、厳密な同条件ablationではない。**診断を外すだけで15～20秒が1秒になるとは考えられない。**

現在のcompact auditではmodel / loss / backwardの個別時間やNetwork内部の時間を出力していない。残り9.62秒を個別の関数へ正確に割り振ることは、現在のログだけではできない。ここへ推測の秒数を割り当てない。

## 入力PLY解析も毎回繰り返している

最新ログの`StreamingExactCachePolicy`はdataset RAM cache=0、episode common RAM cache=0。要求worker数4はforkによるteacher/cache複製を避ける既存処理で0へ変更され、DataLoader並列読込も無効になっている。

入力はASCII PLY。`models/utils/data/dataset.py`の`_load_ply_numpy`は各行をPythonでdecode/splitしてlistへ格納し、その後float32へ変換する。別processで、GPUを使わず、既存の関数による読み取りだけを1回ずつ測定した。

| ファイル | 点数 | 読込秒 |
|---|---:|---:|
| longdress 1051 | 765,821 | 2.8075 |
| loot 1000 | 784,142 | 2.0431 |
| redandblack 1451 | 720,568 | 1.8523 |
| soldier 0536 | 1,059,810 | 2.7508 |

実行中processの読込関数へinstrumentationを注入した値ではないが、CSVに見えるStep外約2.54秒と整合する。同じ100フレームを毎Episode使っても、現在は解析済み点群をRAM再利用していない。

## 次の対処順

1. GPUを専有できる条件で、同じframe・設定・Actual頻度の短期比較を行う。GPU競合中の時間だけで関数の遅さを判断しない。他のjobは今回停止・変更していない。
2. Actual呼び出し回数は維持し、NativeのNN・CPU統計・入力転送・待ちを分離する。同時に既存のphase/Network timingをcompact auditへ出せるようにして、残り9.62秒を分解する。これには次回起動時の計測設定または独立wrapperが必要。
3. ASCIIの再解析を、数値一致を確認した解析済みcacheや安全な先読みに置き換える。全点・float32・座標順を維持して検証する。fork workerを単純に有効化する案は採らない。
4. 純粋なsource勾配診断は必要な期間に絞る。次回起動時の`--heuristic_guidance_online_source_grad_probe false`は診断用backwardを省けるが、今回の主要な高速化策ではない。

現環境ではActualだけで平均8.71秒かかるため、その他だけを最適化して1秒/Stepを達成することはできない。

成果物は`/data/maejima/log/independent_experiments/20261103_live_latency_bhhr1lj2/`。`summary.json`、log/CSVの読み取りsnapshot、GPUの生記録、loader実測値、source hashを保存した。調査前後でtrain/modelsの変更は0件。計測後も既存訓練とworkerは継続している。
