# Trueルートの追加高速化：厳密格子探索・固定データ再利用

2026-10-28。直前の16候補版を基準に追加で最適化した。`test_route=false` の既存計算、Actual評価頻度、探索範囲、損失重み、100フレーム/Episodeを維持する。

**同条件の8更新で、編集集合hash・操作数・Amount・Actual圧縮損失・幾何損失が8/8一致した。** 通常の再訪3 Stepでは8.3602→5.7543秒、約31.2%短縮。ただし、勾配と更新はbitwise一致ではなく、長期の精度不変を保証する検証ではない。

## 原因と変更

プロファイルStepでは614万回のPython呼出しがあり、主な負荷は以下だった。

| 原因 | 確認した処理 | 採用した変更 |
|---|---|---|
| Geometryで保持点もKD-treeへquery | 整数格子上の全点、約0.7–1.1M点/方向 | 座標一致と26近傍をGPU上で厳密照合。未被覆queryだけ既存KD-treeへ戻す |
| 固定Encoder特徴を繰り返しdisk read | disk hit後にRAMへ登録されず、プロファイルでは約1.89秒 | 元dtypeの生bytesを損失なし圧縮して上限付きRAMへ保持。Adapterは毎回計算 |
| 同じpoolのProxy観測を再計算 | 検証済みmapped cacheから復元した後も、全候補dictを走査。約0.66秒 | 同一input/codec/pool署名・dtypeで検証されたrate/geometry観測を再利用。RD重みは再計算 |
| 合法poolを同じ入力でも毎回再構築 | 例：135,157候補のdict。プロファイルでは約1.54秒 | input内容・座標順序・codec設定・探索phaseをkeyに、poolを損失なし圧縮RAMへ保持 |

これらの秒数はプロファイル中の当該Stepの値であり、全Stepの平均ではない。最終版では呼出し回数、待機、共有ディスク負荷も異なるため、プロファイル値だけで全体速度を予測しない。

### 格子探索が厳密である理由

float32の有限な整数座標で、座標・keyの安全範囲を満たす場合だけ適用する。距離0の照合後、各queryの26近傍をすべて照合する。その範囲の非ゼロ距離二乗は1・2・3。範囲外の整数格子点は少なくとも距離二乗4なので、範囲内に候補があれば、その最小距離は全体の最小距離である。

同距離の場合は元referenceの最小indexを選択し、重複座標もこの規則を守る。選んだquery/referenceを**元のCUDA Chamfer kernel**へ渡して距離とbackwardを計算する。保持点の距離・座標微分はともに0。

26近傍に点がないqueryは従来の厳密KD-treeへ渡す。非整数・過大な座標・未対応形状は従来経路へ戻す。点の間引き、近似NN、geometry量子化、loss補正は行っていない。実験の8更新は両方向すべて格子探索で解け、KD-treeへ渡したquery数は0だった。遠距離fallbackは単体試験で検証した。

### 固定特徴とpoolの再利用条件

固定特徴はEncoderの出力とfull-to-coarse indexだけを保存し、元dtype・全bitsを維持する。学習対象のAdapter、Gate、Where、Amount、Actor、Criticの出力・勾配をcacheしない。Encoderのparameter/buffer versionが変わった場合はRAM cacheとEncoder fingerprintを無効化する。既存clear処理でも解放する。

poolは学習可能なscoreを含まない合法候補データ。入力file identityだけでなく、実際の座標値と**行順序**をhashする。同じ点数でも並べ替えや座標変更はmissになる。探索phaseが変われば古いpool cacheを解放する。評価時はpool cacheを使わず従来生成を行う。新しいscore計算・Amount/Whereの乱数samplingは毎Step実行する。

上限は既定で固定特徴128 MiB・128件、pool 128 MiB・128件。満杯時は新規登録を抑え、循環データでの無駄なcache入替を避ける。全100フレームが必ず収まるとは仮定しない。今回の4フレームではcompressed payloadが固定特徴1,266,701 bytes、pool 11,502,128 bytes、合計約12.18 MiBだった。これはpayload量でありprocess RSSそのものではない。

`args.py` のTrue専用設定：

```text
--test_route_frozen_cache_memory_mb 128
--test_route_frozen_cache_max_entries 128
--test_route_pool_cache_memory_mb 128
```

各memory設定を0にすると該当するRAM再利用を無効にする。Amountの16候補、Fine、3操作の同時学習、探索の減衰、poolの更新時期は維持した。

## 実際の訓練比較

seed22、同一初期state_dict、同一4フレーム、16候補、FP32、Native SparsePCGC CUDA teacherで、それぞれ8更新を行った。既存訓練processには接続せず、cache・registry・checkpoint・logは私有ディレクトリを指定した。本訓練の100フレーム/Episodeは変更していない。

最初の4更新はworker起動・GT初回encode・disk cache読込みを含む。Step 5は両runでcProfileを有効にしたため、通常速度の比較から両方とも除外する。次表は**同じStep 6–8の3更新**の平均。Step 5以外の遅いサンプルを選別していない。

| 計測対象 | 変更前16候補版 | 最終版 |
|---|---:|---:|
| 表示Step時間 | 8.3602秒 | **5.7543秒** |
| Geometry | 1.610843秒 | **0.012445秒** |
| pool生成/復元 | 1.483499秒 | 0.955652秒 |
| Network forward全体 | 3.017013秒 | 2.117647秒 |
| Actual encode | 2.674161秒 | 2.709747秒 |
| main等のbackward | 0.235246秒 | 0.211155秒 |
| 全8更新のTorch peak allocated | 11,680.995 MiB | 11,681.809 MiB |

pool処理はNetwork forwardに含まれる。表の各列を単純加算しない。Actual encoderは変更しておらず、その平均時間もほぼ同じ。GPUピークは約0.81 MiB増で、実質的に同程度だった。この値はモデルprocessのTorch allocatorであり、別processのcodecや共有GPU全体とは異なる。

最終版Step 6–8の実時間は5.4429 / 4.6304 / 7.1897秒。初回4更新は20.5587 / 11.8363 / 15.2477 / 17.2947秒。cache miss時も即座に定常時間になるとは報告しない。

中間版では共有ホストの利用可能RAMが約6.3 GiBとなり、安全閾値8 GiBを待つ `SparsePCGCHostAdmissionWait` が発生し、Step 5が109.694秒になった。閾値を下げたり他processを停止したりしていない。ディスク待ち・host/GPU容量待ちを、純粋な計算時間と区別する必要がある。最終版ではそのStep 5は5.7869秒だった。

## 損失・編集・勾配・更新

8更新すべてで以下が一致した。

- 初期state_dict。両方に存在する同じNaN sentinel bufferを同一と扱う。
- Add / Prune / Adjustの選択数、選択Amount、pool件数。
- 同一GTに対するremove/add集合のhash。
- Actual圧縮損失CとGeometry Gの丸め前の値。

例：Step 1のC=11.7634763718%、G=0.05678662658。Step 6のC=1.8340797424%、G=0.004719167482。いずれも前後同値であり、最適化を圧縮率そのものの改善とは報告しない。

以下は8更新中の最大値。clip後の勾配と、optimizer.step前後の実parameter差を計測した。各列の最大値は同じStepとは限らない。

| module | clip後L2 前→後 | clip後RMS 前→後 | Δθ L2 前→後 |
|---|---:|---:|---:|
| Add Where | 0.863843→0.862601 | 0.00655424→0.00654482 | 0.00610878→0.00610874 |
| Prune Where | 0.966995→0.966652 | 0.01089818→0.01089432 | 0.00566024→0.00566067 |
| Adjust Where | 0.973183→0.973183 | 0.00738384→0.00738384 | 0.00924228→0.00924199 |
| Add Amount | 0.860271→0.860271 | 0.11394567→0.11394567 | 0.00006560→0.00006560 |
| Prune Amount | 0.892006→0.892006 | 0.11814914→0.11814914 | 0.00006565→0.00006565 |
| Adjust Amount | 0.705571→0.705571 | 0.09345510→0.09345510 | 0.00006557→0.00006557 |
| Feature Adapter | 0.000007457→0.000007501 | 1.54025e−7→1.54924e−7 | 0.00123016→0.00123027 |

全勾配は有限。全clip後勾配vectorの前後cosineは最小0.999731、相対L2差は最大約2.32%。state要素の最大絶対差は約0.000253。**勾配・更新まで完全一致とは主張しない。** CUDAの非決定性・加算順序・学習中の微差の増幅などの寄与は完全分離していない。損失と編集集合が一致した8更新を超える長期挙動、D1/D2 PSNR・BD-rate・独立Validationの性能不変は未検証。

## 回帰と保全

235テスト合格。元CUDAとの距離/index・gradient比較、fractional fallback、遠距離query、重複/tie、境界、損失なしfloat32/float16/bfloat16/int32保存、NaN・negative zeroのbits、cache上限、返却値変更の隔離、重み変更時の無効化、Adapterへのgradient、poolの座標順序/探索phase/評価時の扱い、RNGを検証した。

Falseの全既存1,380 args、および8i/MVUB/UVGの9 pool・RNGが作業開始時と一致。FalseのNative Actual訓練を本ターンで再実行したわけではない。

`train.py`、Actual encoder/workerは作業開始時とSHA256一致。損失重み、clip、optimizer、checkpointのparameter構造は変更していない。既存の未コミット変更を保持し、既存checkpoint・既存log・既存訓練process・G-PCC/OctAttention/native codecソースには書き込んでいない。

## ファイルと再現資料

主な実装：

- `models/utils/loss/autonomous_lattice_geometry.py`：厳密な整数格子NNと元kernelへの接続。
- `models/utils/loss/autonomous_exact_geometry.py`：対応条件で新経路へ分岐。
- `models/utils/cache/lossless_tensor_ram.py`：損失なし固定Tensor/pool RAM storage。
- `models/network.py`：Trueの固定Encoder特徴だけを再利用。
- `models/modules/heuristic_guidance.py`：検証済みcacheの固定Proxy観測を再利用。
- `models/utils/pointcloud/autonomous_proposals.py`：Trueの同一pool再利用と無効化。
- `models/utils/config/args.py`：3つのTrue専用cache設定。

成果物：`/data/maejima/log/independent_experiments/20261028_exact_speed_p_gs5kbl/`

- `before/` / `final_source/`：source内容とSHA256。最終119ファイルを構文解析済み。
- `before16/` / `final16/`：前後8更新、同一seed・データ、重み・勾配・timing・cache統計。
- `after16/`：pool再利用を加える前の中間版。host admission待ちも記録。
- `final_comparison.json`：全8更新の数値比較。`filtered_release_summary.json`：Step 6–8集計とmodule更新量。
- `gradient_difference_audit.json`：cosine、gradient差が大きいparameter。
- `tests_release.txt` / `false_isolation_final.json`：235テスト、False保全。
- `incremental_source_changes.patch` / `git_status_before.txt`：今回の差分と作業前状態。

`experiments/autonomous_speed_probe.py` にcProfileとcache観測を追加した。診断専用のread-only tensor-cache fallbackは既存cacheを読むだけで、新規書込みは私有cacheへ行う。`experiments/summarize_autonomous_speed.py` で丸め前の値を集計する。

毎StepのNative Actualを維持する現在の条件では、Actualだけで約2.7秒が残る。1秒未満は達成していない。今回の変更は次回起動から反映され、実行中processへhot patchしていない。
