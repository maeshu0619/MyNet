# Actual teacherの厳密なCPU処理高速化

2026-10-29。作業開始commit `16bd91f`、開始時のGit状態はclean。

今回の追加最適化では、同一条件の8更新でActual圧縮損失、Geometry、操作数・Amount、編集後voxel集合、native統計がすべて一致した。通常の再訪3 Stepの平均は **5.5310→4.6542秒（15.85%短縮）**。Actual encode呼出し部分は **2.4816→1.6505秒（33.49%短縮）**。1秒未満、長期収束、BD-rateの不変を確認したものではない。

## 原因と採用した方法

「Actual時間」の大部分をGPUのニューラル圧縮そのものとみなすのは誤りだった。native `encoder_multiple.py` は付加統計を別threadで計算し、点の量子化・3次元uniqueに加え、各深さの親を3次元unique/inverseで集計する。native loaderもMinkowski入力構築前に `np.unique(integer_xyz, axis=0)` を行う。これらはGPU符号化と並行する部分があり、個別時間を単純加算できない。

| 対象 | 変更 | 同一性の条件 |
|---|---|---|
| 付加的なpoint/node/single-child統計 | Morton整数keyを1回uniqueし、3-bit右shiftと連続group数で各深さを集計 | 元reader、2段階round/int64、bbox移動、深さを維持。各軸の幅が21 bit以上、非有限値等は元関数へfallback |
| native loaderの重複除去 | 整数XYZを単射の1次元辞書順keyへpackしunique | 元の辞書順、dtype、座標が一致。signed整数Nx3、pack範囲が安全な場合だけ。その他は元NumPy |

loader本体を別実装へ置き換えていない。元関数の同じ `__code__` を使い、コピーしたglobalsのローカル `np` だけをproxyにする。`unique(axis=0)` の該当呼出し以外は元NumPyへ渡す。native readerのint32変換、量子化、sparse_collate、Minkowski構築を保持した。NumPy全体のmonkey patchやcodecソースの編集は行っていない。

変更はTrueのSparsePCGC / ana_den6_online / dense_lossy / skip_decodeに限り、worker起動flagで有効になる。nativeニューラルmodel、算術coder、探索、損失、勾配置換、clip、optimizerには変更がない。未対応APIはinstallしない。

### CPU単独測定

同じ読み込み済み配列を使い、native→高速版と高速版→nativeの両順序で測定した。PLY読込み時間をこの表に含めない。

| 入力 | native統計 2回 | Morton統計 2回 | unique点 / node / single |
|---|---:|---:|---:|
| longdress 1051 | 3.3042 / 2.1331秒 | 0.05776 / 0.03745秒 | 765821 / 286235 / 35692 |
| loot 1000 | 1.6988 / 1.7857秒 | 0.03810 / 0.03999秒 | 784142 / 292285 / 35179 |

全整数統計が一致した。CPU単体の数十倍という比率を、そのまま訓練全体の倍率とは扱わない。

## 同条件の実訓練比較

`paired_before` / `paired_after` は同一seed22・初期state_dict・4診断フレーム・順序・16 Amount候補・CUDA teacherで各8回optimizerを更新した。両方とも現行ソースを使い、新flagだけfalse/falseとtrue/trueに変えた。既存の100フレーム/Episode設定は変更せず、独立ハーネス内だけ4フレームを2巡した。Actual encodeや探索を省略していない。詳細な勾配分解probeは両方で無効、通常backwardと更新は実行し、clip後gradientsとstate_dictを保存した。

共有hostのRAM不足による待機が予備runで発生したため、最終ペアは**両方とも既存設定 `--sparsepcgc_worker_cpu_trim_interval 1`**を指定して比較した。これは私有実験だけの条件であり、本番既定16は変更していない。従って下表はこの共通条件での速度である。

初回4 StepはGT encode、worker起動、cache読込みを含む。Step 5は両方でcProfile有効なので、通常速度比較には同じStep 6–8を使う。遅いStepだけを片側から除外していない。

| 測定 | 最適化なし | 最適化あり |
|---|---:|---:|
| 表示Step 6 | 5.3160秒 | 4.5935秒 |
| 表示Step 7 | 5.6102秒 | 4.1758秒 |
| 表示Step 8 | 5.6668秒 | 5.1934秒 |
| 上記平均 | **5.5310秒** | **4.6542秒** |
| Actual encode平均、Step 6–8 | 2.48156秒 | 1.65052秒 |
| Network forward平均、Step 6–8 | 2.15961秒 | 1.99370秒 |
| pool生成/復元平均、Step 6–8 | 1.02740秒 | 0.86222秒 |
| plan選択平均、Step 6–8 | 0.67135秒 | 0.63186秒 |
| Geometry平均、Step 6–8 | 0.01411秒 | 0.01623秒 |
| 初回・profileを含む全8 Step平均 | 12.27641秒 | 9.47448秒 |
| Torch peak allocated、全8 Step | 11681.809 MiB | 11681.635 MiB |

pool/選択はforward内、Actualはcompression loss内に含まれる。各行を合計しない。pool等は今回変更していないため、その時間差を今回の機構の効果と断定しない。共有GPU/CPU負荷によるばらつきがある。Torch peakはNetwork processだけで、別processのteacherやGPU全体の使用量とは異なる。

### 損失・編集

8/8で初期重み、選択plan、最終編集集合hash、Actual C、Geometry G、native bit/point/node/single/input SHAが一致。

| Step | C（前後同値、%） | G（前後同値） |
|---|---:|---:|
| 1 | 11.763476371765137 | 0.05678662657737732 |
| 2 | 5.651525020599365 | 0.023981116712093353 |
| 3 | 8.84109878540039 | 0.032482560724020004 |
| 4 | 1.949860692024231 | 0.01779257133603096 |
| 5 | 12.703719139099121 | 0.041119419038295746 |
| 6 | 1.8340797424316406 | 0.004719167482107878 |
| 7 | 14.715252876281738 | 0.04713744670152664 |
| 8 | 5.473537445068359 | 0.04715434089303017 |

これは広い探索をする初期重みの速度検証であり、圧縮改善を達成した結果ではない。旧den6の−3.8%への収束や、自律探索の長期性能をこの8 Stepから保証しない。

### 勾配・実更新

以下は8更新中の各指標の最大。各列の最大が同一Stepとは限らない。全勾配は有限。

| module | clip後L2 前→後 | RMS 前→後 | Δθ L2 前→後 |
|---|---:|---:|---:|
| Add Where | 0.863841→0.865597 | 0.00655423→0.00656755 | 0.00610874→0.00610878 |
| Prune Where | 0.966990→0.967004 | 0.01089812→0.01089828 | 0.00566011→0.00566026 |
| Adjust Where | 0.973183→0.973183 | 0.00738384→0.00738384 | 0.00924225→0.00924185 |
| Add Amount | 0.860271→0.860271 | 0.11394567→0.11394567 | 0.00006560→0.00006560 |
| Prune Amount | 0.892006→0.892006 | 0.11814914→0.11814914 | 0.00006565→0.00006565 |
| Adjust Amount | 0.705571→0.705571 | 0.09345510→0.09345510 | 0.00006557→0.00006557 |
| Coarse Amount | 1.000000→1.000000 | 0.01460516→0.01460516 | 0.00027915→0.00027915 |
| Actor | 0.855523→0.855525 | 0.05032489→0.05032501 | 0.00149943→0.00149943 |
| Proposal | 0.724959→0.724959 | 0.04264467→0.04264467 | 0.00149915→0.00149915 |
| Feature Adapter | 7.56528e−6→7.54614e−6 | 1.56259e−7→1.55864e−7 | 0.00123006→0.00123017 |

全clip後gradient vectorの前後cosineは最小0.99998029、相対L2差は最大0.6279%、state要素の最大絶対差は0.0001978915。**backward・更新のbitwise一致ではない。** CUDAの非決定性等を完全には分離していないため、長期精度不変は未検証とする。module別の相対更新率と全Step値は `paired_update_audit.json` に保存した。

## 符号化・復号の追加検証

訓練中の全stream SHAには差があった。これを黙って無視せず、native環境でGTと独立した混合編集点群について、nativeを2回、統計のみ、統計＋loaderの4条件、合計8回実行した。編集fixtureはAdd128/Prune128/Adjust128であり、学習済みplanや圧縮改善例ではない。

未変更native同士でも全stream SHAは異なった。差はbase座標の格納順で、全算術chunkのhash、base座標集合、count側情報、符号長は各入力の4条件で一致した。実ファイルをunpackしてnative decoderに渡し、復号low座標から元SRで最終点群を復元した。encoder latentの使い回しではない。count側情報はnativeの値を実験JSONへ保存したもので、新しい自律codec/containerの実装ではない。

| 入力 | 物理stream bits | 最終D1相当MSE | 対称Chamfer和 | 最大点距離 |
|---|---:|---:|---:|---:|
| GT、4条件同値 | 38272 | 0.2971138164140184 | 0.5884129581194562 | 4.123105625617661 |
| 混合編集、4条件同値 | 38304 | 0.29726398205324744 | 0.5900399701757983 | 5.385164807134504 |

全8例でencode lowとdecode lowの座標集合hashが一致。同じ入力の4条件で最終復号点群hashも一致した。D1相当は両方向MSEの最大、Chamferは両方向平均二乗距離の和。MPEG D2 PSNRやBD-rateを測った結果ではない。物理stream bitsと、訓練の既存logical bits/side情報の定義は区別している。いずれの定義も変更していない。

## 回帰・保全

- **245テスト合格**。全255 leaf mask、負値・重複・端数・2段round・21-bit境界、overflow fallback、辞書順/dtype、loader code同一性、グローバルNumPy非変更、install idempotence、route/mode/skip_decode条件を含む。
- Falseの元1,383引数（既定/明示false、3dataset）と9pool/RNGが一致。Falseの3mode×skip_decode有無のworker起動コマンドも全体一致。False本訓練を今回再実行したものではない。
- 開始時119のtrain/modelsソースに対し、変更はargs・actual_encoder・workerの3ファイル、新規helper2ファイルのみ。現121ファイルのAST検査が通過。`train.py`、Network、loss本体、optimizer/checkpointのparameter構造は変更していない。
- codecソース、既存checkpoint/log、既存訓練process、G-PCC/OctAttentionには書き込んでいない。検証出力は私有ディレクトリ。

## 設定とファイル

`args.py` のTrue専用設定は既定True。

```text
--test_route_native_morton_stats true
--test_route_native_exact_loader true
```

falseにすれば各高速化を無効にできる。既存の起動済みworkerは再起動していない。次回の通常起動で新flagを受け取る。

- `models/utils/loss/autonomous_native_stats.py`: 厳密Morton統計とnative fallback。
- `models/utils/loss/autonomous_native_loader.py`: 元loader codeを維持する辞書順unique。
- `models/utils/loss/actual_encoder.py` / `sparsepcgc_teacher_worker.py`: True限定flagとworker内install。
- `experiments/native_stats_benchmark.py`: nativeとのCPU比較。
- `experiments/native_codec_repeat_decode.py`: 実stream・復号・geometry比較。
- `experiments/autonomous_speed_probe.py`: 既存私有訓練ハーネスへflags、stream監査、共通trim設定を追加。
- `tests/test_autonomous_native_stats.py` / `test_native_loader_exact.py`: 新規10テスト。

成果物root：`/data/maejima/log/independent_experiments/20261028_native_stats_speed_jzcx152l`。

`paired_summary.json`、`paired_update_audit.json`、`decode_verification.json`、`decode_repeat/results.jsonl`、`stats_cpu_warm/results.json`、`false_regression.json`、`false_worker_command.json`、`source_verification.json`、`tests_all_final.txt`、before/final_source manifestを保存した。各runの `integrity.json` はsource変更なし。旧成果物を上書きしていない。

### 再現例

Network環境は `/home/maejima/miniconda3/envs/myOA/bin/python -B`、native環境は `/home/maejima/miniconda3/envs/sparsepcgc/bin/python -B`。共通環境は `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`。実行時の全引数は各runの `invocation.json` と `effective_args.json` を参照する。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /home/maejima/miniconda3/envs/myOA/bin/python -B experiments/autonomous_speed_probe.py \
 --out /tmp/native_speed_new_run --steps 8 --teacher-device cuda \
 --bins 0.0001,0.0003,0.0006,0.001,0.002,0.003,0.005,0.007,0.009,0.01,0.012,0.015,0.02,0.025,0.03 \
 --cpu-profile-step 5 --native-stream-audit \
 --native-morton-stats true --native-exact-loader true --native-worker-trim-interval 1
```

新しい出力先を用いる。この例はcold cacheの新runなので、上表のwarm再訪時間を直接再現するものではない。比較時は両側のseed cache、設定、計測Stepも揃える。

## 残る制限

本番既定trim16の予備run `final16` ではRAM admission待機によりStep 6が126.3235秒、Step 6–8平均45.1885秒になった。統計のみの予備runでもStep 2に124.8229秒が発生した。これらも成果物に残しており、最終ペアと混ぜない。RAM閾値を下げたり他processを停止したりしていない。ホスト不足による待機は今回の整数アルゴリズムだけでは解決しない。

残る定常コストはNetwork forward約2秒（pool復元とplan選択を含む）とActual約1.65秒。大量のPython候補dictのmaterialize、score/選択、native I/O等を次に調べる価値がある。Actualの削減頻度、探索候補数、点数、精度を下げて1秒未満に見せる変更は採用していない。
