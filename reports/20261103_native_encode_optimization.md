# SparsePCGC Actual圧縮の最適化と実測

2026-11-03。計測前のtrain/models全ソース・Git差分を保存して調査した。実行中の訓練はPID 766550、native workerは767918。両者へコードを注入せず、停止・再起動していない。

**結論：モデル常駐とCPU/GPU並列化は既に実装済みだった。今回採用したのはTrue経路の固定形式binary PLY直接読込で、読込部分を約0.04秒短縮した。15～20秒/Stepを解決した、あるいは1秒/Stepを達成したとは報告できない。**

## 実際の圧縮経路

`actual_encoder.py`の`_lazy_init()`がworkerを1回起動し、workerの`_build_encoder()`がモデルを生成する。以降は同じencoderがrequest loopで`encode_one()`を処理する。dense / SR / AEのcheckpoint読込は初期化時の3回で、実験の各encode中の再読込は全件0回だった。

Trueの既定dense_lossy/skip_decode経路は、加工点群をbinary PLYへ保存→永続worker→入力読込・量子化・重複除去・ME構築→downscale→BasicCoder各階層の8-stage予測→実算術符号化→stream pack→logical rate集計、の順。入力点群のnode/single-child統計は既に`ThreadPoolExecutor(max_workers=1)`でGPU符号化と並列計算している。

階層・8-stageには予測依存があるため、独立な仕事として同時実行するだけでは同じ確率にならない。複数worker化もモデルメモリ・GPU競合を増やし、逐次optimizer更新を保つ通常の1-Step経路の短縮を保証しないため採用しなかった。

## 比較したアプローチ

| 案 | 実測・確認 | 判断 |
|---|---|---|
| モデル再読込を省く | 初期化時3回、各encode追加0回 | 既に実現済み |
| CPU統計とGPU圧縮の並列化 | nativeコードで既に並列 | 既存動作を維持 |
| CPU searchsortedへの置換 | 実際の検索138/138回がCUDA入力でfallback | 不採用 |
| GPU検索key・GT sortのrequest内cache | 138回中88回cache hit、mask一致。7反復のbasic encode中央値は元0.3256秒→試作0.4758秒 | 遅くなるため不採用。totalは読込等の変動で逆転するため採用根拠にしない |
| 最終classifier出力のソート時にME再構築を省く | 同じ出力に対して座標・特徴・sigmoid一致。basic encode中央値0.3256→0.2980秒だが平均0.3342→0.3709秒、total中央値1.1618→1.2850秒 | 一貫した高速化の根拠がなく不採用 |
| 1次元unique keyを逆変換して座標復元 | CPU反復で元入力中央値0.0320→0.3324秒。座標は一致するが遅い | 試作を撤回。元のunique処理を維持 |
| 算術coderの内部自己decode検査を省く | native profileの48回合計約0.008秒 | 利益が小さく不採用、検査を維持 |
| workerが生成したbinary XYZをNumPyで直接読む | 下表。native int32配列と完全一致 | **採用** |

時間は共有環境の観測値。GPU比較は同一モデルを保持し、同一入力、順序を交互にして7反復したが、他process負荷は制御していない。CPU試験も入力別に15反復し、比較方式の実行順を交互にした。

## 採用変更と速度

`models/utils/loss/autonomous_native_loader.py`に固定形式専用readerを追加した。worker自身が書く「little-endian float32のXYZのみ」の完全一致headerとpayload長を確認する。Open3D PointCloud→float64配列の生成を省き、`np.fromfile`から**従来と同じint32切り捨て**を行う。その後の量子化・重複除去・ME構築は従来通り。

別header、属性付き、別形式、不完全payload、非有限値、int32範囲外などは元readerへ戻す。新しい点群cacheは保持しない。点数・順序・精度・Actual頻度・学習損失を変更しない。

| 入力 | 点数 | native読込中央値 | 直接読込中央値 | 差 |
|---|---:|---:|---:|---:|
| 元点群 | 765,821 | 0.038421秒 | 0.002320秒 | −0.036101秒 |
| Add/Prune/Adjustを各128適用した独立fixture | 765,821 | 0.045342秒 | 0.003572秒 | −0.041770秒 |

これは**読込関数単体の時間**であり、Step全体の改善率ではない。微小な4点fixtureでは直接読込のheader検査の方が遅かった。大規模点群向けの変更である。

本変更は既存の`--exact-unique-loader`を通じてのみworkerへ入る。これは`test_route true`かつSparsePCGCのautonomous探索、dense_lossy、skip_decodeの既存guard下でのみ有効。False、decode有効、losslessでは有効にならないことを既存のworker引数テストでも確認した。args/train/Network/worker本体/codecソースは変更していない。

## 正確性と検証の限界

- CPU実測は元点群・編集点群・負座標/小数/重複/int32境界fixtureを対象に、native readerとのdtype・値・行順の完全一致を全件確認。
- 採用readerのnative encode/decode比較は完了した7件で、physical main stream **38,272 bits**、既存のcount side情報8 bitsを含むlogical rate **38,280 bits**、node数286,235、single-child数35,692、復号765,821点、復号点集合SHA256が全て一致。復号はnative main stream decode後、native downscaleから得た既存のcount/AE side情報を使ってupscaleしたもの。
- 未採用方式も含め、67件のencode結果を保存。そのうち46件はnative decodeまで完了。編集fixtureではphysical 38,304 / logical 38,312 bitsおよび復号集合が比較方式間で一致した。ただし採用readerの編集fixtureについては**CPU入力一致まで**で、独立encode/decode完走は未実施。
- 元方式の反復だけでもstream hashが変わるため、圧縮ファイル全byteの一致は主張しない。実測bitsと復号座標集合を別々に検証した。試作sortの確率一致は同じclassifier出力を固定して比較した。
- 採用readerの完了7件のtotal中央値は元1.0828秒（3件）、変更0.9140秒（4件）。しかしCPU/GPU負荷変動、件数の違い、実験中断があるため、**この差を本変更による0.17秒の改善とは判定しない**。
- 5反復×2入力を予定したreader実験中、共有hostのMemAvailableが2 GiB未満へ低下したため、私のprivate PID 1506173だけをSIGINTで停止した。未完了試行は成果へ含めない。追加のtrain前後比較はメモリ競合を増やすため実施していない。Falseの再訓練数値一致、長期の学習性能向上、全データセットでの高速化は未検証。
- 最終テストは**443件合格**。native codec関連53ソースのSHA256一致。作業前snapshotとのproductionソース差は`autonomous_native_loader.py`だけ。既存checkpoint・既存ログを上書きせず、既存main/workerの生存を確認した。

## 長時間化について残る結論

前段のlive調査では18.33秒/StepのうちActualが8.71秒、その他が9.62秒。今回の別process profileでは1回のencode+cleanupが約1.34秒で、その中で入力準備0.51秒、BasicCoder0.67秒、予測forward約0.51秒、membership約0.12秒、sort約0.08秒だった。並列統計や入れ子関数の時間なので全項を足し合わせない。共有GPUには別jobがあり、今回の独立値を本番100フレームの速度として保証できない。

「モデルを毎Step読み直している」「算術coderの自己decodeがほぼ全時間」という原因ではなかった。**現状の制約でStep全体の大幅短縮は未達**。次の計測では同一frame・行動・乱数を固定した専有GPU条件を確保し、native NN/ME kernelとmainのloss/backwardを分ける必要がある。他processを今回停止・変更していない。実行中processへ今回のファイル変更は反映されず、次回の新規起動から有効になる。

## 再現用コード・成果物

- `experiments/native_encode_optimization_probe.py`: 同じnativeモデルの反復encode、段階別時間、checkpoint load回数、任意のnative decode。
- `experiments/native_symbol_preparation.py`: 未採用の検索・ソート・自己検査省略の試作。通常訓練へ接続していない。
- `experiments/native_reader_probe.py`: GPUを使わないnative読込との完全一致・速度比較。
- `tests/test_native_loader_exact.py`: header/payload/値範囲によるfallback、変換・行順の回帰確認を追加。
- `tests/test_native_symbol_preparation.py`: 試作検索/ソートの一致・cache invalidation検査。
- 全成果物: `/data/maejima/log/independent_experiments/20261103_native_encode_4n19ybd9/`。`before/`、`codec_sha.json`、`verification_summary.json`、`final_integrity_and_reader.json`、`reader_cpu_v2/results.json`、`reader_parity_timing/records.jsonl`、`tests_final.txt`など。

CPU再現例（新しい出力先を指定する）:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/maejima/miniconda3/envs/sparsepcgc/bin/python -B \
experiments/native_reader_probe.py \
  --input /data/maejima/log/independent_experiments/20261103_native_encode_4n19ybd9/parity/gt.ply \
  --out /tmp/native_reader_comparison_new
```

Native GPU試験には新規modelのhost/GPU余裕が必要。probeには起動時16,000 MiB、各試行前10,000 MiBのhost admissionを追加した。途中で他processの使用量が増えれば停止する。最初の`.npy`をPLY readerへ渡した試行とCPU probeのimport順の試行は失敗したため実測成果に含めず、修正後の`_v2`等だけを集計した。
