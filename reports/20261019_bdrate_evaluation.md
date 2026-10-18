# 実bitstream・最終復号点群に基づくBD-rate評価システム

2026-10-19（JST）。実装・検証結果。**実装の動作確認と、提案手法の研究上の性能評価は区別する。**

## 1. 追加したファイル／既存機能の保護

- `test_BDrate.py`：設定・CLI。冒頭にmodel日時、checkpoint、評価ファイル、codec、RD設定、resolution、出力先を集約。
- `utils/bdrate/io.py`：既存binary PLY writer再利用、実byte計数、排他的ファイル生成、SHA256、subprocess記録。
- `utils/bdrate/codecs.py`：既存G-PCC／Draco command builderの再利用、native encode／decode。
- `utils/bdrate/learned_worker.py`：OA／SparsePCGCの独立codec worker。実算術符号化、復号、付加情報の物理保存。
- `utils/bdrate/inference_worker.py`：既存`test.py`のmodel loader、前処理、full-cloud推論、post-hardening、保存を再利用。
- `utils/bdrate/metrics.py`：MPEG `pc_error_d`の直接実行・結果検査。GT法線の検査と互換保存形式。
- `utils/bdrate/curves.py`：Akima補間、共通区間の解析積分、異常検出、密／疎測定の比較。
- `utils/bdrate/runner.py`：点群・RD点ごとの逐次処理、失敗記録、provenance、保護対象hash確認。
- `utils/bdrate/report.py`：Excel、JSON、RD曲線PNG、完全なdataset平均／有効分のみの平均の区別。
- `utils/bdrate/README.md`／`requirements.txt`：仕様・再現条件・評価環境依存。
- `tests/test_bdrate_evaluation.py`：28回帰テスト。
- `tests/bdrate_native_smoke.py`：解析法線付き合成曲面での実codec検証。
- `tests/bdrate_reference_validation.py`：外部公開BD-rate実装との100ケース比較。

`train.py`、`test.py`、`args.py`、Network、loss、既存codecソース、checkpointは編集していない。既存訓練processの停止・再起動もしていない。既存trackedファイルの差分は0。

実行時は学習関連Pythonソース、codec側Pythonソース、使用binary・設定・codec checkpoint・固定encoder checkpointのhashを開始／終了時に比較する。評価器自身のソースと指定checkpointの途中変更も検出する。途中変更時は集計値を無効化する。既存の出力ディレクトリには上書きせず停止する。

### 既存コードの再利用判断

`utils/compress/evaluation.py`は独自法線推定と独自PSNRを含む。`D1PSNR.py`／`D2PSNR.py`は`peak²/MSE`であり、そのままMPEGの`3*resolution²/MSE`と同一視できないため正式評価には使わない。これらの既存関数は変更していない。

訓練用teacher wrapperは推定rate、原点除去、外部量子化、復号skip、一時ファイル削除などを持つため、その返り値をBD-rateの実測値として使わない。再利用するのは検査済みのnative command builder／codec演算である。

## 2. モデル・データの指定と実行

`test_BDrate.py`冒頭の例は`model_date="20260801"`、`model_time="072958"`、`model_name="best_loss_joint"`。checkpoint未指定時は既存argsの`_pretrained_checkpoint_path`で解決する。

`compression`はRD曲線を取得するcodec、`network_compression`は学習時のNetwork構成であり区別する。例えばSparsePCGC向けに学習したNetworkを変更せずG-PCCで評価する場合、前者をG-PCC、後者をSparsePCGCにする。同じモデルのMINE出力を各RD点で再利用する。各rateに合わせた再学習やNetworkの暗黙の設定変更はしない。

例（実在する評価用GT・対応法線・互換checkpointに置き換える）：

```bash
cd /home/maejima/MasterEx/myNet
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/maejima/miniconda3/envs/myOA/bin/python -B test_BDrate.py \
  --codec G-PCC --network-codec SparsePCGC --network-dataset 8i \
  --input /path/to/holdout/loot.ply \
  --normals /path/to/holdout/loot_normals.ply \
  --resolution 1023 --checkpoint /path/to/compatible_model.pth
```

GT自体に正しい法線があれば`--normals`は省略できる。法線がない場合はD2を実行せず失敗理由を保存する。D1だけを意図して測る場合に限り`--d1-only`を明示する。法線やresolutionをbbox・PCA等から暗黙に作らない。

`--`以降は既存test.pyの引数を渡せる。例：CPU診断なら`-- --cpu --allow_slow_knn_fallback true`。`--cpu`は値を取らない既存flagである。checkpoint／入出力／codec／探索／AMP等の保護設定は転送引数から上書きできない。

評価対象は明示したファイルだけ。既存Datasetが要求する`<dataset>/<sequence>`の形で一時symlinkを用意するが、sampling/cacheのキーはOriginal絶対パスへ解決して既存関数に渡す。実行ディレクトリのUUIDにより入力samplingが変わることを防ぐ。データ漏洩の最終確認には、指定ファイルと実際の学習splitの照合が別途必要である。

## 3. Codec別RD設定

| Codec | 変更する実在パラメータ | 既定6点 |
|---|---|---|
| G-PCC | `positionQuantizationScale`（v23ではsequenceScaleの互換alias） | 0.125 / 0.25 / 0.5 / 0.75 / 0.875 / 0.9375 |
| Draco | position quantization `-qp` | 5 / 6 / 7 / 8 / 9 / 10 |
| OctAttention | 公式dataPrepareと同じ量子化step `qs` | 8 / 6 / 4 / 3 / 2 / 1 |
| SparsePCGC dense lossy | `scale_AE` / `scale_SR` | (1,0) / (0,1) / (1,1) / (0,2) / (1,2) / (0,3) |

G-PCCの既定設定は手元の`cfg/octree-predlift/lossy-geom-lossy-attrs/longdress_vox10_1300/r01..r06/encoder.cfg`で確認した6量子化値。共通設定はr01を使い、attribute coding無効、trisoup無効、native量子化、原点・座標scaleをstreamに保存する。**このプリセットを全dataset・全論文のCTCと同一だとは主張しない。** `--gpcc-config`と明示RD設定で対象実験へ合わせる。

Dracoはpoint-cloud mode、compression level 7固定。attribute／法線を除いたgeometryだけを入力する。native PLY loaderがfloat32／int32限定のため、float32で座標を厳密に表せない入力は暗黙に丸めず失敗する。

OAは`round((xyz-min(xyz))/qs)`で公式と同じgridを作る。学習済みモデルの最大深さ12、最小symbol数の制約を検査する。entropy推定値でなく算術streamを保存し、過去の復号symbolだけから確率を再計算する自律decoderで最終点群を復元する。

SparsePCGCは既存のdense lossless／AE／SRモデルとdownscale→encode→unpack→decode→upscale演算を再利用する。既存`LossyCoderDense.test`はupscaleにencoder側`x`を渡しているため、その関数の返り値をそのまま使わず、wrapperで**実復号したlow-resolution tensor**をupscaleへ渡す。lossless部分の座標一致も確認する。

### decoderに必要な付加情報

- OA：算術streamに加え、symbol数・qs・XYZ offset・context長・深さを56-byteの`MYOA0001` recordへ実際に保存し、復号時に読み直して計上する。元occupancyやencoder確率を含む`.mat`をdecoderへ与えない。
- SparsePCGC：AE／SR、voxel size、XYZ offset、各stage点数（uint64）、AE stream bytesを`MYSP0001` recordへ保存し、復号時に読み直す。既存の「stage点数あたり4 bits」等のlogical推定ではない。

いずれもcodec本体を変更しない評価用wrapperである。side情報を含む物理rateは、論文のcontainer／logical overhead規約とは異なる。比較時には必ず明記する。固定された共有codecモデルの重みは通常のcodec評価と同様に各frameのbitsへ加えず、hashを記録する。

密な10～15点は`--points-json FILE`で指定する。キーは上表の実在パラメータのみ。SparsePCGCに未実装の連続quality値を仮定しない。必要なら実在する`voxel_size`も明示できるが、voxelizationまで変えた別条件として扱う。

## 4. D1／D2とBD-rateの式

全ての測定で、参照は**Original GT**、評価対象は**実streamからの最終復号点群**。

`bpp = 全geometry stream・復号必須side recordの実bytes × 8 / Original GTのvertex数`

加工後／復号後の点数で割らない。GTに重複があっても分母はOriginal vertex数として固定し、metric内部の重複処理とは分ける。

手元のMPEG `pc_error_d` version 0.13.4を使用する。D1はpoint-to-point、D2はGT法線に基づくpoint-to-plane。対称集計結果`mseF`／`mseF,PSNR`を読み、

`PSNR = 10 log10(3 * resolution² / symmetric MSE)`

の定義を既知誤差テストで確認した。symmetric、dropdups=2、neighborsProc=1、averageNormals=1、1 threadを両系列で固定する。GT法線は座標・順序が完全一致し、finiteなunit vectorであることを要求する。

**互換性の回帰防止：** このbinaryはbinary double型法線で既知のD2誤差1を0と出すケースがあった。GT座標を保持したgeometry-only参照と、法線propertyだけfloat32へ統一した別PLYを`-n`へ渡す。法線推定ではなく保存形式の統一であり、両系列で同じ値・同じhashを使う。元法線と変換後法線のhashを保存する。

各metric独立に、quality軸上で`log10(bpp)`をSciPy `Akima1DInterpolator`で補間し、両曲線の共通区間`[qL,qH]`だけを積分する。

`Δ = ( ∫ log10(rate_MINE(q))dq − ∫ log10(rate_Baseline(q))dq ) / (qH−qL)`

`BD-rate[%] = (10^Δ − 1) × 100`

積分はpiecewise cubicの解析積分。PSNR一致許容誤差・外挿・測定値の事前丸めはない。負が改善。D1／D2を平均して混ぜない。

無限PSNRは記録したうえで有限の曲線knotから除外し、警告する。異なるrateで同一PSNR、非単調性、有限点不足、共通区間なしは計算不能として値を空にする。厳密一致する重複RD knotだけは冗長として記録付きでまとめる。最低4有限点、6点推奨。狭い区間・75%未満の共通区間、曲線交差、Akimaの負勾配overshootも記録する。

密な測定と6点subsetのBD-rateを、通常の区間と**両推定で同一の共通区間**の双方で比較する。差はpercentage pointsで保存する。実際の非同一曲線の補間誤差は実測対象ごとに検証が必要であり、自己比較の誤差0を一般的な誤差上限にはしない。

## 5. 保存場所と形式

既定：`<既存argsのlog_root>/YYYYMMDD/MyNetwork_test_BDrate/<codec>/<HHMMSS_UUID>/`。

- `results.xlsx`：BD-rate、各RD点、Sampling validation、Provenanceの4 sheet。
- `configuration.json`／`summary.json`：丸めない測定条件・計算結果・hash・環境version。
- `rd_points.jsonl`：各点の成功／失敗を即時flush＋fsync。異常終了時も完了済み点を保持。
- `cloud_XXXX/NN_Baseline|MINE/`：実bitstream、decoder side record、最終復号PLY、metric用GT／法線、全command／stdout／stderr／returncode、measurement JSON。
- `cloud_XXXX/network/`：既存test.pyの推論profile、MINE PLY、checkpoint厳密照合、実効args、provenance。
- `rd_XXXX.png`：D1／D2 RD曲線。無効曲線を勝手に補正しない。
- `manifest.json`：成果物のSHA256。

dataset全体平均は各点群BD-rateの算術平均。失敗した点群を除いた平均を「全体平均」として出さない。全体が不完全な場合、正式平均は空欄、有効分のみの平均を別fieldに出す。

## 6. 実測による動作・精度検証

### 計算・既存route

- 新規28テスト合格。既知−50%、＋25%、0%、異なるknot数、区間限定、非単調・点不足・欠損法線・NaN・外挿禁止・D1/D2・物理side bits・ファイル保護等。
- 既存`test_geometry_codec_routes.py`の9テスト合格。既存routeの変更はしていない。
- FAU-LMS公開`bjontegaard 1.3.0` wheelを既存環境へinstallせず一時ディレクトリから読み、seed75の100ケースで照合。最大絶対差 **1.4210854715202004×10⁻¹⁴ percentage points**。
- 100点の平面を法線方向へ1だけずらすnative metricテスト：D1 MSE=1、D2 MSE=1、resolution1023で両PSNR=64.9687 dB。ツールが出す約6桁の文字列精度を保持し、それ以上の内部精度を得たとは主張しない。

### 実codec自己比較（学習効果の証拠ではない）

解析法線付き576点合成曲面を使用。既存データセットの欠損法線へ代替値を入れた実験ではない。

- G-PCC native release-v23.0-rc2：6点×2系列のencode／native decode／D1／D2成功。通常CTC6点では合成点群特有の非単調を検出してBD-rateを拒否。別途明示した`.03125,.0625,.125,.25,.5,.75`の単調6点では自己比較D1／D2 BD-rate=0%。
- Draco：既定6点×2系列成功。D1／D2とも0%。さらにqp4～15の12点×2系列も成功。6点subsetと12点の自己比較BD-rate差は0ポイント（自己比較なので期待値通り）。
- OA：qs8の実算術符号化＋自律復号成功。120 symbols、241復号点、算術456 bits＋side448 bits＝**904 bits**、D1 39.5934 dB／D2 43.4654 dB。
- SparsePCGC：AE1/SR1の実encode／decode／low-resolution一致検査／upscale成功。主stream3040 bits＋side1144 bits＝**4184 bits**、576復号点、D1 24.1057 dB／D2 52.6201 dB。高いD1誤差を隠さずそのまま記録。既定6点×2系列も全て圧縮・復号・metricに成功したが、非単調曲線のためBD-rateは空欄。
- OA／SparsePCGCの上記単点を再実行し、主stream・side record・復号PLYのhashがそれぞれ一致した。SparsePCGCは別途、負座標＋voxel_size=2のlossless wrapper検証も行い、`unique(round(input/2)*2)`との座標完全一致を確認した（7664 bits）。

成果物：

- `experiments/results/20261019_bdrate_native_v2/reference_validation.json`
- `experiments/results/20261019_bdrate_gpcc_valid_v1/results.xlsx`
- `experiments/results/20261019_bdrate_draco_v3/Draco/results.xlsx`
- `experiments/results/20261019_bdrate_final_dense_v1/results.xlsx`
- `experiments/results/20261019_bdrate_final_learned_v1/`
- `experiments/results/20261019_bdrate_sp_curve_v1/SparsePCGC/results.xlsx`
- `experiments/results/20261019_bdrate_signed_control_v1/validation.json`

### 学習済みNetwork経路の確認

指定例`20260801/072958/best_loss_joint.pth`を実際に読み込んだ。現在のNetworkではPointTransformer adapter等の学習可能キーが欠け、Amount headの一部shapeも異なる。**形状補完・ランダム初期値で埋めて「当該学習済みモデルの性能」とすることを拒否した。** 失敗ログは`20261019_bdrate_checkpoint_v2/`に保存。互換性のあるモデル／元の構成設定が必要である。

別の既存G-PCC checkpoint `20261017/192050`では厳密照合を通過し、CPUで既存test.py full-cloud推論、MINE保存、6点×2系列のActual encode／decode／D1／D2まで完走した。同じGTで2回保存したMINE PLYはSHA256が一致：`474257a924102793813283b3ed480aa873ee0881d08b2ed583e74b58f0192871`。

この合成点群ではAdd／Prune／Adjustは0件だったが、既存推論のvoxelizationは行われるため、MINE PLYはGTそのものとは限らない。得られたRD曲線が非単調だったため、BD-rateは空欄にした。**これを提案手法の改善率・汎化性能の実証とは扱わない。** 成果物は`experiments/results/20261019_bdrate_final_network_v1/`。保護ソース／評価器ソース／checkpointは全て開始・終了hash一致。

## 7. 未検証事項と修士論文での注意

1. 実際のholdout datasetについて、互換checkpointと正しいGT法線を指定した大規模なMINE BD-rateは未測定。初期例の旧checkpointを評価成功と報告しない。
2. 非同一曲線の6点対10～15点の誤差検証は、実際の評価対象で実施する必要がある。今回のDraco自己比較だけではその誤差は推定できない。
3. OAの自律decoderはnodeごとのforwardが必要で高コスト。576点合成曲面の既定6点検証ではqs8/6/4/3/2の両系列は成功したが、qs1のCPU workerは300秒timeoutを記録した（`20261019_bdrate_oa_curve_v1/`）。小規模実decodeは確認済みだが、大規模frameの一律な実行時間・GPU／CPU間bit-exact性を保証しない。timeout時に推定rateへfallbackしない。
4. SparsePCGC合成点群の非単調性・大きい歪みは検出済み。実際の対象で有効なRD点を測り、必要なら明示設定を見直す。都合の悪い点を黙って消す処理はない。
5. OA／SparsePCGCのwrapper side overhead、codec version／config、共有重み、resolution、GT法線生成元、split、metric versionを論文へ記載する。CTC・論文の同一条件を再現したこととは区別する。

## 参考にした一次資料

- [FAU-LMS bjontegaard：Akima、共通区間、log-rate積分](https://github.com/FAU-LMS/bjontegaard)
- [MPEG G-PCC TMC13](https://github.com/MPEGGroup/mpeg-pcc-tmc13)：実際のRD設定は手元v23のCTC／TMC3.cppも確認。
- [Google Draco](https://github.com/google/draco)：position quantization。
- [OctAttention公式](https://github.com/zb12138/OctAttention)：量子化、算術coder、causal decoder。
- [SparsePCGC公式](https://github.com/NJUVISION/SparsePCGC)：dense lossyのAE／SR rate設定。
