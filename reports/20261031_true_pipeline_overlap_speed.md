# Trueルートの候補復元・Actual待機の最適化とFalse保全確認

2026-10-31。開始commit `0e232c5`、開始時tracked/untracked差分なし。

## 結論

**True専用の候補復元とCPU/GPU待機の重複を削減した。ただし、通常訓練に近い24 Step比較で全Step時間は3.828→3.642秒、約4.84%の短縮に留まる。大幅な全体高速化や1秒/Stepは達成していない。**

Falseは変更前・中間・最終コードでそれぞれ実訓練を8 Step行った。最終8 Stepすべてで選択plan、操作数、編集voxel hash、Actual bits、圧縮率が変更前と一致した。Geometryもログの4桁精度で一致した。勾配・更新パラメータのbit単位の完全一致は主張しない。

`train.py`、`args.py`、`network.py`、各head、lossの目的・重み、codecソース、既存checkpointは変更していない。既存processへsignalを送らず、実験の出力・cache・checkpoint・registryは独立ディレクトリに保存した。実行中の訓練は既に読み込んだコードで継続し、変更は次回起動から有効になる。

## 1. 最新ログと計算経路

開始時の最新訓練は `20261031/MyNetwork_train/SparsePCGC/134351.txt`、PID 954131の `train.py --test_route true`。開始時点で確認した47 Stepのうち直近20 Stepは平均約4.83秒、Actual部分は約0.47秒だった。ユーザーの訓練とそのnative workerを稼働させたまま実験したため、GPU負荷・計測時刻による変動がある。

現在の経路は、固定入力・特徴の読込 → native CDF context準備 → 自律候補生成/復元 → learned Where/Actor/Proposalと3操作別Amount → joint planと点編集 → Geometry → Actual encoderとproxy更新 → Actual方策損失 → policy勾配捕捉と全体backward → 既存の勾配置換・clip → optimizer → 記録、である。

最初のcProfileでは、warm Stepのnetwork forward 1.445秒の中で候補pool復元が0.607秒、そのうち`pickle.loads`が0.568秒だった。対象フレームには45,950候補/操作、合計137,850候補がある。復元時に大量のPython辞書・listを生成し、自動循環GCが繰り返し大きなグラフを走査していた。

一方、Actual待機にはnative側の入力処理・符号化・統計・allocator処理とGPU共有の待ちが含まれる。全Stepを次のforwardまでprofileした独立StepではActual encode区間1.506秒、native報告のwallは1.336秒、`enc_time`は0.293秒だった。これらは異なる測定区間であり、差をそのまま「不要なcodec演算」とは扱わない。native側は既にCPU統計とGPU encodeを並列化していた。

## 2. 採用した変更

### 候補poolのGCをまとめる

True専用`AutonomousPoolRAM`を追加した。zlib圧縮、pickle protocol、候補の内容・順序、完全な入力内容hash、cache容量制限を維持する。復元/初回構築中だけ自動循環GCを止め、generation 1までまとめて回収し、元のGC状態を例外時にも戻す。参照カウントは止めない。

generation 0だけを回収する最初の案では、復元が速くなっても後続のplan生成へGC待ちが移った。これは全体の改善とは認めず、generation 1まで対象とする方式へ変更した。

### Actual待機中に次の固定poolを復元する

実際に観測したframe遷移から、既にcacheへ格納済みの次poolをCPU thread 1本で先読みする。次回getでも座標内容・順序・設定の完全hashを検証し、予想が外れれば通常経路を使う。追加のdecoded poolは最大1つ。遷移表もcacheの格納元数を超えて増えない。

学習中のlogit、選択結果、勾配、RNG、GPU Tensorは先読みしない。Actual request送信後に開始し、codec応答の待機を終える前にfutureをjoinする。codec timeout/errorでもjoinし、GCを元へ戻してからモデル/損失計算を再開する。phase変更時にはexecutorを閉じる。

### Trueのclip記録転送をまとめる

モジュールのグループ分け、clip上限、`clip_grad_norm_`の呼出順、実際のclip演算を維持した。記録用の各parameterのdouble二乗和をdeviceごとにまとめてCPUへ転送し、Pythonでの加算順序も維持した。単体テストでは逐次版とclip済み勾配・記録値が一致した。

最初のprofileではclip記録0.163秒、変更後0.006秒だったが、**変更なし再実行でも0.008秒だった**。GPU待ちが含まれるため、この1 Stepの差を96%高速化の根拠とはしない。

## 3. 数値結果

同じseed 22、同じ15 Amount bins、同じ入力順・codec設定で比較した。Trueの速度試験は4系列各1フレームを繰り返す4-frame Episodeであり、本番の100-frame Episodeの設定を変更したものではない。最初の5 Stepを除いた値をsteadyとして集計する。先読みは本番でも初回の候補構築をなくす仕組みではない。

| 測定対象 | 変更前 | 最終案 | 条件 |
|---|---:|---:|---|
| network forward区間平均 | 1.631秒 | 0.971秒 | 軽量計測24 Step、6～24 Step |
| main threadのpool生成/復元区間平均 | 0.714秒 | 0.014秒 | 同上。復元演算自体はActual待機と重ねる |
| 初回4フレームのpool区間平均 | 1.959秒 | 1.635秒 | cache missを含む |
| 全Step wall平均 | 3.828秒 | 3.642秒 | 6～24 Step、約4.84%短縮 |
| 全Step wall中央値 | 3.699秒 | 3.667秒 | 同上。中央値の改善は約0.85% |
| main processのCUDA最大allocated | 11,683.34 MiB | 11,683.34 MiB | 同上。native workerとは別の指標 |

軽量計測は、診断用の全Tensor保存と追加の強制CUDA同期を外して実行した。これらを有効にした16 Step比較では全wallは3.542→3.618秒であり、全体短縮を確認できなかった。都合のよい区間だけを全体の高速化として扱わない。

短い共有GPU上の比較であるため、全体の約5%差を独立環境でも再現する統計的保証はない。候補復元区間の削減・先読みhitは直接確認したが、残るActualと学習処理は依然必要である。

## 4. 計算結果とGPU非決定性

診断用Tensor保存付き16 Stepでは、変更前と最終案のActual圧縮率は16/16一致、操作数は16/16一致、編集hashを含むplanは13/16一致。Geometry最大差は`9.4622373581e-7`だった。

同じ変更前ソースをsnapshotから再実行した対照実験でも、plan一致は12/16、Actual最大差0.0278587 percentage point、Geometry最大差`9.4622373581e-7`だった。したがって、同seedでも候補の同点付近の順位やGPU勾配のbit完全一致は保証されていない。

軽量24 Stepでは操作数24/24一致、plan一致14/24、Actual最大差0.0417976 percentage point、Geometry最大差`2.5816261768e-6`。**全Stepの完全一致や、長期収束後の最終圧縮率維持を確認済みとはしない。** 新cacheの値・順序・fresh object性と、clip演算の一致は別途単体テストで確認した。

## 5. Falseの実訓練による確認

seed 21、productionと同じ100-frame Episode、通常の連続RNG、longdress 1051～1058の先頭8 Stepを使った。private den6 teacher/cacheのみを新規作成・再利用した。

| Step | Actual圧縮率: 変更前→最終後 [%] | Geometry: 変更前→最終後（ログ精度） |
|---:|---:|---:|
| 1 | −3.259507→−3.259507 | 0.0330→0.0330 |
| 2 | −3.970223→−3.970223 | 0.0309→0.0309 |
| 3 | −3.777417→−3.777417 | 0.0345→0.0345 |
| 4 | −3.390515→−3.390515 | 0.0336→0.0336 |
| 5～8 | 全4件で一致 | 全4件でログ値一致 |
| 8 Step平均 | −3.494144702→−3.494144702 | 0.0327625→0.0327625 |

8/8で候補数、Amount、編集voxel hash、GT/MINE bitsが一致した。最終False比較の全parameter最大絶対差は`1.4769703e-5`、勾配最大絶対差は`1.2598932e-4`であり、Tensorのbit完全一致とは異なる。Geometryの比較は4桁ログであり、未記録のフル精度一致は主張しない。

Falseの既定/明示argsは3 datasetで一致、9 poolとRNGが一致、12組のnative worker command・codec fingerprintも一致した。Falseでは先読みhookを実行しない。Falseの変更前43秒台→変更後7秒台という時間差は、private teacher初回構築の有無によるもので、今回の最適化効果には数えない。

直近に完走Stepのあった旧Falseログ`20261028/.../145827.txt`の最初のframeはGT 38,280、MINE 37,040 bitsで−3.239289%。今回の**編集前から**GT 38,288、MINE 37,040 bitsで−3.259507%だった。現行legacy workerには、native normalizerが既にlogical bitsを返す場合にもstage補助bitsを加算し得る既存処理がある。旧GT cacheのproducerまで今回追跡していないため、旧値の生成経緯は断定しない。Falseを保持する指示に従い、この会計処理や旧cacheを変更せず、現行の同条件before/afterを比較した。

## 6. 追加で試し、本訓練へ採用しなかった案

Codecの既存SR pooling 2段を整数floor/uniqueでまとめ、巨大なraw SparseTensorを作らず同一のcoded-level Tensorを作る独立prototypeも検証した。入力の削減やcodecの品質設定変更を加えるものではない。試験は`experiments/fused_native_pool_probe.py`だけで行った。

GT/既存編集fixtureの2点群 × native/fused各2回、計8回Actual encodeした。coded-level座標・features・tensor strideは全8件で基準と完全一致し、同一入力の物理/論理bitsも一致した。CUDA最大allocatedはnative約0.447 GiB、fused約0.292 GiBだったが、平均wallはnative0.401秒、fused0.736秒で、速度改善の証拠を得られなかった。GPU負荷の変動もあり、この小規模試験を一般的な優劣とはしない。**本訓練のcodec経路には組み込まなかった。** stream hashはnative同士の再実行でも異なるため、byte完全一致とは扱わない。

単純なpool先読みthread案も試したが、復元中GCによりActual待ちが増えた。採用案はGCをまとめ、待機から戻る前に完了を保証する版である。世代0回収だけの案、thread化だけの案を成功結果へ混ぜていない。

## 7. ファイルと再現

- 新規`models/utils/cache/autonomous_pool_ram.py`: True専用のlossless cache、GC、bounded先読み。
- `models/utils/pointcloud/autonomous_proposals.py`: True専用cacheの利用、初回構築のGC、phase終了処理。
- `models/utils/loss/actual_encoder.py`: Trueだけでrequest送信後の先読みと例外時join。codec演算・request内容は維持。
- `models/utils/training/autonomous_joint_learning.py`: Trueのclip記録転送をまとめる。
- `tests/test_autonomous_pipeline_speed.py`: fresh object性、key不一致、容量、遷移表の上限、GC状態、例外時join、clip一致、False requestを検証。
- `experiments/autonomous_speed_probe.py`: before snapshot再実行、False・本番frame数・連続RNG、軽量計測、全Step profileを追加。
- `experiments/summarize_autonomous_speed.py`: FalseのActual計測値とGeometry記録精度も区別して集計。
- `experiments/verify_autonomous_false.py`: command/fingerprintの保全確認を追加。
- 独立追加: `pool_decode_benchmark.py`、`summarize_pipeline_benchmark.py`、不採用prototype `fused_native_pool_probe.py`。

成果物root: `/data/maejima/log/independent_experiments/20261031_training_pipeline_berwltut/`。

`summary_final.json`、`false_final_comparison.json`、`true_comparison_final.json`、`true_unchanged_control.json`、各runの`invocation.json`/`effective_args.json`/`steps.jsonl`/`integrity.json`、開始時source snapshot、profile、独立checkpointを保存した。`invocation.json`はharnessによる隔離先設定を含み、通常訓練へそのまま流用する設定ファイルではない。

75件の関連テスト合格。新cacheのclearテストでbase classにclear APIがない不整合を検出し、自身のentries/bytesを解放するよう修正して再実行した。各実訓練runのsource変更検出は空。最終修正後にもFalse実訓練8 Stepを再実行した。

CPU thread数2、両環境のPythonに`-B`、`PYTHONDONTWRITEBYTECODE=1`を使用。評価データを減らしたことを本番の高速化として数えず、dtype、候補数、Amount分布、探索schedule、損失、勾配経路、Actual回数を変更していない。残る全体時間について、大幅短縮を達成済みとは報告しない。
