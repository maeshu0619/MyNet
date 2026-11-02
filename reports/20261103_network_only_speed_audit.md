# Network側の訓練コードに限定した時間短縮の再調査

2026-11-03。今回は調査と`experiments/`の独立試作のみ。本番のtrain.py / Network / loss / args / worker / codecは編集していない。開始時にGit状態とソースを保存し、終了時に本番135ファイルとSparsePCGC本体53ファイルのSHA256不変を確認した。既存checkpoint・ログを上書きしていない。

**精度を変えずに削れることを最も強く確認できたのは、Networkへ渡す入力点群の毎StepのASCII再解析である。全内容照合付きの解析済みdisk cacheなら、実4系列で1.38～2.16秒→0.023～0.049秒だった。これは圧縮コードの最適化ではない。ただし表示Step時間の外側の処理なので、表示15～20秒がそのままこの差だけ短くなるとは主張しない。**

## 1. 入力PLYの重複解析

`models/utils/data/dataset.py:113`付近は、各点の行をPythonでdecode/splitし、百万行近い文字列のlistからfloat32配列を作る。現在の訓練ではdataset/episode RAM cacheが0であり、固定100フレームを各Episodeで再読込している。

既存parser、NumPy loadtxt、NumPy fromstringを、4系列の先頭フレームとfloat32の丸め境界等を含む小数fixtureで比較した。方式の順序は交互、各3反復。RGB・行順・負ゼロを含め配列の全byteを照合し、45/45件で一致した。

さらに、**元parserで得た配列そのもの**をprivate NPYへ保存する試作を比較した。毎回元PLYの全byte SHA256を確認し、NPYをpickle無効で読み、配列全byteのSHA256も照合する。filenameやmtimeだけを信用せず、原データ・cacheの変更を検出する。元の入力ファイルには書き込まない。各5反復、20/20件でdtype・shape・全byteが一致した。

| 入力 | 元parser中央値 | loadtxt中央値 | 内容検証付きNPY cache中央値 |
|---|---:|---:|---:|
| longdress 1051 | 1.721621秒 | 0.428815秒 | 0.029854秒 |
| loot 1000 | 1.712532秒 | 0.216795秒 | 0.043060秒 |
| redandblack 1450 | 1.376467秒 | 0.164837秒 | 0.022618秒 |
| soldier 0536 | 2.156065秒 | 0.377607秒 | 0.049203秒 |

NPY cacheは作成後のhit時間。初回parserと保存は別途必要。4フレームの結果を100フレーム・全datasetで確認済みとは扱わない。読み込み・hashにはOS page cacheの影響もある。大規模RAM cacheを有効にする案とは異なり、必要なフレームだけをロードする。

**実装する場合の第1候補**は、True側datasetの入力経路にだけ内容照合付きparsed cacheを設けること。Falseと既存loaderは維持し、miss時は元parserを使う。原ファイルのhash、loader/schema version、RGB有無、shape、dtype、配列hashを照合し、cacheが破損した場合は再解析する。実験のassert停止をそのまま通常訓練へ持ち込まず、安全なfallbackとatomic保存が必要。今回の試作はまだその本番実装ではない。

## 2. Actual方策勾配保存の余分なbackward

`train_runtime.py:5181`の`_capture_den6_policy_only_gradients()`は、policy loss全体を`backward(retain_graph=True)`し、decision headの勾配だけcloneしてから全parameterのgradを消す。その後`train.py:4391`等で全体lossをもう一度backwardし、保存していたdecision勾配へ戻す。

したがって「2回目のbackwardを丸ごと削除する」は正しくない。2回目は上流特徴や他lossの実更新に必要である。一方、1回目で計算して直後に捨てる**非対象parameterへの勾配**には削減余地がある。

独立試作では、1回目を`torch.autograd.backward(policy_loss, retain_graph=True, inputs=params + decisions)`へ限定した。decision Tensorも含め、既存のretained grad監査を維持する。元のcapture/restore方式と、reentrant/non-reentrant checkpointの2条件で、必要なparameter勾配・decision勾配が完全一致。各条件で3 Stepの全parameter/gradient・Adam更新後parameter・乱数状態も完全一致した。

**これはCPUの検証用モデルでの結果**。実Networkの全head、checkpoint再計算、AMP、hook等について同じ結果になることは未確認。GPUの実際の短縮秒数も未計測なので、本番へは組み込んでいない。Actual/Geometry/Proxyのloss倍率変更、stop_gradient追加、学習対象除外を行う案ではない。

## 3. 候補選択・診断の重複処理

`structure_actuator.py:5789`の`_score_stats()`は、min/max/mean/stdごとに`.cpu()`する。1組につき4回のhost転送を、同じreductionを計算した4 scalarのstack→1回のhost転送にまとめる余地がある。loss・候補数・samplingを変えずに同期回数を減らせる候補であり、GPU混雑時には特に切り分ける価値がある。ただし今回はGPU速度を検証していない。

同ファイルの6638～6798付近はlegacy support/ordered_indicesを構成した後、autonomousでは`sample_where_order()`の結果でordered_indicesを上書きする。8295付近では比較用deterministic順位も再度sortする。診断用順位の重複計算を調べる対象である。

ただしlegacy側のdeterministic IDsは他の監査・比較へも渡される。また診断モードではrandperm等を使う箇所がある。**上書きされるからといって前半を一括削除すると、ログ・乱数状態・候補選択を変え得る**。同じ入力Tensor、温度、stable sort、同点順序を明示した共有と、全debug field・RNG・最終編集集合の照合が必要。今回ここは編集していない。

## 4. 独立訓練で取得できた区間と限界

現行コードで7 Stepを予定したが3 Stepまで完了し、4 Step目のGPU admission待ち中に私のprivate訓練PID 1699750だけをSIGINTで止めた。子worker 1700749の終了も確認した。理由は共有GPU空きが1 GiB未満になり、既存訓練との競合を避けるため。

| 区間 | 完了した初期3 Stepのwall区間 |
|---|---:|
| Network forward | 6.832～7.571秒 |
| その中の候補pool生成/復元 | 1.559～2.952秒 |
| その中のplan選択 | 1.358～1.944秒 |
| Actual方策勾配capture | 0.636～0.820秒 |
| 全体backward | 0.324～0.402秒 |
| Geometry loss | 0.065～0.144秒 |
| optimizer.step | 0.0019～0.0035秒 |

これらは入れ子区間であり、足し合わせない。軽量wrapperのwall時間はGPU待ちを含み、純kernel時間ではない。初期3 Stepでpool/frozen cache missを含むため、成熟後の速度ではない。planned Step6のcProfileは未取得。4-frame Episode、初期重み、source-grad追加診断OFFの独立設定であり、本番100-frame Episode/学習済み状態への外挿はしない。短縮後の実訓練やActual精度維持を確認した比較でもない。

その間、既存訓練ログに`SparsePCGCAutoCPUFallback: reason=encode, free_mb=2605.9, required_mb=4096, ... selected_device=cpu`が記録され、既存訓練PID 766550のworkerはCPUのPID 1703418へ自動移行していた。私のprivate検証も同じGPUを使用していたため、競合へ寄与した可能性があり、この時間帯を通常条件の比較として使わない。GPUの他jobも稼働していたため寄与の秒数・単独原因は断定しない。既存processに対する停止、設定書換え、workerの手動切替・再起動は行っていない。以降はGPU実験を中止した。

## 5. 推奨順と保護

| 優先 | 対象 | 数値・精度の確認状況 | 次に必要な確認 |
|---:|---|---|---|
| 1 | Network入力の内容照合付きparsed disk cache | 実点群20/20で全byte一致、約1.35～2.11秒/読込を削減 | Trueだけへ分岐、cache破損/変更時fallback、100-frameで総wall比較 |
| 2 | capture backwardの対象限定 | CPU checkpoint fixtureで勾配・3 Adam更新・RNG一致 | GPU余裕のある環境で同一graph上の全実head照合、更新後Actual/Geometry比較 |
| 3 | score診断のhost転送集約 | 4 scalar転送/組を1回にできるコード上の候補 | GPUで統計値完全一致と同期回数・総wallの測定 |
| 4 | legacy/Trueの順位診断重複 | 上書きと再sortを確認、他consumerあり | 同点・debug・RNGを含む同値性確認後に限定的に共有 |

元のfloat32、全点、候補集合、Amount/Where分布、Actual頻度、loss式、clip、optimizer更新、checkpoint方式は一切変更していない。**現在は改善候補の実測調査であり、訓練の高速化を本番へ実装済み、最終精度維持を検証済み、1秒/Step達成済み、とはしない。**

成果物は`/data/maejima/log/independent_experiments/20261103_network_only_rf8i6b4w/`。`summary.json`、`before/manifest.json`、`codec_sha.json`、入力比較・cache結果、`capture_selftest.txt`、中断した訓練の`steps.jsonl`と`integrity.json`を保存した。追加ファイルはこのレポートと以下の3つだけ。

- `experiments/network_input_speed_audit.py`: 元parser/loadtxt/fromstringの全byte一致・速度比較。
- `experiments/network_input_cache_audit.py`: 元parser由来の解析済み点群の、入力/配列全内容照合付きcache試作。
- `experiments/network_capture_speed_audit.py`: 必要な勾配だけのcapture試作とCPU更新一致確認。`--self-test`はGPUを使わない。
