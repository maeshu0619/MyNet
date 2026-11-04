# 3～5秒/Stepから10～20秒/Stepへの長時間化の再調査

2026-11-03 06:13 JST以降。比較対象は実在する旧ログ`20261031/MyNetwork_train/SparsePCGC/134351.txt`と、現在実行中の`20261102/.../221459.txt`。今回は読み取り調査のみ。追加の訓練・GPU計算・native encodeは行っていない。SparsePCGC本体、本番コード、args、checkpoint、既存ログ、実行中processは変更していない。

**旧訓練が3～5秒だったことを、同じ100フレームを使う実ログで確認した。現在はActualがCPUへ移行したうえ、Networkが使うGPUも別jobと強く競合している。現在のActual以外の区間も約5.59秒増えているため、CPU切替だけで全遅延を説明することもできない。**

## 1. 同じフレーム・同じglobal stepで比較

現在の完了1,646 Stepの末尾100件、`global_step=1546～1645`に対応する旧100件を抽出した。入力pathと順序は100/100件一致。両者とも初期cache構築後で、区間内のGT再encode=0、全操作No-op=0だった。旧runの後半の探索停止済み期間だけを使った比較ではない。

| 同一100 Stepの平均 | 旧訓練 | 現訓練 | 観測された増加 |
|---|---:|---:|---:|
| 表示Stepと同じ`step_before_audit` | **4.1398秒** | **13.5282秒** | +9.3884秒 |
| 上記に含まれるActual区間 | 0.7073秒 | 4.5098秒 | +3.8025秒 |
| その他：Network・loss・backward等 | 3.4325秒 | 9.0183秒 | +5.5859秒 |
| data load後～cleanup完了 | 4.2169秒 | 13.7222秒 | +9.5054秒 |

Actualは既存の`actual_encode_time_total`を使う。native NNだけの時間ではなく、処理全体・転送・worker待ち等を含む。その他はStepからActualを引いた残差であり、Network forwardだけを意味しない。増加の分割は観測上の内訳で、各原因へ因果的に割り当てた秒数ではない。

比較区間の平均編集数は旧15,987.97点、現13,671.22点。現在の方が約14.5%少ない。操作位置・weightsは異なるが、「操作数が何倍にも増えたため」という説明は支持されない。両者ともCが平均+15.77 / +13.20%の探索中であり、旧−5%の収束後だけと比べたものでもない。

より広く、両runの`global_step=100～1645`（1,546件）でも、旧3.6682秒、現13.6459秒だった。同じ100-frame Episode同士で、旧の3～5秒という実績は再確認できた。4-frame反復の独立試験2.75秒を本番100-frameへ外挿した数値ではない。

## 2. 確認した実行条件の違い

### GPUは空きVRAMがあっても計算が空いていない

追加GPU処理を起動せず、`nvidia-smi pmon -s um -c 20 -d 1`で20秒観測した。別jobの1 processだけで、20有効sampleのSM activityは**平均77.95%、49～99%**だった。本訓練mainは数値が得られた10 sampleで平均18.3%、残りは`-`だった。`-`を0とみなして平均していない。

全体GPU使用率は100%。これを本訓練がGPU全体を使えていると読むことはできない。pmonの割合は予約容量ではなく、process間の値も単純に足して占有率へ換算しない。**共有競合を直接確認したが、旧と同じ専有条件で比較していないため、競合が正確に何秒を増やしたかは未確定。**

今回の同一runでも、最初の100 Stepは18.76秒、global_step 500～599は19.36秒、900～999は7.28秒、その後の比較末尾100件は13.53秒と変動した。各blockの入力集合は固定100フレーム。parameterや行動は学習で変わるため完全な同一計算のablationではないが、長時間化が一定のコード追加コストだけで起きているとは扱えない。

### Actual workerはCPUへ切り替わったまま

ログには次の記録がある。

`SparsePCGCAutoCPUFallback: reason=encode, free_mb=2605.9, required_mb=4096, requested_device=auto, selected_device=cpu`

これは`global_step=1355`の記録後。現在のworker PID 1703418は起動引数`--device cpu`で、04:56:01に起動している。main PID 766550は前夜22:14:57から継続している。

Network側wrapperの`actual_encoder.py:640`の`_restart_worker_on_cpu()`は、`args.sparsepcgc_device="cpu"`に変更して永続workerを作る。その後、GPUの空きが増えても自動でGPUへ戻す経路はない。今回の観測開始時には18,166 MiB空いていたが、workerはCPUのままだった。

**現在のActual 4.51秒と旧GPU時の0.71秒は、この実行条件の違いを含む。** ただし、CPU切替前の直近100件でもStep13.78秒、Actual6.00秒で、すでに遅かった。さらに最初の600 StepではActual約8.6～9.0秒だった。CPUへ切り替えたことが遅延の開始原因、あるいは切替直前より遅くした原因、と断定しない。混雑GPUで走り続ければ必ずCPUより速い、という状況ではない。

過去の私の独立GPU検証も競合へ寄与した可能性があり、この切替時間帯を通常条件の性能比較へ使うべきではない。今回は追加GPU計算を一切行わず観測した。他のjobや既存workerを停止・変更していない。

### 電力制限は有効、thermal throttleの証拠はない

観測時のSM clockは約1620～1635 MHz、温度81～82℃、SW Power CapはActive、上限300 Wだった。一方HW/SW Thermal SlowdownはNot Active、GPU Slowdown Tempは95℃。温度だけからthermal throttlingを原因としない。旧高速区間と同時条件のclock記録はないため、この制限だけで何倍遅くなったかも確定しない。

## 3. コード・設定から確認したこと

旧・現在で、次は同じだった。

- 100-frame Episode、4系列25フレームずつ、順序、DataLoader並列無効、入力RAM cache=0。
- native dense_lossy、m=8 / AE=0 / SR=2、skip_decode、Actual頻度1/Step、点数間引きなし。
- KNN backend、head chunk=131072、freezeしたencoder、main/Where/Amount/CriticのLR群。
- 両runともTrueの自律探索で、den6の固定Amount/Where planを使う旧Falseとの比較ではない。

現在は学習する探索幅が5 scalar増えている（optimizerのdecision群のParameter object数45→46）。これは計算が完全同一という意味ではない。Native/Where/Amountの微分や自己模倣の重みも一部変わる。**その純粋な計算増分を、共有負荷のない旧新同条件実行で計測したわけではないためゼロとは断定しない。** ただし候補・点数・teacher回数が数倍になった記録はない。

今回直前に追加した条件別baselineは、現在走っているprocessより後のディスク上の変更であり、現在の遅延をその変更のせいにする根拠はない。mainは起動時に読み込んだコードで継続している。10/31のshadow指標/GC等の高速化ソースのmtimeは現在のmain起動より前であり、全最適化が未反映だと決めつけることもしない。

### 勾配診断だけでは説明できない

現在の末尾300 Stepでは、10 Stepごとのsource-grad診断あり30件が15.411秒、なし270件が13.385秒。Actualはそれぞれ4.831/4.821秒とほぼ同じ。差約2.03秒/診断Stepは、平均すると概ね0.20秒/Stepの規模である。

frameや行動を固定したablationではないが、診断を省くことだけで約9.39秒の増加がなくなるという根拠にはならない。compact logにforward/loss/backwardの全内訳はないため、残差9.02秒を特定関数へ推測で割り振らない。

## 4. 表示外の入力読込と一時的なRAM待ち

同一step範囲の、前Step cleanup後～次Step data load後（Episode境界除外99件）は旧1.9945秒、現2.3703秒。これは上表の表示Step時間の**外側**。同じinterval同士で足すと、通常の処理間隔は旧約6.21秒、現在約16.09秒である。Episode末Validation・plotはさらに別。

前回の内容検証付きparsed disk cache試作はこの区間を短くする候補であり、表示13.53秒をそのまま3秒へする対策ではない。本番へはまだ導入していない。

現在logには過去に5件のHostAdmissionWaitがあり、available 1,800～7,729 MiBで必要8,192 MiBを下回っていた。これは一時的な長いStepの要因。今回の観測時はMemAvailable約26 GiB、vmstatのsi/so=0、CPU idleは約80～89%だった。swap空き0だけから現在もswap転送が主因とは判断しない。

## 5. 対処の優先順位

1. **旧実績と同等のGPU計算資源を確保して比較する。** 空きVRAMだけでなく他jobのSM activityを見る。他のprocessを勝手に止めない。旧新の初期重み・frame・RNG・deviceを揃えた比較で、コード固有の増分を測る。
2. その条件で次回起動時のActual実行先をGPUへ固定する運用を検討する。`--sparsepcgc_device cuda --sparsepcgc_auto_cpu_fallback false`は既存の実行設定であり、SparsePCGC本体の改造ではない。ただしVRAM不足時には待つため、資源確保なしに付ければ速くなるという指示ではない。CPU/GPUで8 bits差のあった前回検証もあり、再現性のためにも実行先を揃える。
3. Network側のforward/loss/2回のbackward/診断/clipを、次回の短い計測または専用環境で分離する。現在の共有GPUへ追加モデルを立てて競合を増やす試験は行わない。
4. 入力parsed cacheと不要な診断の抑制は、別の改善として数値一致を確認して採用する。Actual間引き、点群削減、低精度化、学習停止で3～5秒を作る方針は採らない。

**今回の確定結果は「旧3～5秒は実在し、現在は計算資源・Actualの実行先が違い、Actual以外も大きく遅延している」まで。GPU共有が増加分の何秒か、コードだけを同じ環境で戻せば何秒になるか、GPUへ戻すだけで3～5秒を再現できるかは未実測。**

## 6. 成果物

`/data/maejima/log/independent_experiments/20261103_speed_regression_0605hqh3/`:

- `summary_final.json`: 凍結した比較区間、同一frame照合、CSV時間、GPU観測の最終集計。
- `log_comparison/before/` / `current/`: audit JSONL snapshot、設定、admission/fallback event。
- `gpu_pmon.txt` / `gpu_state.txt` / `vmstat.txt`: 読み取り観測の生データ。
- `source_before.json`、`git_status.txt`: 作業開始時の保護記録。

追加scriptは`experiments/audit_live_speed_regression.py`のみ。sourceを変更せずログ比較を再現する。既存ログは継続中のため、最終数値は上記凍結snapshotの範囲を基準とする。`summary_final.json`は最初のsnapshotのwindowを維持したまま、matched CSVとpmonを追加したもので、継続logから集計windowをずらしていない。
