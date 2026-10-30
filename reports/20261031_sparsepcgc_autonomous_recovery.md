# SparsePCGC自律探索のNo-op縮退と処理時間の調査・修正

2026-10-31。対象は `--test_route true` のSparsePCGC dense lossy（AE=0、SR=2、vs=1、pq=1）。Falseルート、G-PCC、OctAttention、native codecソースは変更対象から除外した。

## 原因を確認したログ

最新の通常訓練ログは `/data/maejima/log/20261028/MyNetwork_train/SparsePCGC/214015.txt`。Episode番号は `TrainingWindow: episode=` から復元した。旧Den6用の未接続フィールドに表示されるゼロを、実勾配ゼロの証拠には使っていない。

| Episode | 実測Step数 | 圧縮損失平均 C [%] | 負のCのStep数 | 3操作すべてゼロのStep数 |
|---|---:|---:|---:|---:|
| 1 | 100 | 7.7236 | 0 | 0 |
| 2–10 | 900 | 2.2212 | 7 | 5 |
| 11–30 | 2,000 | 0.5518 | 13 | 71 |
| 31–60 | 3,000 | 0.1688 | 14 | 206 |
| 61–95 | 3,500 | 0.0377 | 8 | 1,780 |
| 96–131途中 | 3,594 | 0.0190 | 0 | 3,594 |

この記録では最良の共同planも約−0.064%。「−4%の行動を発見したが保持できなかった」という証拠はない。codecに対して有効な位置をほとんど発見できず、探索した編集の多くが不利なため、AmountがNo-opを学ぶことが主因だった。

元の合法ランダムpoolに対する独立32共同planのActual比較でも、最良は−0.062683%。元のden6 anchorとの候補一致はlongdress 1051でAdd 28/766、Prune 19/766、Adjust 0/383だった。この有限実験は探索不可能の証明ではないが、単にAmountの候補数やnoiseを増やすだけでは不十分であることを示す。

## 原因の切り分け

1. SparsePCGC dense SR=2では、元点群の微小編集すべてが符号化される粗い占有集合を変えるわけではない。実際のdownscale座標が `unique(raw // 4)` になることをnative encoderの出力と照合した。単なる点数・ノード数削減はActual削減量を代替しない。
2. 元のTrue poolは合法な空間候補を無順位に保持していたが、Actorにはnativeの占有確率が渡らなかった。候補ごとの価値が非常に疎な状況で、共同planの単一報酬から良い位置を発見する必要があった。
3. AmountとWhereへ同じ探索planのコストを一括で伝えていたため、悪いWhereの結果が「そのAmountは悪い」とも学習された。3操作ゼロへの収束はこの条件では合理的だが、目的とする探索ではない。
4. Native確率入力・探索・自己模倣などを含む中間版ではseed22は改善したが、seed21はNo-opのままだった。seed21のAmountを評価時だけ各0.1%に固定するとCは+1.316%、Whereを初期重みへ戻すと+19.506%。Whereの学習は起きていたが、まだ有利な編集には届いていなかった。Amountだけの問題とは判断しなかった。
5. 別件として、nativeがすでに加算したpoint-count side informationをwrapperでも再加算していた。GTの物理stream 38,272 bitsに対しnative logical値は38,280 bits、旧wrapperは38,288 bits。古いGT cacheとの組合せはNo-opでも約+0.02%を生んだ。これは−4%未達の主因ではない。Trueの会計とcache識別だけを修正した。

## 実装したTrue専用の経路

| 対象 | 変更内容 | 制限・意味 |
|---|---|---|
| Native context観測 | 初回GT encodeで最初の符号化階層の8 stageを観測し、実coderと同じ16-bit整数CDFの確率を保存 | codecの学習・stream生成を変更しない。編集後のActual候補教師とは呼ばない |
| 候補生成 | 符号化占有を変え得るsingleton-cell Pruneと、隣接空cellへのAdd/Adjustを追加。符号長変化の正負両方を含むbandから採取し、空間ランダム候補も保持 | den6順位・Amount・完成planは参照しない。Adjustは各軸1voxel以内、bbox保護を維持 |
| Actor | 全3操作で共有するnative-bit選好を1個の学習パラメータとして追加。初期値0。既存Actorのoptimizer・clip・checkpointの対象に含める | native gainを固定係数で最終順位へ足さない。固定context gainは観測特徴で、Actual効果の保証ではない |
| Where探索 | 共有選好からGaussian係数をサンプルし、条件付き候補選択と合わせて正しいlog probabilityを使用 | 共有Gaussianのlog probabilityは共同planに一度だけ加える。係数のsampleをpathwise教師にしない |
| Amount/Where credit | 探索中、半分は現在のclean Whereで3操作Amountを同時評価、半分はAmountをサンプルしたまま3操作Whereを探索 | 操作種類をAddだけ等へ切り替えない。各Stepの実行は常に3操作の共同plan。clean試行の位置選択は決定論的 |
| Amount探索 | 各操作15個の正値bin＋No-opと連続Fine。探索期間には25%を上限とする一様混合を使用 | Fineが低値へ縮退しても絶対的な比率を再探索できる。混合分布の密度を計算する |
| Actual自己模倣 | 同一GT・codec・幾何定義で実測済みの `C<0` かつ `3C+150 max(G−τ,0)<0` の完全共同planだけ保存し、Amountと選択prefixを学習 | 候補単体へ共同planの改善量を割り当てない。現在のpoolに完全には存在しない教師は使用しない |
| No-opからの回復 | 探索終了後、学習中に全操作No-opとなった場合だけ既定10%で共同再探索 | evalでは無効。通常の探索noiseは既存scheduleで減衰する |
| ログ | Trueは実際に使用する共同方策・自己模倣・時間のcompact auditを出力 | 旧Den6の未接続診断値を現行学習信号と混同しない |

圧縮・幾何の損失式、重み `w_com=3 / cp_lambda_geom=150`、GeometryのAdd plane-fit、Actual encode頻度、各操作最大3%は変更していない。GeometryのAdd plane-fitは追加点の平均誤差なので、少数Addにしても損失が点数比例では下がらない。この既存目的自体の変更で学習曲線を操作していない。

主な実装は `autonomous_codec_context.py`、`autonomous_context_proposals.py`、`autonomous_actual_replay.py` と `structure_actuator.py` のTrue分岐。Native contextを読み込むcacheは入力voxel SHA・codec checkpoint fingerprintで識別し、既存cacheは上書きしない。

## Actual oracleとNetworkの結果を分ける

Native観測を使う独立oracleではC−4.619%も存在したが、完全な訓練GeometryはG=0.043372でden6参考G=0.033875より悪かった。このplanを品質維持成功とは扱わない。同じ実験のPrune-only候補ではC−3.699%、G=0.003006であり、品質を維持できる負のCの行動が存在した。これらはNetworkの学習結果ではない。

以下のNetwork検証は元の `train.py` のforward、loss、backward、Adamを独立processで実行した。訓練は各系列1フレーム、計4フレームを反復し、Validationは別のlongdress 1151 / loot 1100 / redandblack 1550 / soldier 0636。通常訓練の100 Step/Episode設定は変更せず、独立probeだけ4 Step/Episodeにした。Validationは探索なし、各点群全体をActual encodeし、元のGeometry lossを計算した。

| 方式 | seed | 訓練Step | 固定Validation C [%] | 固定Validation G |
|---|---:|---:|---:|---:|
| 元の学習済み130.pth・旧候補方策、独立8更新で再確認 | 22 | 8 | 未測定（訓練8件は全No-op） | 未測定（訓練G=0） |
| 共有選好・credit分離前の中間修正 | 21 | 64 | 0 | 0 |
| 共有選好・credit分離前の中間修正 | 22 | 64 | −5.107219 | 0.01466747 |
| 共有選好＋credit分離＋Actual自己模倣・学習途中 | 21 | 64 | +3.259829 | 0.06458925 |
| 共有選好＋credit分離＋Actual自己模倣 | 21 | 128 | −4.031746 | 0.02983207 |
| 共有選好＋credit分離＋Actual自己模倣 | 22 | 128 | −3.771274 | 0.00220380 |

上記128 Stepは探索multiplierが1の初期区間であり、seed21の直近32 Stepの探索込み訓練C平均は+18.4105%。表の負値はノイズを除いたValidationの値である。探索込み訓練平均を−4%と報告しない。seed21の64→128 Stepは同じ学習系列の重みを評価したもので、noise低下だけでは説明できない改善を確認した。

seed21の共有Native選好は0→0.006936に更新された。評価時にこの1パラメータだけ0へ戻すとC=+0.088684%、G=0.02982483へ悪化した。Amountだけ初期重みへ戻すと全4件No-op、C=G=0。初期モデル全体もNo-opだった。固定Native順位を初期状態から適用した結果ではなく、選好とAmountの学習が最終編集へ効いている。

### 同じValidationフレームに対するden6 anchorの再符号化

既存のimmutable anchorを読み、入力SHA・codec設定・最終voxel集合hashを照合して再実行した。旧cacheは上書きしない。Gは同じplane-fit関数を含めて再構成した。圧縮は今回のnative codecで再encodeした値。

| フレーム | den6 C [%] | seed21 C [%] | seed22 C [%] | den6 G | seed21 G | seed22 G |
|---|---:|---:|---:|---:|---:|---:|
| longdress 1151 | −3.670886 | −3.869801 | −3.562387 | 0.03403937 | 0.02997818 | 0.00217748 |
| loot 1100 | −3.905250 | −4.139991 | −3.883910 | 0.03214304 | 0.02929924 | 0.00218404 |
| redandblack 1550 | −3.709949 | −4.403223 | −4.215852 | 0.02924097 | 0.03030121 | 0.00225829 |
| soldier 0636 | −3.131929 | −3.713969 | −3.422949 | 0.03806394 | 0.02974966 | 0.00219540 |
| 平均 | −3.604504 | −4.031746 | −3.771274 | 0.03337183 | 0.02983207 | 0.00220380 |

**平均では両seedともC・Gが改善したが、各フレームの完全な非劣化は未達。** seed21のredandblackはGが約0.00106悪化し、seed22のlongdress/lootはCが約0.109/0.021ポイント悪化した。den6を全条件で優越した、という結論にはしない。既存のValidation基準によるbest昇格条件を緩和せず、元のbest checkpointを置き換えていない。

### 探索終了状態の追加16 Step

128 Stepのseed21重み・方策memoryを読み込み、探索multiplier=0で16 Step更新した。全16件のCは負、訓練平均−3.857041%。No-op縮退は起きなかった。ただしValidationはC−3.570388%、G=0.03072186へ悪化した。この試験は**Adamを新規作成したwarm start**であり、通常のoptimizer継続ではない。またpoolのphaseも新規runのものへ変わった。これだけから長期訓練の退行原因を断定せず、性能維持を解決済みともしない。長期ではValidationの最良時点の保存と追加検証が必要である。

## 勾配と実更新の確認

全2 seed×128 Stepの保存済みoptimizer直前勾配はfiniteだった。以下はseed21のStep64。L2/RMSはpolicy-only勾配置換・role clip後の合成勾配であり、Actual単独勾配とは呼ばない。

| モジュール | L2 | RMS | そのStepの実更新L2 |
|---|---:|---:|---:|
| Candidate Actor | 0.743207 | 0.043643 | 0.00039234 |
| Proposal | 0.669062 | 0.039357 | 0.00035503 |
| Add Where | 0.259896 | 0.001972 | 0.00200534 |
| Prune Where | 0.891268 | 0.010045 | 0.00190386 |
| Adjust Where | 0.371611 | 0.002820 | 0.00243437 |
| Add Fine Amount | 0.622960 | 0.082513 | 0.00002339 |
| Prune Fine Amount | 0.353575 | 0.046832 | 0.00003066 |
| Adjust Fine Amount | 0.697786 | 0.092424 | 0.00001509 |
| Coarse Amount | 1.000000 | 0.014605 | 0.00008633 |

独立source-gradient観測も初期8 Stepと探索終了状態16 Stepで実行した。探索終了状態では決定論的方策の直接REINFORCE勾配は0になる一方、Actual実測済み共同planの自己模倣からWhere/Amountへ勾配が到達した。例として追加Step7の自己模倣由来clip前L2はAdd/Prune/Adjust Where=0.016865/0.007833/0.019778、Fine Amount=16.0241/2.26783/10.7167、Coarse Amount=30.4596。Fine groupは19.4103→1、Coarse groupは30.4596→1へ既存clipを適用し、Where groupは0.027147のままだった。ノルムを同じ値へ増幅して揃えたものではない。

初期Where探索試行でAmountの直接勾配が0になるのはcredit分離の仕様で、別のclean-Where試行では3操作Amountを同時更新する。自己模倣成立時にはその経路から両方を更新する。全source・RMS・cosine・clip前後・実更新は独立実験のJSON/PTHに保存した。

source観測は追加backwardを使うため、速度比較から除外した。観測なしの同seed試験との最初8 StepのCは完全一致したが、3 Stepで最終voxel hashが異なり、G最大差は約1.27e−6だった。GPU演算順序等の影響を排除できないため、診断による全パラメータ・全voxelの完全一致とは報告しない。

## 時間計測と再現条件

候補の再生成・Python object化・特徴対応付けが大きかった。Native候補の整数配列cache、poolの損失なし圧縮RAM cache、同一reservoirを保持する期間の2→8巡回、明示OMPを尊重するTrue専用CPU並列上限、compactログを追加した。凍結特徴や候補を再利用しても学習済みlogitは再計算する。

Native context入力を追加した最初の実装と候補cache最適化後を、同seed22・64 Step・同4入力・同codecで比較した。Step33–64のwall平均は9.8999→5.3927秒（約45.5%短縮）。これはNative入力追加前の元コード全体との対照試験ではない。最初の8 planとCは一致。reservoir保持期間を変えた後の全64 Stepが同一軌跡だとは主張しない。Actual encode頻度や損失を省略した高速化ではない。

最終の共有選好版128 Stepでは、最後32 Stepのwall平均はseed21=3.9998秒、seed22=3.6775秒。ただし前述の共有machine上の測定であり、この差のすべてをコード最適化の効果とみなさない。

pool4入力の圧縮済み合計は約22.4MB。通常の100入力では384MiBのcacheに収まらず巡回missする可能性があるため、Trueの既定上限を768MiBへ設定した。これは上限であり、実使用量は入力・poolの大きさによる。既定値変更自体の100フレーム実走計測は未実施。

計測環境はmyOA Python、native codecはsparsepcgc Python、RTX A6000、`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`。速度計測ではsource-gradient probeとgrad auditを無効にした。独立probeのNative CPU trim間隔は1、通常設定の16は変更していない。共有machineなのでwall timeは他processの影響も受ける。初回frame/phaseのcache構築はなお重く、全Stepを1秒未満にはできていない。通常のsource-gradient probeは既定で10 Stepごとの追加backwardを行うので、必要な診断期間以外は `--heuristic_guidance_online_source_grad_probe false` で省略できる。

## 保護・再現・未検証事項

成果物rootは `/data/maejima/log/independent_experiments/20261031_autonomous_collapse_9x96rlck`。開始時の121 source、Git状態・差分・HEADを `before/` 等へ保存した。継続中にHEADが16bd91f→21defa7になったため、最終Git差分だけで今回の全作業量を判断せず、開始snapshotとのSHA比較も使用する。

Falseの既存args（3 dataset、明示Falseと既定）、9 pool、RNG、native workerの6組のcommand・codec fingerprintが開始snapshotと一致。通常のcheckpoint・ログ・codecソースを変更せず、既存processにsignalは送っていない。停止した独立probeがある場合も独立実験として失敗を記録し、成功結果へ数えない。

既存関連テストと追加テストを合わせて261件合格。Native bit会計、整数CDF、poolの合法性・符号付き入力、Amount混合密度、完全共同plan自己模倣、Actorの中立初期値と実更新、False/G-PCC分岐、既存forward/勾配/voxel実行契約を確認した。

主な成果物は `research_summary.json`、`supplementary_summary.json`、`latest_log_audit_with_episodes.json`、`heldout_den6_reference_v3/summary.json`、各runの `steps.jsonl` / `step_*.pth` / `final_validation*.json` / `integrity.json`。初期・後半のsource勾配は `shared_s21_initial_gradients/` と `shared_s21_late_phase/`。再集計は `experiments/summarize_autonomous_collapse.py` を使い、出力先が既存なら停止する。最終sourceは `final_source/`、開始時からの差分は `task_source_changes.patch`、保護確認は `final_integrity.json` に保存した。

失敗記録も保持した。`mature_before`は独立probeのRAM待機中に中断、最初のcontext試験は設定listの検査で停止して修正後に再実行した。den6 comparatorの最初2回はASCII PLYのreader、旧anchorのschema/identifier互換性で停止し、最終 `_v3` で4件すべて入力・voxel hashを検証して再符号化した。これらの途中結果を完走結果へ混ぜていない。

通常訓練の382 Episode、100フレーム、広いseed集合、最終復号点群のD1/D2・BD-rateは今回の短期検証では未確認。表のGは編集直後の点群とGTの既存訓練lossであり、復号後PSNRではない。長期収束・すべてのdatasetでの−4%を保証しない。

Trueは学習パラメータと入力特徴の意味が変わるため、旧True checkpointを黙って再開しない。runtimeに `native_cdf_context_shared_actor_v2` を記録して互換性を確認する。新規実験はmyOA環境から次で開始する。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B train.py \
  --compression SparsePCGC --test_route true --more_training false
```

Falseの従来学習を選ぶ指定は引き続き `--test_route false`。
