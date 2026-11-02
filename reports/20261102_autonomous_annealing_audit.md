# True経路の90 Episode前後の急低下と後半学習の監査

2026-11-02。対象は `/data/maejima/log/20261031/MyNetwork_train/SparsePCGC/134351.txt` と同prefixのEpisode CSV、frame manifest、既存checkpoint。全16,880 Step、完了168 Episodeと169 Episodeの80 Stepを読み取った。新規訓練・再符号化は行っていない。

**急低下の主因は、悪化の大きい探索行動がノイズ減衰によって消えること。その後は95.5 EpisodeでActual方策学習が停止し、過去の成功行動の自己模倣が続く。事前検証した固定den6 planへ突然置換する経路は確認されなかったが、探索停止後の学習には修正すべき制限がある。**

訓練コード・args・checkpoint・既存ログ・processは変更していない。追加は独立集計コード、本レポート、新規結果ディレクトリだけ。対象のtrain.py、args.py、structure_actuator.py、探索設定・replay実装は起動時commit `0e232c5` と現在で内容一致を確認した。前回の速度最適化を急低下の原因とは判断しない。

## 1. 観測値

ここでCは実測 `100*(MINE_bits/GT_bits-1)` [%]。CSVの別列 `compression` はsurrogate予測なので混同しない。

| Episode | 訓練Actual C [%] | 固定Validation C [%] | 訓練G | Validation G | 探索倍率の最小～最大 |
|---|---:|---:|---:|---:|---:|
| 40 | +11.772905 | −5.172969 | 0.02419634 | 0.00715663 | 約0.969～0.986 |
| 80 | +6.611876 | −4.894455 | 0.01216217 | 0.00904944 | 0.270681～0.287958 |
| 85 | +1.389509 | −4.714495 | 0.01219365 | 0.01035858 | 0.172330～0.195452 |
| 89 | −2.454039 | −4.656029 | 0.01204599 | 0.01165688 | 0.070015～0.094500 |
| 90 | −4.753100 | −4.809790 | 0.01133143 | 0.01056348 | 0.047898～0.069779 |
| 95 | −5.059226 | −4.945074 | 0.01000778 | 0.01003885 | 0.000065～0.001491 |
| 96 | −5.131529 | −5.003797 | 0.00985711 | 0.00982185 | 0～0.000061 |
| 100 | −5.234731 | −5.018256 | 0.00948878 | 0.00947888 | 0 |
| 150 | −5.025631 | −4.414201 | 0.00888886 | 0.00915771 | 0 |
| 168 | −5.208770 | −3.838467 | 0.00910290 | 0.01058403 | 0 |

全完了Episodeで同じ100フレーム・同じ順序。各Episodeで100回のoptimizer更新が記録されている。Validation signatureも固定。ただしValidationは **longdress 1151～1155の5フレームだけ** で、訓練とは非重複だが4系列全体の汎化評価には不足する。

40 Episodeですでに探索なしのValidationは約−5.17%。90 Episodeで初めて良い方策が現れた、という解釈は支持されない。一方、後半の訓練Cは約−5でも固定Validationは悪化しており、性能が良い状態で収束したとも言えない。Geometryも100→168でValidation側が悪化している。

## 2. 急低下の原因：共同Whereノイズが符号長の選好を反転する

[structure_actuator.py](/home/maejima/MasterEx/myNet/models/modules/structure_actuator.py:5747) のTrue経路は、学習パラメータ `native_context_preference` にGaussianを加え、3操作に共通の係数を使う。おおむね半数はWhereを決定論的にしてAmountを探索し、残りで係数と候補順位を探索する。

`sampled_preference ~ Normal(learned_preference, max(4*exploration_multiplier, 1e-6))`

候補scoreにはActor・Proposalの学習scoreに加え、この係数×GTのNative CDF由来の固定context符号量特徴が入る。係数が負なら、この特徴に対する選好が反転する。固定context特徴は編集後Actualの代用ではなく、最終評価は全点群Actualで行われている。

| Episode | Where係数が負：Step数 / 平均C | Whereノイズなし：Step数 / 平均C | Where係数が非負：Step数 / 平均C |
|---|---:|---:|---:|
| 80 | 19 / **+48.908869%** | 47 / −4.169047% | 34 / −2.121639% |
| 85 | 9 / **+52.442028%** | 52 / −4.116798% | 39 / −3.050099% |
| 89 | 4 / **+43.868774%** | 50 / −4.819470% | 46 / −3.910991% |
| 90 | 0 / 該当なし | 52 / −4.835710% | 48 / −4.663606% |

**89→90で、大幅に悪化する少数行動がなくなり、全体平均が急低下した。** 89 Episodeは負係数の4件を除いた96件の平均が約−4.38%であり、4件が平均を約1.93ポイント押し上げていた。連続的なnoise減衰でも、負係数の発生、離散bin、top-k順位は非線形に変わるため、損失曲線まで滑らかになる保証はない。

この表は既存ログの条件別集計であり、同一重み・同一frame・同一Amountを固定した因果ablationではない。悪化の全量を係数の符号だけに帰属するものではない。ただし急低下の時期、実行行動、Actual値が対応している。

## 3. 95.5 Episode後は新しいActual方策勾配が止まる

ログに明記された設定は382 Episode×100 Step、探索25%、最大強度保持40%、末尾smooth tail25%。探索終了は **global_step=9550、95.5 Episode**。末尾は滑らかでも、その後の分岐は完全な決定論的選択になる。

- [sample_joint_amounts](/home/maejima/MasterEx/myNet/models/utils/training/autonomous_joint_learning.py:16)：phase=0でargmax、log_probを`logits.sum()*0`にする。
- [where_prefix_log_prob](/home/maejima/MasterEx/myNet/models/utils/training/autonomous_joint_learning.py:87)：phase<=0で勾配ゼロの値を返す。
- [joint_actual_policy_loss](/home/maejima/MasterEx/myNet/models/utils/training/autonomous_joint_learning.py:104)：現在Actual/Geometryの方策項は、このlog_probに掛かるためActorへの勾配がゼロになる。
- [train.py](/home/maejima/MasterEx/myNet/train.py:2660)：hard選択のheadへ不対応なSTE勾配を入れないため、decision勾配を保存・復元している。surrogateの勾配がこの停止を補う構造ではない。

実ログのsource勾配probeで、終了後 **733/733件** がActual圧縮方策Actor=0、Geometry方策Actor=0、**733/733件** が自己模倣Actor>0だった。Actual結果で学ぶCriticや自己模倣までゼロという意味ではない。

| Episode | Actual圧縮方策Actor L2 | Geometry方策Actor L2 | 自己模倣Actor L2 |
|---|---:|---:|---:|
| 80 | 802.665 | 7.43594 | 0.248333 |
| 95 | 895.275 | 28.8748 | 0.250138 |
| 97 | **0** | **0** | 0.247124 |
| 100 | **0** | **0** | 0.242426 |
| 167 | **0** | **0** | 0.143990 |

各Episodeの10 probe平均、clip前のsource L2。役割ごとのパラメータ数も異なり、更新量や望ましいnormとして使う数値ではない。

探索末尾には別の不連続もある。Amount温度は最低0.05、Fine sigmaは最低0.02のままphase>0を保ち、phase=0で一度にsamplingを止める。またNative係数sigmaが極小になるためscore-function勾配が大きくなり得る。実ログの**表示step=9550（zero-based global_step=9549）**でActual方策Actor L2=1,170,064.54、その次のprobeで0。clip後の更新が同じ大きさという意味ではないが、滑らかな学習経路とは言えない。

## 4. 固定planへの強制置換か

確認した経路では、事前実験の完成plan・dataset別Amountを90 Episodeで読み込む処理はない。初期native preferenceは0、ログ上の初期探索は広い。checkpoint中の値も80:0.449847、90:0.514284、100:0.602246、167:0.946152と連続して変化し、90で特定値に代入されていない。

一方、[autonomous_actual_replay.py](/home/maejima/MasterEx/myNet/models/utils/training/autonomous_actual_replay.py:17) には**同じGT/Codecで過去に自分が実行して負のRDを得た共同行動の自己模倣**がある。外部den6教師ではないが、後半にはこれが主要なActor更新源になる。97 Episode以降は毎Stepで使用されている。

保存するのは入力ごとの実測best共同行動1件。古い行動も継続して模倣し、weightは`min(-best_RD,1)`で飽和する。現在の行動より教師が良いかの比較を必須としていない。この構造が後半の退行へ寄与するかは、自己模倣を条件付きにした同条件ablationが必要で、現時点で因果確定はしない。

93～95 Episodeには全3操作のcoarse binが0.002に集中し、以後も続く。ただし実行割合はFineと競合解決によって異なり、100 EpisodeではAdd約0.2213%、Prune約0.3213%、Adjust約0.1810%。**0.002 binの選択と実行Amount固定は別**である。166→167で全100フレームの操作countは変わっており、後半に何も更新されていないわけではない。

補助poolの更新は8 Episodeごと、探索終了後はphaseを固定する。89 Episodeにはpool切替で保存行動が全100件不成立となり、自己模倣使用率0/100→90 Episodeで92/100へ戻る。一時的な教師空間の不連続はあるが、急低下の主因と確定するには分離実験が必要。

## 5. 追加で確認した保存・評価上の問題

最初のValidationはno-opでG=0。この値を相対Geometry gateの基準に固定しており、以後の正のGは基準×1.2を満たさない。CSVでは以降 `fixed_validation_geometry_or_safety_failed` が続く。`restore_best=False`、restore上限0なので、後半の退行に対して巻き戻しも行っていない。

したがって「圧縮が−5になって安定したから最適な品質のモデルが保持された」と判断できない。修正時にはgateを見た目だけ通す緩和をせず、許容するGeometry品質を事前に定義し、同じ固定評価条件で有効なモデルを選ぶ必要がある。

## 6. 推奨する修正順序（本番コードは今回は未変更）

1. **学習用の行動分布と評価用greedy選択を分離する。** 評価は常に決定論的にする。訓練は広い探索を減衰させた後も、低頻度のAdd/Prune/Adjust共同の近傍探索と少量/no-op比較を残し、新しいActual/Geometryでheadを更新できるようにする。単にargmaxのlog_probを非ゼロに見せる変更は、正しいon-policy勾配にならないため行わない。前半25%でbehavior noiseを完全ゼロにする条件を維持するなら、後半の更新には別のActual比較・適切なoff-policy教師を明示的に設計する必要がある。
2. **負方向も含む大きな共有係数ノイズから、行動の変更幅を制御した探索へ移す。** 初期の正負検証は許し、実測で繰り返し大きく劣る方向を後半まで高確率で試さない。係数を固定の正値へ縛るのではなく、Actual evidenceと方策の変化量に基づいて共同探索を調整する。sigma極小でのscore-function発散と温度下限→停止の不連続も避ける。
3. **実測自己模倣を保持しつつ、教師の優位性を確認する。** これを一括削除すると以前のno-op/0%縮退を再発する可能性がある。現在行動に対して実測RDが良い場合の教師化、品質条件・candidate-space一致、複数の非劣解の保持を検証する。best1件への恒常的な追従を避け、損失倍率だけで制御しない。
4. **品質gateとValidationを直す。** no-op G=0の相対基準問題を明示した品質基準へ置き換え、4系列を含む固定非重複ValidationでC/Gの退行を監視する。探索つき訓練平均とgreedy Validationは別々に評価する。

−5%級の行動が訓練中と探索なしValidationの双方に存在することは実測済みなので、その経路を捨てる必要はない。ただし上記変更後にも−5%を維持できる保証はない。同一初期重み・frame順・seedで、前半だけでなく探索終了前後・後半も比較し、複数seedの固定Validation C/Gと新規Actual教師の勾配・clip後更新を確認してから採用する。曲線に人工的な揺れを加えることは対策にならない。

## 7. 成果物と検証限界

- 独立集計：[audit_autonomous_annealing.py](/home/maejima/MasterEx/myNet/experiments/audit_autonomous_annealing.py)。出力先は新規ディレクトリに限定する。
- [集計JSON](/home/maejima/MasterEx/myNet/experiments/results/20261102_autonomous_annealing_audit/summary.json)：全Episode、条件別C/G、bin分布、勾配source平均、checkpoint係数。
- [診断図](/home/maejima/MasterEx/myNet/experiments/results/20261102_autonomous_annealing_audit/annealing_audit.png)：探索あり訓練と探索なしValidationを比較。
- `integrity.json`：監査前後で既存train/modelsと入力ログ・CSV等のhash一致。`launch_source_comparison.json`：重要経路の起動時commitとの一致。`probe_boundary.json`：境界の生probeと後半733件の集計。

Episode実測C平均は元CSVと絶対差1e-7未満で一致し、同一frame順も検証した。既存joint-learning / autonomous-search / codec-contextの44テスト合格。これは現在の分岐の動作確認であり、問題を修正した証拠ではない。新規訓練・Actual再encode・後半への介入実験は行っておらず、改善策の効果を実証済みとは扱わない。
