SparsePCGC True経路：60 Episodeの広域探索から実測良好planの周辺探索へ

2026-11-04。開始時のGit状態はclean。変更前のtrain.py/test.py/models全136ファイルとSHA256を独立ディレクトリへ保存してから編集した。

実装した訓練計画は以下。対象は`SparsePCGC / test_route=true / test_route_learned_exploration=true`だけで、新しい`test_route_phased_exploration`は既定True。

| 時期 | 動作 |
|---|---|
| Episode 1〜60 | 変更前と同じ広いAmount/Where samplingと自己模倣。新しい近傍prior・探索幅収縮は適用しない |
| Episode 61〜180 | 同じframe・GT hash・codec・Geometry設定で自分が過去に実行した良いplanの周辺を、徐々に選びやすくする。5つの探索幅も連続的に縮める |
| Episode 181以降 | 小さな探索・大域Amountの低確率試行・Actual方策勾配を残し、より良いplanが見つかれば更新する |

良い参照planは既存の自己模倣memoryに保存された、Actual C<0かつ`3*C + 150*max(G-tau,0)<0`の共同行動。同じframeの**過去**の実測だけを使い、den6研究で得た完成plan・既知の最適Amountは読み込まない。Actual教師の意味・損失重み・Codecは変更していない。これは既存の重み付きRD条件であって、D1/D2無劣化を新たに保証する条件ではない。

参照が未発見、fingerprint不一致、候補IDが現在のpoolに存在しない場合、そのframeでは広い探索を継続する。参照があるからといって最終planをそのまま置き換えたり、一定損失へ代入したりしない。61 Episode以降に見つけたさらに良いplanも引き続き参照候補となる。

近傍の定義と勾配

- Amount：過去のcoarse binとFineを基準に、実行要求割合のlog距離に応じた有限のsoft priorを加える。Fine平均も最大49%だけ参照へ寄せるため、Network側の平均への微分は少なくとも51%残る。自己模倣はpriorを加える前のNetwork logits・平均を学習する。
- Where：過去のattempted prefix候補を残しやすくし、一部を他候補へ入れ替える。候補除外や順位固定はしない。これは選択集合の近傍であり、点群内で別候補までの物理距離を追加制限する方式ではない。現在の安全maskは維持する。
- Whereのpriorは温度に合わせた有限oddsで与える。固定score加算を小温度で割って過去prefixをほぼ強制する設計にはしていない。
- Add/Prune/Adjustはこれまで通り同じStepで同時に決定・編集し、完成planのActualとGeometryから学習する。
- categorical＋Fine Normal＋uniformの混合密度、およびWhereのPlackett–Luce確率は、実際にsamplingした分布を使う。sampling後のhard置換による不正な方策creditは加えない。
- current frameの観測を参照へ保存するのはActual評価後。今回の行動が自分自身の比較教師になることはない。
- broad期間末の補助reservoirを以後維持し、探索中の良い候補IDが定期的なpool入替で失われるのを抑える。現pool内の代替候補は残す。以前のphaseでしか使えない参照を無理に投影せず、利用できない場合は広域探索へ戻る。

探索幅の連続的な収縮

既定100 Step/Episodeでは、global_step 6000の時点の新しいprior強度は0。その後に滑らかに上がり、180 Episode終了時に最終水準へ到達する。

`u = clamp((global_step / fixed_episode_steps - 60) / 120, 0, 1)`

`a = 0.98 * (3*u**2 - 2*u**3)`

正の探索幅は`exp((1-a)*log(learned_width) + a*log(local_floor))`へ移行する。明示的なnoise=0設定は0のまま維持する。終点でもa=0.98なので学習幅への勾配が残り、samplingの打切り・log_probの一斉ゼロ化はない。

| 制御 | local floor | 生の学習幅を初期値のままにした場合の最終値 |
|---|---:|---:|
| Native係数標準偏差 | 0.05 | 0.05458 |
| Where温度 | 0.10 | 約0.1047 |
| Amount温度 | 0.15 | 約0.1580 |
| Fine対数標準偏差 | 0.03 | 約0.0314 |
| Amount一様混合率 | 0.005 | 約0.00541 |

この表は設定に基づく値で、実訓練後の計測値ではない。実際は学習幅と有効な自己発見参照の有無に依存する。

設定・出力

[args.py](/home/maejima/MasterEx/myNet/models/utils/config/args.py:3625)の3項目を追加した。

```bash
python train.py --test_route true \
  --test_route_learned_exploration true \
  --test_route_phased_exploration true \
  --test_route_broad_search_episodes 60 \
  --test_route_local_search_episodes 120
```

`test_route_phased_exploration=false`で今回の段階移行だけを外し、変更前の学習幅のみのTrue経路へ戻せる。`test_route_learned_exploration=false`では今回の機能は無効となり、既存annealed比較経路を維持する。`test_route=false`には新しいprior・scheduleを適用しない。

起動時`AutonomousLocalSearch`、各Stepの`AutonomousLearningAudit.local_search`に段階・強度・参照成立理由・参照C/G・参照prefix数を記録する。`learned_exploration`にはそのStepで実際に使った有効幅が出るため、内部の生parameter値と区別して監査できる。

変更したファイル

| ファイル | 役割 |
|---|---|
| models/utils/training/autonomous_local_search.py（新規） | 段階、同一条件の参照照合、幅収縮、soft Amount/Where prior |
| models/modules/structure_actuator.py | 実samplingとlikelihoodへ同じ近傍分布を接続。既存候補ID集計を再利用 |
| models/network.py | 復元済みを含む既存実測memoryへの読み取り参照をactuatorへ渡す |
| models/utils/training/autonomous_joint_learning.py | Amount混合分布の整合したsampling/likelihood。既存memory keyを共有helperへ移動 |
| models/utils/training/sparsepcgc_autonomous_policy.py | 設定検証と局所探索中のreservoir維持 |
| models/utils/config/args.py / train.py / autonomous_audit.py | 設定と監査出力 |
| tests/test_autonomous_local_search.py（新規） | 分布、境界、教師条件、勾配・更新、評価分離の検証 |
| experiments/verify_phased_exploration.py（新規） | 変更前snapshotとの比較とCPU sampling診断 |

計算量を増やす追加encodeはない。参照planは既存の圧縮保存memoryから読み、候補IDの既存集計を再利用する。今回の追加CPU処理を含む実際のStep時間は未計測なので、速度が不変だとは断定しない。

検証結果

- 関連145テスト合格。Exact-plan適用・既存loss、共同Amount、conditional baseline、codec context、False/geometry routes、head chunking、新規近傍分布を含む。
- 広域期間のsamplingについて、変更前snapshotとの96条件比較で選択、Fine sample、log_prob、logits、乱数状態が一致。
- 変更前後のloss/backwardを同じ解析用fixtureで比較し、loss、勾配、runtime memory key・baselineが一致。
- late phaseでもWhere・各Fine Amount・coarse Amount・Gateの非ゼロ有限勾配とAdam更新を確認。探索幅parameterにも更新が残ることを確認。
- Whereの全順列確率を列挙して総和1、期待score勾配0を確認。Amountは実際のcategorical/Normal/uniform混合密度と比較。
- 参照なし・他frame/GT/codec・不適合prefix・非有益教師のときに縮小しないこと、明示的noise=0、no-op支持、評価時に参照・乱数を使わないことを確認。
- Falseの3 datasetの既存args、9 pool/RNG条件、12 worker command/fingerprintが変更前と一致。
- 変更したPython 10ファイルの構文解析、git diff --check合格。native SparsePCGC Python 53ファイルはSHA256変更0。

Whereは等logit・32候補・参照4候補、Amountは6 bin中の0.1/0.2/0.3%を近傍（参照0.2%）とする**人工sampling診断**では以下だった。圧縮性能の実験ではなく、どの程度分布が局所化し別候補も選ぶかの確認である。各段階400回、Amountは3操作同時。

| Episode先頭 | Amount近傍bin選択率 | Where参照候補overlap | Where選択集合が参照と異なる割合 |
|---|---:|---:|---:|
| 60 | 50.2% | 12.8% | 100.0% |
| 100 | 57.5% | 31.8% | 100.0% |
| 120 | 66.7% | 54.4% | 96.3% |
| 150 | 94.3% | 83.8% | 54.5% |
| 180 | 99.7% | 91.6% | 30.5% |

この診断の参照圧縮値はテストfixtureでありActual結果ではない。実際の点群の圧縮率や学習済みNetworkの選択分布へ、この割合を移して解釈しない。

未検証事項・運用

長期訓練、固定Validationでの新方式のActual圧縮率・Geometry・D1/D2は未実施。−5%到達、単調低下、無劣化を保証する結果は得ていない。今回の確認は、要求した行動分布・段階移行・勾配経路が実装され、既存分岐と整合する範囲。離散選択・frame差・新bestの発見による損失の上下は残る。

現在のmain PID766550と既存workerは変更・再起動していない。今回の設定は次回起動から有効。11/02起動の現processには11/03の既存条件別baseline修正も未反映なので、広域期間の不変性は**今回の作業開始時ディスク上の実装**との比較であり、稼働中processと全学習軌跡が一致する主張ではない。

スケジュールは現在runのglobal_train_stepに基づく。既存のmore_trainingはglobal_train_stepを0から始めるwarm startなので、その場合もこのrunの最初の60 Episodeが広域探索になる。既存の最適化・step復元方式を今回変更したものではない。

成果物：`/data/maejima/log/independent_experiments/20261104_phased_exploration_e4p1jev2/`。before/、source_before.json、codec_before.json、false_regression_verified.json、sampling_diagnostic_final.json/log、tests_verified.log、integrity_verified.json、patch.diff。新規訓練・GPU実験・Codec再encodeは実施せず、既存checkpointとログへの書込みも行っていない。
