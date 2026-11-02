# 自力探索・急低下・No-op収束の再監査とWhere学習信号の修正

2026-11-03。対象はSparsePCGC / `test_route true` / `test_route_learned_exploration true`。10/31の旧方式へ戻すのではなく、今回の「自力探索を続け、実測から良い行動を学ぶ」要求を基準に調査した。既存の未コミット成果物を保存・保持した。

**95.5 Episodeでsamplingと新規方策勾配が停止する旧条件は、現在の学習型探索では回避されている。一方、最終的に滑らかに−5%へ収束することを確認済みとはしない。今回は、Whereの高分散な学習信号に対し、目的関数の期待勾配を保つ過去条件別baselineを追加した。**

## 1. 既存ログで確認できた学習

`20261102/MyNetwork_train/SparsePCGC/221459.txt`から最初の1,499 Stepを凍結保存して集計した。今回の修正は実行中processへ適用していないため、以下は全て**修正前の現在方式**の実績である。

- 全操作No-opは2/1,499 Step。現時点で「全てNo-opに逃げて0へ収束した」とする証拠はない。
- 実行した共同行動のActual C≤−5%は76回、52フレーム。最小Cは−6.293567%。単発最良値を収束性能とは扱わない。
- Native探索標準偏差は4.00から約3.98とほとんど変わらず、Where探索側の悪化が訓練平均を押し上げている。
- clean WhereでAmountを試す系列と、Whereを探索する系列がある。どちらもAdd/Prune/Adjustを同時に実行する。3操作を順番に学習させる方式ではない。

同じsignatureの固定Validationは以下だった。通常のNetwork eval/greedy経路であり、この評価のために既知の−5% planを読み込ませたものではない。

| Episode | 訓練Actual C平均 [%] | 固定Validation C [%] | Validation G |
|---:|---:|---:|---:|
| 1 | +16.571906 | 0.000000 | 0.00000000 |
| 8 | +7.276360 | −2.267848 | 0.00101416 |
| 11 | +11.008871 | −2.469567 | 0.00119246 |
| 12 | +12.575843 | −5.374092 | 0.00780307 |
| 14 | +17.214532 | −5.330115 | 0.00739168 |
| 15 | +14.525089 | −5.164658 | 0.00807789 |

**良い決定論的行動は既に学ばれているが、探索行動の平均はまだ高い。** CだけでなくGも変化している。Validationはlongdressの固定5フレームであり、全4系列の汎化や最終精度を保証しない。初期No-opのG=0から品質を保ったまま−5%になった、とは主張しない。幾何・安全gateが不合格の記録もあるため、−5%だけでcheckpointを無条件に推奨しない。

11→12 Episodeではgreedy Validationが急に変わっているが、訓練の探索倍率は1のままだった。旧95.5 Episodeのnoise終了とは異なる。AmountのargmaxやWhere順位は離散的なので、parameterが連続的に更新されても出力が段階的に変わることがある。今回、同じ固定重みで各選択を変える因果ablationまでは行っておらず、この変化を特定headだけへ断定して帰属しない。

## 2. 旧方式と現在方式の差

旧annealed方式では、38,200 Stepの25%＝9,550 Stepでphase=0となり、Amountはargmax、Where/Amountのsample log-probが0になった。これは旧ログ・コードで確認済みの、後半に新しいActual方策勾配が止まる原因である。

現在は`exploration_controls()`が存在する学習型経路で、`structure_actuator.py`が実際のsampling倍率をtraining=1 / eval=0へ切り替え、5つの学習済み探索幅を使う。Episode境界はnoise終了条件にならない。これは今回新しく直した部分ではなく、11/02の修正が有効に残っていることを再確認したもの。

候補poolの空間reservoirには、従来の25%境界でrotationを止める条件が残る。これは行動の強制再生やpolicy gradient停止とは別の制限で、有限の合法pool全体からの探索は続く。任意座標・全編集組合せを探索できるわけではない。今回は候補空間・rotationを変更していない。

## 3. 今回確認・修正した問題

Where探索では、先に共通Native係数zをNormalから引き、次に各操作の候補順位aをPlackett–Luceから選ぶ。係数の符号・大きさで配置全体の性質が大きく変わる。

従来は、同じframe/familyのbaselineを用い、

`(R - b_family) * (log p(z|state) + log p(a|state,z))`

を計算していた。`R = 3*C + 150*max(G-tau,0)`。係数zの大きな悪化まで、個別順位を作るActor/Proposal/Whereに強く混ざる。推定式の期待値は正しくても分散が大きく、有限Stepとclipの下で学習を妨げ得る。

今回、同一frame/codecの**過去**Actualから、Native係数がnegative/nonnegativeの2条件でC/GのEMAを保持する。どちらの符号を優遇するかは固定せず、それぞれの実測値だけを用いる。現在の観測はbaselineを読んだ後に保存する。

修正後は、

`(R - b_family) * log p(z|state)`

`+ (R - b_past(state, condition(z))) * log p(a|state,z)`

となる。zはWhere順位を引く前の条件なので、このbaselineをWhere scoreから引いても期待勾配を変えない。**Native Normalのscoreからは引かない**。そこへ条件別baselineを使うと、学習対象z自身に依存するbaselineになりbiasを作るためである。

Amount clean試行は従来通り。同じframeでもWhere履歴とAmount履歴を混ぜない。初めての条件は従来family baselineへfallbackする。旧runtimeに新しい履歴がなくても動作し、既存のruntime state保存の中へ履歴を保持する。新しい学習parameterやoptimizer groupは追加しない。

この修正は分散を下げるcontrol variateであり、C/Gの測定値やloss重み、−5という目標値、優先Amount、優先Where、den6教師を追加するものではない。最終planのActual評価回数、sampling/noise分布、候補数・上限、Geometry定義、clip上限、LRは維持した。新しいscalar処理・小さな履歴の管理分のoverheadはあるが、追加codec encodeや大きなTensor cacheはない。

期待勾配の一致はclip前のscore推定についての性質である。clipやAdamを通した個々の更新・その後の学習曲線を同一にする変更ではなく、履歴が成立した後の更新方向・ばらつきが変わることを意図している。

## 4. ログによる反事実baseline比較

凍結した既存Actualログの時間順を保ち、過去frame/codec/family、過去Native条件だけでbaselineを再計算した。現在のrewardや未来の観測をbaselineへ混ぜない。入力logのactive重み3/150、tau=0を確認して使用した。

| 対象 | 件数 | 従来のRD差分散 | 条件別baselineのRD差分散 | 減少 |
|---|---:|---:|---:|---:|
| Where試行全体 | 748 | 19,505.19 | 14,133.96 | 27.54% |
| global_step≥500 | 491 | 17,778.00 | 10,450.73 | 41.22% |
| 同条件の過去観測あり | 548 | 15,790.47 | 8,444.16 | 46.52% |

これは**既存実測値に基づくRD差の分散**。実parameter勾配の分散や、修正後のActual性能改善率ではない。履歴なし期間はfallbackするため、全ての期間・frameで必ず分散が下がる保証もない。

## 5. 検証

- 2段階離散方策の全actionを列挙し、補正後の期待gradientが元の期待costの厳密微分と一致することを検証した。同じ条件別baselineをNative scoreにも誤適用するとbiasが出ることもテストした。数値fixtureでありActual codec実験とは呼ばない。
- loss関数でWhere側のC/G baselineが過去同条件のみを使い、Native scoreのgradientは変わらないことを確認した。初回、別GT、clean Amount、無効flag、旧debug、履歴save/restoreも確認した。
- **実際のStructureRepairActuatorとheads**をCPUで動かし、global_step 0 / 9549 / 9550 / 38200、2 seed、annealed/learnedの16条件を確認した。明示したanalytic costでbackwardとAdam更新を行う接続試験であり、Actual圧縮の再訓練ではない。
- learned経路の9550/38200でもsampling倍率1、likelihood非ゼロ、parameter更新ありだった。9550のWhere試行例はActor L2=3.9535、Proposal=0.5690、Where合計=20.5528、更新delta L2=0.0024178。Amount試行例はAmount合計L2=5.6751、更新delta L2=0.0015423。familyに応じて他のdecisionへのActual creditが0なのは設計通り。これらを本番のActual勾配値とは扱わない。
- 今回変更前snapshotとの上記16条件で、最初の観測の編集集合・操作数・乱数・gradient/更新が全件一致。過去条件別履歴が成立してから、意図したgradient補正が有効になる。
- Falseの既存args値は3 datasetの既定/明示Falseで一致。9 poolとRNG、12組のworker command/fingerprintが一致した。G-PCC/OctAttentionのforward branchには新処理を入れていない。
- 全テスト**449件中444件合格、GPU等を要する5件skip**。CUDA_VISIBLE_DEVICESを空にしたCPU試験のみで、既存訓練とのGPU競合を追加していない。

## 6. 今の結論と未確認事項

| 問題 | 判定 |
|---|---|
| Episode 95.5で方策学習を時計だけで止める | 学習型Trueで回避されている。late-stage実モジュールで再確認 |
| den6既知planへ強制収束させる | 現在の対象経路・今回の修正では追加していない |
| 良い−5級の行動をNetworkが全く発見できない | 現ログでは支持されない。52 frame/76試行、greedy固定Validationも−5級 |
| No-opの0へ既に逃げ切っている | 現ログでは支持されない。No-op 2/1499。ただし将来のcollapseを保証して防いだわけではない |
| Whereの高分散なcredit | 条件別baselineを追加し、期待gradientの正しさとログ分散低下を確認 |
| 探索幅が最終的に十分小さくなり訓練平均も−5へ達する | **未確認**。現時点ではσ≈3.98で、収束済みではない |
| 修正後の長期Actual/Geometry・未見系列の精度 | **未検証** |

離散Amountのargmax、candidate順位、希少な大幅悪化行動により、学習による変化でも損失曲線は急に動き得る。見た目を滑らかにするためのloss補正、固定割合への誘導、特定Episodeでの−5行動への置換は行わない。今後は探索中平均とgreedy固定Validation、探索幅、No-op率、C/Gを分けて確認する必要がある。低いCだけでGeometry非劣化を認定しない。

**現在走っているprocessは旧loaded codeのまま。今回の変更は次回起動から有効になる。** 明示設定は`--test_route true --test_route_learned_exploration true --test_route_where_conditional_baseline true`。最後をFalseにすれば今回のbaseline補正だけを外した比較ができる。SparsePCGC本体、実圧縮worker、既存checkpoint、既存ログは変更していない。

## 7. ファイルと再現

- 新規`models/utils/training/autonomous_conditional_baseline.py`: past-only条件別baseline。
- `autonomous_joint_learning.py`: Where scoreだけへのcontrol variateと履歴保存。
- `structure_actuator.py`: Native scoreとWhere順位scoreを区別して保持。sampling処理は維持。
- `args.py`: True学習型専用の切替flag。
- `autonomous_audit.py`: 実際に使ったbaselineのscope/履歴有無を記録。
- `tests/test_autonomous_conditional_baseline.py`: 6件の追加検証。
- `experiments/conditional_where_log_audit.py`: 実測ログの凍結・past-only ablation。
- `experiments/conditional_where_module_probe.py`: CPU実モジュールのhorizon/gradient/update検証。

成果物: `/data/maejima/log/independent_experiments/20261103_joint_credit_vnfx_ut4/`。`before/`、`log_audit_frozen/summary.json`、`validation_snapshot.csv`、`module_before/` / `module_after/`、`module_comparison.json`、`false_regression.json`、`all_tests.txt`を参照。初期の`log_audit/` / `log_audit_final/`より、凍結1499 Stepを使う`log_audit_frozen/`を最終集計とする。
