# 幾何Octree圧縮ルート: joint候補選択への変更と実測

2026-10-16。前回の読み取り専用監査を基に実装した。

**状態: 設計上の制限の修正と短期訓練の検証は完了。ただし圧縮改善（Actual負値）、まして−40～−50%の達成は未確認。今回の短期訓練ではまだ未編集よりbitが増えており、「研究目標を解決した」とは判断しない。**

## 1. 変更範囲

- 対象はG-PCC/OctAttentionかつ `geometry_octree_joint_policy=True` の経路。
- SparsePCGC固有の候補生成・量子化・Actual encoder・Octree解析式は変更していない。共通ファイルでは幾何codec判定の分岐を追加し、SparsePCGCの計算を維持した。
- 実行中だった `python train.py --compression G-PCC`（PID 320815）は停止・再起動・checkpoint変更していない。このprocessは旧コードで動作し続ける。変更は次の新規起動から反映される。
- 全検証の出力・学習済み重み・Surrogateは `/tmp/mynet-geometry-joint-ETWeHV` に隔離した。

## 2. 「3操作を同時に考える」の実装

提案された考え方は重要である。ただし、3操作を一括出力するだけでなく、**合成した配置に対する評価**を持つ必要がある。

今回、3種類のpoolというインターフェースは維持し、独立Top-Kの合成を次へ置き換えた。

`全3操作pool → 共通Octree状態で全候補の限界コストを計算 → 全操作横断で1候補を選択 → occupied集合・親mask・必要な祖先maskを更新 → 影響する残り候補を再評価 → 1つの複合plan → Actual encode`

- Addを選ぶと、同じ周辺のPrune/Adjustの評価も変わる。選択順を固定のAdd→Prune→Adjustにしない。
- 個別点に加え、兄弟occupancyのまとめた追加・削除、subtree単位の削除候補を導入した。bundleは候補1個ではなく、実際のvoxel操作数で予算を消費する。
- 9パラメータの `geometry_joint_context` を追加。動的なrate・geometry・相互作用の特徴を、実行した複合planのActual-RDから学ぶ。元配置だけを見るActorとは別に、更新後の配置を学習に反映できる。
- Where/Amount/Gate/Fineの因子切り替えで他因子をanchorへ固定する処理を、幾何joint経路では使用しない。全操作・各因子が同じStepで変化できる。
- 完全な全組合せ探索や最適解保証ではない。計算量を抑えた条件付きjoint samplingである。

主な実装:

- [geometry_joint_policy.py](/home/maejima/MasterEx/myNet/models/utils/pointcloud/geometry_joint_policy.py)
- [geometry_joint_proposals.py](/home/maejima/MasterEx/myNet/models/utils/pointcloud/geometry_joint_proposals.py)
- [geometry_octree_rate.py](/home/maejima/MasterEx/myNet/models/utils/pointcloud/geometry_octree_rate.py)

## 3. その他の修正

### 候補生成・教師

- source配列の先頭だけを採用していた空間偏りを修正。全体から親ノードを層化抽出し、全child slot・移動先を評価する。
- 固定26近傍密度scoreから、階層的occupancy変化の符号付きproxy gainへ変更。負のgainも残す。
- 上位候補だけでなくreservoirを残し、`pool_rank` は実際のscore順へ合わせた。
- codecの実確率を完全再現したものではない。empirical costはあくまでpriorであり、Actual bitsと呼ばない。Actualと符号が合わないpriorの権限は、実測に基づいて低下させる。
- `actor_proxy_weight=0`でも残っていたproxy Critic→Actor蒸留を、幾何経路の既定で0にした。

### Octree解析

- `GeometryOctreeStructureAnalysis`だけでoccupancyパターン確率を親総数で正規化し直した。
- entropy特徴が全点1へ飽和する問題を修正。SparsePCGCが使う継承元の式は変更していない。

### Amount / Gate / Fine

- 幾何codecに無関係なSparsePCGCのconfidenceが、Gate/Fineのforwardを初期値付近に制限する処理を解除。
- Amountはno-opを含む全classをNetworkが選べる。永続的なHeuristic Amount priorは加えない。
- 既定の正binを0.05%、0.15%、0.30%、0.60%、0.90%とし、ノイズは既存scheduleに従って0へ減衰する。割合をEpisode数に応じて強制的に減らす処理は入れていない。
- 共通Amount初期化がSparsePCGC用の3%を読み、最大binを優先していた点も修正。幾何codecでは `geometry_octree_initial_total_ratio`（既定0.25%）を初期化に使う。初期化後は通常の学習パラメータである。
- 安全な既定上限0.99%は維持。幾何専用 `geometry_octree_max_edit_ratio` を追加し、より大きいbinを明示して研究できるようにした。大きい上限での性能・品質は今回未検証。

### 幾何RD・監査

qs=2では、編集前からChamfer約2.18が存在した。その固定歪みを操作の歪みとして学ばせないよう、同じ入力・同じgridのno-edit Chamferを測定・cacheする。

幾何経路のpenaltyは次とした。

`P_geom = max(0, (G_raw − G_no_edit) / quant_step² − cp_tau_geom)`

`Actual Actor/Critic RD = 3*C_actual + 150*P_geom`

- raw幾何損失のプロットとActual bit差分の定義は変更していない。
- 主損失もこのpenaltyを使用し、固定検証も同じ3:150の係数に揃えた。support cap等の安全処理は残る。
- 幾何codec既定は `cp_tau_geom=0`、`cp_geometry_backward_scale=1`。SparsePCGC用に校正された0.036は転用しない。
- 圧縮が正値でも、同一フレームの圧縮を維持・改善している場合は幾何・効率の補助信号を利用できる。
- no-opでWhereが0勾配になるのは正当。旧「必ずWhere>0」監査を幾何no-opへ適用しない。偽の勾配や強制編集で回避していない。
- proxy信頼度等をruntimeへ保存する。旧幾何runtimeと新しいRD/action binは互換ではないので、新規学習を推奨する。

実装: [geometry_rd.py](/home/maejima/MasterEx/myNet/models/utils/pointcloud/geometry_rd.py)、[geometry_codec.py](/home/maejima/MasterEx/myNet/models/utils/config/geometry_codec.py)。

## 4. 実際の訓練結果

8iのlongdress/loot/redandblack/soldierを各1フレーム、2 Episode反復して8 Step。seed=21、探索schedule基準382 Episode、validationは訓練と重ならないlongdress 1052の1フレーム。G-PCCの変更前相当は `--geometry_octree_joint_policy false` により同一入力で再実行した。

表は実bitstreamの `100*(edited_bits−GT_bits)/GT_bits`。正値は悪化。Lossの定義は再校正したため、旧Lossと新Lossの絶対値を性能改善率として比較しない。

| run | E1 Actual[%] | E2 Actual[%] | E1 raw幾何 | E2 raw幾何 |
|---|---:|---:|---:|---:|
| G-PCC 変更前相当 | +3.277571 | +2.756795 | 2.184874 | 2.184418 |
| G-PCC 変更後 | +1.327680 | +1.773167 | 2.184722 | 2.187685 |
| OctAttention 変更後 | +0.633469 | +0.960005 | 2.181979 | 2.183467 |

| 固定検証 | E1 Actual[%] | E2 Actual[%] | E2 raw幾何 |
|---|---:|---:|---:|
| G-PCC 変更前相当 | +3.979915 | +3.964059 | 2.184438 |
| G-PCC 変更後 | +1.183932 | +1.183932 | 2.182023 |
| OctAttention 変更後 | +0.793483 | +0.869455 | 2.179490 |

両codecとも8/8 optimizer更新成功、監査した勾配にNaN/Infなし。G-PCCはno-op 0/8、OctAttentionはno-op 1/8。G-PCCの軽減は全Stepをno-opへ置き換えた結果ではない。

**解釈の制限:** 初期の悪化軽減は確認したが、E2訓練平均はE1より悪い。8 Stepで学習曲線が改善・収束した証拠ではない。候補生成・初期化・行動分布も変更しているため、全差分をNetworkの学習成果とは呼ばない。G-PCCの訓練raw幾何はE2で旧版より約0.00327大きい。長期の品質保持も未確認。

### 勾配と実更新

optimizer直前のparameter gradient L2最大値。平均や圧縮単独勾配ではない。

| head | G-PCC | OctAttention |
|---|---:|---:|
| Candidate Actor | 0.004480 | 0.001340 |
| joint context（9 parameters） | 0.027994 | 0.017175 |
| Coarse Amount | 0.988605 | 0.999551 |
| Add Amount | 0.327017 | 0.276158 |
| Prune Amount | 0.939680 | 0.885710 |
| Adjust Amount | 0.001638 | 0.002609 |
| Gate | 1.000000 | 1.000000 |

各groupについて期間内の非ゼロ勾配と実parameter更新を別途確認した。ただし各headへ同じ量が流れているわけではない。

source別Actor probe（Candidate Actor＋joint context）は、G-PCCのActual-RD系平均約0.00761に対してproxy系0。OctAttentionでもproxy系0を確認した。Actual-RD系には幾何成分も入るので、圧縮対幾何の分離測定と解釈しない。

### joint化が実際に機能している証拠

- 8 Stepすべてfactorized group=`all`。他因子を固定した切り替えはない。
- 独立評価gain合計と、共通状態更新後のgainの差はG-PCCで約−1461.86～+2.67、OctAttentionで約−522.41～+10.45。**これはproxyモデル上の相互作用であり、実bit削減量ではない。**
- 各training Stepは編集後Actual encode=1、候補別Actual encode=0。best-of-Nのcodec oracleを追加していない。
- 単体テストではノイズにより実際のremove/add集合が変わること、ノイズ0ならseedを変えても同じ集合になることを確認した。
- joint選択処理の平均はG-PCC約0.23秒、OctAttention約0.28秒。候補生成と初回no-edit幾何参照計算は別にCPUコストが加わる。計算量が完全に不変とは主張しない。

## 5. SparsePCGC回帰

編集前後で同じseed・4入力・初期Surrogateの4 Step訓練を実施した。

| Episode平均 | 編集前 | 編集後 |
|---|---:|---:|
| Loss | −6.004306436 | −6.004306436 |
| Actual[%] | −3.610591054 | −3.610591054 |
| raw幾何 | 0.03218259802 | 0.03218259802 |

Episode CSV全列が記録精度で一致。既存のargs 1362項目も変更なし（幾何専用の新規7項目を追加）。

ただし保存重みの全テンソルはbitwise一致ではなく、最大abs差は約1.32e−5だった。その原因を特定したとは主張しない。短期の数値・実行結果の一致を確認したもので、全382 Episodeの同一性保証ではない。

## 6. 試したが既定に採用しなかった案

- G-PCC用の近隣ノード・既符号化child条件付きbinary proxyも実装・CPU検証したが、同じlongdressの合計0.3%編集でActualが+1.81%となり、empirical priorより良い証拠がなかった。実験オプションとして残し、既定にはしていない。いずれも実codec確率を完全再現しない。
- 広範囲のcoarseningでは同点群で約−33.30%を得たが、raw幾何が約2.18→4.85へ悪化した。これは独立した手動対照実験であり、Networkが達成した値ではない。既定の訓練に採用していない。
- longdressの対照planではbbox最小値は前後とも[0,0,0]だった。その例の悪化を、bbox shiftだけで説明することはできなかった。

## 7. テスト・証拠・次回起動

全268件のunittest、diff whitespaceチェックが成功。

- [joint選択・幾何参照・非線形性・noise集合変化のテスト](/home/maejima/MasterEx/myNet/tests/test_geometry_joint_policy.py)
- [実訓練の勾配・更新量記録ハーネス](/home/maejima/MasterEx/myNet/tests/run_geometry_training_smoke.py)
- [codecを順番に検証するハーネス](/home/maejima/MasterEx/myNet/tests/run_geometry_training_matrix.py)

主要ログ:

- G-PCC旧相当: `/tmp/mynet-geometry-joint-ETWeHV/gpcc-legacy-control/log/20261016/MyNetwork_train/G-PCC/151931.txt`
- G-PCC最終: `/tmp/mynet-geometry-joint-ETWeHV/gpcc-final-verified-budget-fixed/log/20261016/MyNetwork_train/G-PCC/153600.txt`
- OctAttention最終: `/tmp/mynet-geometry-joint-ETWeHV/octattention-final-verified-budget-fixed/log/20261016/MyNetwork_train/OctAttention/153824.txt`
- 各runの `log/decision_updates.jsonl`: head別gradientとparameter差分。
- SparsePCGC: `/tmp/mynet-geometry-joint-ETWeHV/sparse-before`、`sparse-after`。

新規起動する場合はmyOA環境で、例として:

```bash
python train.py --compression G-PCC --more_training false
python train.py --compression OctAttention --octattention_teacher_device cuda --more_training false
```

既存訓練と同時に長期runを追加することは推奨していない。現在のrunをどう扱うかはユーザーが決める。旧checkpointから新しいAmount class/RDへそのままresumeしないこと。

次に必要なのは、複数の固定検証フレームを用いた長期学習と、同じ幾何品質でのRD比較。今回確認したのはjoint探索・Actual学習経路・初期悪化軽減までであり、−40～−50%の圧縮効率向上、幾何損失の長期低下、早期収束問題の解消はまだ実証できていない。plane-fitの従来fallbackも今回のqs=2経路では未解決である。
