# G-PCC最新訓練の読み取り専用監査

調査日: 2026-10-16（JST）。対象コード: git `aead676`。調査開始時のworktreeはclean。

コード・設定・checkpointの編集、optimizer更新、訓練processの停止・再起動は行っていない。このレポートだけを新規保存した。追加実験はCPU上の独立processで既存候補生成器・G-PCC encoderを呼び、encoder自身の一時ファイルは終了時に削除した。GPUで別訓練は起動していない。

## 結論

最も強く裏づけられた原因は、**G-PCCで符号量を減らす候補・評価設計ができていないまま、その評価を事前順位とproxy教師として残していること**である。単なる学習停止や探索ノイズ無効化ではない。

1. G-PCC専用の量子化・Actual encoderへの分岐は実際に動作している。しかし候補の価値はG-PCCの符号化contextではなく26近傍の占有数だけから計算される。3点群の対照実験では、ヒューリスティック上位のAdd・Prune・Adjustは単独でもすべて符号量を増やした。
2. 通常Octree analyzerは専用クラス名でも共通実装の継承である。その占有パターン確率の分母に不整合があり、3点群で該当entropy特徴が全点1に飽和した。
3. poolには空間的な偏りがあり、操作もごく少量の局所編集に限定される。学習可能なglobal scaling、まとまったsubtree collapse、Adjustによる既存点への統合は現在の候補経路にない。
4. 実際の探索・パラメータ更新は存在する。一方、後半のActorではActual-RD由来よりproxy系蒸留の勾配が大きく、圧縮が正値の編集planには幾何改善・操作量削減の補助方策が一度も適用されていない。
5. 過去の「−50%」の該当runは特定できなかった。発見した旧G-PCC訓練はproxy値であり、論文のBD-rate削減とも現在の1点のActual差分とは比較条件が異なる。「昔の実符号量−50%が今回失われた」とは、現在確認できた証拠からは断定できない。

以下では、観測・コードから確定できることと、その影響についての推定を分ける。

## 1. 対象ログと実測

対象は [/data/maejima/log/20261016/MyNetwork_train/G-PCC/014040.txt](/data/maejima/log/20261016/MyNetwork_train/G-PCC/014040.txt)、同prefixの `epi_metrics.csv`、`epi_point_edits.csv`、`checkpoint_metrics_epi.csv`。訓練は調査中も進行しているため、集計範囲を完了済みEpisode 1–122、4,880 training Stepに固定した。全382 Episode完了後の結論ではない。

圧縮指標は実際のbitstreamについての

`C = 100 × (編集後bits − 未編集bits) / 未編集bits`

であり、正値は悪化、負値は改善。baseline・編集後とも同じG-PCC geometry-only設定、prequantize=True、qs=2。元のqs=1の符号量との差ではない。

| Episode範囲 | Actual C平均[%] | 幾何損失平均 | Add[%] | Prune[%] | Adjust[%] |
|---|---:|---:|---:|---:|---:|
| 1–10 | +3.348766 | 2.187951 | 0.112441 | 0.103966 | 0.049217 |
| 40–60 | +2.330972 | 2.185834 | 0.097732 | 0.063170 | 0.032441 |
| 80–95 | +2.194665 | 2.185676 | 0.097822 | 0.063865 | 0.031942 |
| 96–110 | +1.751684 | 2.184866 | 0.073945 | 0.048726 | 0.024361 |
| 111–122 | +1.737509 | 2.184908 | 0.073420 | 0.047784 | 0.023910 |

操作割合はログのcanonical voxelに対するpercentであり、0.073420は7.3420%ではない。

- 4,880 Stepすべてが `source=fresh_teacher`。実符号化を止めてproxyだけを表示している状態ではない。
- no-opは101 Stepで、Actualはすべて0%。それ以外の4,779 Stepはすべて正値、範囲は+1.315110～+5.273935%。負値は0 Step。
- 最初のStepはbaseline=171,688 bits、edited=178,128 bits、C=+3.750990%。Add279、Prune282、Adjust138。
- つまり今回観測しているのは「圧縮改善の学習」よりも、まず**編集による圧縮悪化を小さくしている過程**である。no-opで0%にできることと、−40～−50%を実現することは別問題。

固定5フレーム検証でも、E111–122平均のNetworkは+1.853541%、ヒューリスティック選択は+1.865523%。差は約−0.012 percentage pointにとどまる。この比較は後述のとおり、現行Amount等を完全に固定した昔の独立モデルとの比較ではない。

## 2. 現在有効な経路と、SparsePCGC依存の確認

実行経路は次のとおり。

`G-PCC用qs=2量子化 → unique canonical voxel → GeometryOctreeStructureAnalysis（共通40ch記述子を継承）＋固定Point Transformer特徴 → geometry_octree_proposalsの合法3操作pool → 共通guidance写像・safe判定 → Proposal subset → H＋有界Actor residual（Actual Criticで権限制御） → 探索 → Amount/Gate/Fine・競合排除 → 編集後voxel復元 → 実tmc3 encode → Actual-RD/proxy補助学習`

確認できた分離は以下。

- [Networkのanalyzer分岐](/home/maejima/MasterEx/myNet/models/network.py:121)、[候補生成器の分岐](/home/maejima/MasterEx/myNet/models/network.py:5132)がある。
- [geometry_octree.py](/home/maejima/MasterEx/myNet/models/utils/pointcloud/geometry_octree.py:49)のG-PCC量子化は、Actual encoderと同じ「絶対座標round→整数bbox shift」。復元とcache fingerprintもgeometry codec側に分かれる。
- [geometry_codec.py](/home/maejima/MasterEx/myNet/models/utils/config/geometry_codec.py:35)でSparsePCGCの専用teacher/oracle/filterを無効化。ログでもSparsePCGC teacher/cache=False。
- 実encoderは同梱 `compress/octree/G-PCC/build/tmc3/tmc3`。[実行引数](/home/maejima/MasterEx/myNet/models/utils/compression/gpcc_tmc3.py:136)は同梱encoder.cfgを読み、attributeを無効化、positionQuantizationScale=1、qs=2は事前量子化で適用する。測定値はファイルbytes×8。
- GT cache実ファイルのfingerprintにもcodec=gpcc、qs=2、encoder/configの識別、入力ファイルの識別が含まれている。
- `sparse_tensor.py`由来の低水準関数を一部共有するが、ここで共有するのはtorchによる点の予算縮約等。ファイル名にSparseがあること自体はSparsePCGC専用計算の誤適用の証拠にならない。

したがって、「G-PCC指定なのにSparsePCGCで圧縮している」「そのままSparsePCGCの候補cacheを読む」という仮説は、今回のログ・現行分岐では支持されない。ただし、**分岐して実行可能であることと、G-PCC向けに適切な候補・符号量特徴を設計できていることは別**である。

## 3. 最も重要: 候補の価値がG-PCC符号量に対応していない

### 3.1 確認した式

[geometry_octree_proposals.py](/home/maejima/MasterEx/myNet/models/utils/pointcloud/geometry_octree_proposals.py:62)で、dをsource voxelの26近傍の占有数として、

- Add score = d / 26
- Prune score = (26 − d) / 26
- Adjust score = abs(d − 13) / 26
- geometry_cost = 1 / (d + 1)
- neighbor_bit_risk = 0
- optimistic_gain_bits = 上記scoreそのもの

となる。codec名はcache識別等には使うが、この価値式をG-PCCとOctAttentionで切り替えていない。同じcanonical gridなら同じ式である。

Add/Adjustの移動先は空近傍の中からhash位相で1つ選ぶ。移動先ごとのG-PCC context、符号長、親occupancy変化、隣接ノードへの影響を比較していない。geometry_costも移動先や原表面との距離を評価していない。

このscoreはコード冒頭では明示的にproxyと説明されている。しかし共通[local教師生成](/home/maejima/MasterEx/myNet/models/modules/heuristic_guidance.py:391)では `optimistic_gain_bits` をrate benefitとして読み、標準化したrateとgeometryから教師を作る。今回のgeometry poolでは、ここへ入る「bits」は実bitの推定ではなくHと同じ近傍scoreである。

現在のG-PCC encoderには階層、近傍context、直接符号化等の処理がある。単純な点の増減・26近傍密度と実符号量は同じではない。例えばPruneで点数が減っても、occupancyの予測しにくい穴やcontext変化によりbitが増え得る。

### 3.2 CPU上の実G-PCC対照実験

既存の候補生成器をそのまま呼び、各操作のH score上位を選んだ。合計編集予算はcanonical voxel数の約0.15%、配分は0.4/0.4/0.2。競合した候補はスキップし、各操作の件数を満たす。Actual encoder・cfg・qs・geometry-only等は最新runと同じ。

これは**Network checkpointの厳密なplan再生ではなく、候補・Hの妥当性を調べる独立した対照実験**。単独操作では他操作を実施せず、jointではAdd→Prune→Adjustを適用した。品質同一でのRD比較や全poolの最適解探索ではない。

| 点群 | canonical voxel数 | 未編集bits | Add単独 C[%] | Prune単独 C[%] | Adjust単独 C[%] | joint C[%] |
|---|---:|---:|---:|---:|---:|---:|
| longdress_vox10_1051 | 211,887 | 150,240 | +0.670927 | +0.612354 | +0.772098 | +1.821086 |
| loot_vox10_1000 | 216,179 | 144,536 | +0.675264 | +0.603310 | +0.819173 | +1.970443 |
| soldier_vox10_0536 | 294,679 | 215,664 | +0.660286 | +0.541583 | +0.645448 | +1.813933 |

件数は順にAdd/Prune/Adjust=127/127/64、130/130/65、177/177/89。joint編集後bitsは152,976、147,384、219,576。

選んだ候補のproxy gain合計はjointでそれぞれ+168.30769、+173.80769、+236.34615だったが、実bitはすべて増えた。proxy値を実bit予測として解釈できないこと、そして現在の上位候補群が圧縮改善の出発点になっていないことを直接確認した。

ただし「pool内に良い候補が1つもない」「この式だけが全悪化の原因」とまでは、この3点群の実験から断定しない。

## 4. 候補poolと操作空間の制限

### 4.1 poolの空間偏り【確定】

[候補生成の行選択](/home/maejima/MasterEx/myNet/models/utils/pointcloud/geometry_octree_proposals.py:38)はuniqueした座標を並べ、requested個のsourceを全体から等間隔に抽出する。その後、Pruneはそのsource配列の**先頭budget個だけ**、Add/Adjustも前から列挙してbudget到達でbreakする。

従って、最初のsource抽出が全体に分布していても、最終poolは全体の均等サンプルではない。CPU実測ではPrune poolのsourceはlexicographic順の先頭約30.25%、30.31%、31.06%までに限定された。

| 点群 | 全体x範囲 | Prune poolのx範囲 | Prune/Add/Adjust pool件数 |
|---|---|---|---|
| longdress 1051 | 0–178 | 0–57 | 848 / 848 / 424 |
| loot 1000 | 0–176 | 0–58 | 865 / 865 / 433 |
| soldier 0536 | 0–180 | 0–77 | 1179 / 1179 / 590 |

これらの範囲はcanonical座標。Add/Adjustも先頭から埋める設計なので同種の偏りを持つが、上表のx範囲・quantile実測はPruneについての値。

### 4.2 Proposal後の絞りすぎではない【今回の観測】

E111–122のStep平均poolはAdd939.94、Prune939.94、Adjust470.20件。初回は939/939/470で、safe後・Proposal subsetにも同数が残っている。ログで見える主な制限は巨大poolをProposalで上位だけに削る処理ではなく、**それ以前の小さく偏ったpoolの構築**である。

候補数を全点数と比較すると、各操作で全voxelの約0.4%/0.4%/0.2%のpoolしか用意しない。さらにAdd/Adjustは各sourceに原則1方向だけ。探索してもこのpoolの外は選べない。

### 4.3 大幅な符号量削減を探索する操作表現が狭い【確定＋影響は推定】

- 正のAmount binは合計0.15%、0.20%、0.25%、0.275%、0.30%。別にno-opがある。Fineで小幅調整する。
- E111–122の全480 Stepは最小の正bin=0.15%を選び、実合計操作率平均は約0.144893%。
- 既存の共通安全上限は0.99%。この値を単に上げれば改善する、と確認したわけではない。
- 候補はAdd=1点追加、Prune=1点除去、Adjust=1点除去＋空voxelへ1点追加。
- [幾何codec設定](/home/maejima/MasterEx/myNet/models/utils/config/geometry_codec.py:67)はAdjustの空移動先を必須とする。既存voxelへの統合で点数を減らすAdjustではない。
- qsは固定。現在のpoolにglobal scalingや一括subtree collapseの候補はない。

これだけの編集率から「bit削減は数学的に0.15%以下」とは言えない。codecのcontext変化は非線形だからである。しかし、論文のglobal scaling・広範なvoxelizationと比べ、探索できる再構成の種類・範囲が大きく異なることは確実。

## 5. Octree構造分析に見つかった不整合

[GeometryOctreeStructureAnalysis](/home/maejima/MasterEx/myNet/models/modules/geometry_octree_structure.py:6)は量子化・cache namespaceを切り替え、解析式は共通OctreeStructureAnalysisを継承する。通常のoccupancy、sibling、multilevel topologyを扱うこと自体は正当であり、SparseTensorでしか成立しないわけではない。しかしG-PCCの実符号化確率・contextを再現する解析器にはなっていない。

さらに[_prebuilt_octree_context](/home/maejima/MasterEx/myNet/models/modules/octree_structure.py:1868)に、具体的な確率正規化の不整合がある。

パターンcの親ノード出現数をh[c]、親総数をMとすると、現在の式は

`p_current(c) = h[c] / Σ_c h[c]^2`

になる。occupancyパターンの経験確率として通常使う `h[c]/M` ではない。親ごとに複製した頻度配列をsumするためである。一方、同ファイルの[leaf pattern診断](/home/maejima/MasterEx/myNet/models/modules/octree_structure.py:998)では親総数で割っており、2経路が一致しない。

このpを用いて `pattern_nll = -log2(p)/8`、さらにclamp(0,1)を通す。

| 点群 | 親総数M | 現在の分母/M | 現在式でentropy=1の点割合 | h/Mを用いた参照計算の割合 |
|---|---:|---:|---:|---:|
| longdress 1051 | 55,435 | 2448.53 | 100.00% | 12.97% |
| loot 1000 | 56,514 | 2573.51 | 100.00% | 13.73% |
| soldier 0536 | 77,139 | 2752.56 | 100.00% | 11.47% |

参照計算は正規化だけを変えた独立計算で、実装は変更していない。このentropyは40ch入力に含まれ、context/shape proxyにも利用される。例えば `neighbor_occ*(1-bit_entropy)` の部分は、この3点群では全点0となる。

**確定:** 少なくともこの特徴は区別能力を失う。**未確認:** 分母だけを直した場合に訓練・Actualが何%改善するか。共通経路の問題なので、将来修正する際はSparsePCGCへの回帰検証も必要。

## 6. 探索・学習は動いているが、Actorの教師と制限に問題が残る

### 6.1 探索自体は存在する

探索窓は全15,280 Stepの先頭3,820 Step、約95.5 Episode分。1始まりのEpisode表示ではE96途中で終了する。

最初のStepでは、ノイズ直前のH score標準偏差がAdd≈0.08779、Prune≈0.08773、Adjust≈0.06142。実際のGumbel標準偏差は約0.09449、0.10232、0.08851。ノイズによる候補集合交換数は140/141/69件だった。

E1–10全400 Step平均の交換数は64.84/61.61/26.76件。Where探索でないStepも含む平均である。E111–122は全操作0件。探索は候補集合へ効いているが、悪い・偏ったpoolの中で選択を変えても、それだけで良いplanは得られない。

E80–95の640 Stepの探索groupはWhereAll294、Fine288、Gate58、Amount0だった。WhereAllはAdd/Prune/AdjustのWhereを同時に変えるが、Where/Amount/Gate/Fineという因子間では切り替える。[matched anchor](/home/maejima/MasterEx/myNet/models/modules/structure_actuator.py:5395)が他因子の件数等を固定する経路もある。したがって「全headが同時に自由に探索している」構成ではない。

E96途中で探索・anchor条件が変わり、その後は全Stepで最小正binになる。圧縮損失低下を全部Networkの改善とも、全部ノイズ減衰とも断定できない。少なくともAmount縮小と探索終了が同時期に起き、幾何損失の変化よりも操作量減少が大きい。

### 6.2 パラメータ更新停止ではない

checkpoint_metricsでは全122 Episodeでoptimizer成功率1、非有限勾配によるskipは0。さらに保存済み `0.pth` と `121.pth` をCPUで読み、パラメータ差分を直接確認した。

| head | E1→E122パラメータ差分L2 |
|---|---:|
| Candidate Actor | 1.466055 |
| Candidate Critic | 4.229898 |
| Amount selector | 0.204291 |
| Operation Gate | 0.651580 |
| Add Amount / Prune Amount / Adjust Amount | 0.030328 / 0.023518 / 0.023793 |

これにより「全headの更新が止まっている」は棄却できる。ただし、更新が有益な方向であることまで保証しない。

### 6.3 Actorにproxy教師が間接的に残る【確定】

`actor_proxy_weight=0`で直接proxy模倣を無効にしても、[proxy Criticとの蒸留](/home/maejima/MasterEx/myNet/models/modules/structure_actuator.py:5976)を `actor_critic_distill_weight`（既定0.10）で[local creditへ加える](/home/maejima/MasterEx/myNet/models/modules/structure_actuator.py:7162)経路が残る。ここで蒸留相手はActual Criticではなく `candidate_proxy_critic_values`。従って「proxy imitation=0だからActorは実codecだけから学ぶ」とは言えない。

最新ログの毎10 Stepのsource別勾配probeを集計した。以下はCandidate Actorのparameter gradient L2平均であり、異なるheadのnorm比較ではない。

| 期間 | Actual-RD系Actor | proxy系Actor | entropy Actor |
|---|---:|---:|---:|
| E1–10、40 probe | 0.00004755 | 0.00853730 | 0.00000514 |
| E80–95、64 probe | 0.02586281 | 0.00638990 | 0.00004744 |
| E111–122、48 probe | 0.00024370 | 0.00563025 | 0.00001310 |

初期はproxy系が約180倍、後半は約23.1倍。探索中のE80–95にはActual系が大きい時期もあるため、「全期間ずっとproxyしか学習していない」とは言わない。後半のproxy勾配は、G-PCC符号量ではなく前述の共通近傍proxyを教師にした蒸留を含む。

Actual-RD系sourceは圧縮単独ではなく、コード上 `3*C + 150*relu(G-0.06)` を含む。表のActual列から圧縮と幾何の勾配を分離したと解釈してはいけない。

### 6.4 Hを消せないforward制約も残る

[prior_residual選択](/home/maejima/MasterEx/myNet/models/modules/structure_actuator.py:5648)は `score=H+authority*0.5*tanh(normalized_actor)`。探索終了後はActual CriticのLCBがH cutoffを上回る候補だけresidual権限を残す。G-PCCでH自体が有効だという証拠がない段階でも、安全基準としてHを維持する。

ただしHだけに完全固定されてはいない。E111–122のH Top8とのcombined Top8 overlap平均はAdd0.5953、Prune0.1841、Adjust0.5438。これはTop8の比較であり、実行plan全体のoverlapではない。選択は変わるが、固定検証のActual改善は約0.012ppにとどまる。

### 6.5 順位監査ログにも注意点

geometry poolの `pool_rank` はH score順ではなく、生成順のenumerationである。一方、[順位ログ計算](/home/maejima/MasterEx/myNet/models/modules/structure_actuator.py:7315)はこれを `selected_heuristic_rank_mean`、`outside_heuristic_budget`、rank=0ならheuristic Top1として扱う。

従って、このgeometry runで「rank_meanが大きいから中・低H優先度を十分選んでいる」と読むのは不正確。ノイズの交換数やscoreを直接argsortしたTop8比較と区別する必要がある。さらにgeometry payloadには利用可能なanchor candidate集合がないため、空anchorとの比較から出る `residual_candidate_delta ratio=1` も、H plan全体を100%変更した証拠ではない。

## 7. 幾何損失・学習条件のcodec不適合

### 7.1 G≈2.18の大部分は編集前からの量子化歪み

Gは原点群と編集後canonical復元点群のsquared Chamfer双方向平均和。原点群全体をqs=2で丸めただけの無編集点群をCPUの厳密最近傍検索で評価すると、

| 点群 | 無編集qs=2のChamfer |
|---|---:|
| longdress 1051 | 2.17832927 |
| loot 1000 | 2.17316991 |
| soldier 0536 | 2.18996599 |

となる。ログのno-op101 StepのG平均も2.180865。したがってG≈2.18を「編集で増やした歪み」とみなすことはできない。これはqs=2・現在の出力表現を維持した無編集baselineであり、すべての再構成に対する数学的な最小値と主張するものではない。

`cp_tau_geom=0.06` はこれより桁違いに小さい。さらに全Stepでplane-fit項は `disabled_without_addition_provenance` となる経路を使っている。geometryの全機能をG-PCCへ適切に移植できているとは言えない。

### 7.2 公称重み150と実際の効き方は異なる

実行設定は `w_com=3`, `cp_lambda_geom=150`, `cp_geometry_backward_scale=0.036`, `cp_tau_geom=0.06`, support/main cap=0.75。`w_geom=10000` はcompression_primaryでは非有効なlegacy値である。

[圧縮主目的の合成](/home/maejima/MasterEx/myNet/models/utils/training/compression_primary_loss.py:378)は概略

`L = 3*C + s * {150*relu(G-0.06) + 小さい補助項}`

であり、幾何のbackwardにはさらに0.036を掛ける。sは毎Stepのsupport capから算出されるdetached係数。全4,779編集Stepでsupport/main=0.7500、全101 no-op Stepで0だった。幾何forward値が大きいため全編集Stepでcapが作動しており、正値CではプロットのLはほぼ `5.25*C` になる。

最初のStepはmain=11.2530、raw幾何block=318.0145、幾何block適用後=8.4337、s≈0.0265、dL/dG≈0.1432。E122最終Stepはmain=5.1791、raw幾何block=319.7687、適用後=3.8822、s≈0.0121、dL/dG≈0.0656。

**幾何の数値が大きいのに、全体損失の形は圧縮損失に従い、圧縮悪化が小さくなるほど幾何backward係数も小さくなる**構造である。この局所スカラー係数はNetwork全パラメータの勾配norm比ではない。

また、hard decision headは[policy勾配を保存・復元](/home/maejima/MasterEx/myNet/models/utils/training/train_runtime.py:5175)するため、通常のSurrogate/geometryのpathwise勾配をそのまま受けない。公称150を上げるだけでAmountへの幾何勾配が増えるわけではない。

### 7.3 圧縮改善を前提にした補助方策が今回ほぼ働かない

[geometry policy guard](/home/maejima/MasterEx/myNet/models/network.py:2522)も[efficiency guard](/home/maejima/MasterEx/myNet/models/network.py:2668)も、Actual圧縮差分が `min(0, baseline+tolerance)` 以下であることを要求する。

- 4,779編集Stepでgeometry_policy_weighted≠0は0、eff_guard=Trueも0。
- no-op101 Stepではgeometry非ゼロ86、eff_guard=True93。
- E111–122のFine Amount合算gradient L2平均は約1.10e−7。Where合算は0.004797、Amount合算は0.007021、Gate相当actionは7.47e−5。

つまり幾何改善・操作量削減の専用補助経路は、圧縮が正値の現状の編集planに対して機能していない。ただしActual-RD教師とcandidate-local proxyにも幾何成分があるので、幾何由来の学習が完全にゼロとは言わない。

操作別Where/Amountの細分化grad audit、および圧縮対幾何の全head別gradient probeはこのrunで無効。ログのoperation_grad_normsの0は未測定値を含むため、「Add/Prune/Adjustがすべて勾配0」と解釈しない。今回の読み取り専用監査では訓練を再実行していないので、存在しないsource別normは報告できない。

### 7.4 3種類のRD定義が揃っていない

- plot: capを通した `3*C + s*(150*P_geom+...)`
- Actual Actor/Critic教師: [3*C+150*P_geom](/home/maejima/MasterEx/myNet/models/network.py:992)
- fixed validation比較: [C+150*P_geom](/home/maejima/MasterEx/myNet/models/utils/training/train_runtime.py:6781)

固定検証だけw_com=3が適用されず、相対的に幾何を3倍重視する。これも選択・学習・評価の整合性の問題。ただし今回の+1.7%の全てをこの不一致に帰する証拠はない。

固定検証の `heuristic` overrideは、geometry payloadにanchor件数がない場合、Amount等を現行Networkから取得したままWhereをH順へ戻す。従ってその曲線を「学習していない固定Heuristicモデル全体の性能」と呼ぶことも不正確である。

## 8. 過去−50%・論文との差

### 8.1 保存されている過去G-PCC run

`/data/maejima/log`配下で見つかった旧G-PCC訓練は20260524の173251/173306。173306は17 Episode、5,100 Stepについて

- actual_value_source=proxyが全5,100行
- fresh_actual=Falseが全5,100行
- actual_codec_disabled=Trueが全5,100行
- Episode平均compressionは約+66.63～+78.81、Actual列なし
- Step単位proxyは−100%も含むが、実G-PCCで確認した負値ではない

だった。さらに20260928の `gpcc_173306_ep13_actual_bits.csv` では旧checkpointの保存出力を実符号化した90フレームが+53.73～+138.08%、合計bits比では+103.72%で、こちらも−50%ではない。

この旧モデルは現行と異なる入力・subtree・操作率・保存出力経路である。これらからユーザーの記憶した別runを否定したり、「昔の−50%は必ずproxyだった」と断定したりはしない。比較対象のログ/checkpoint/codec設定が必要であり、今回の保存資料から該当−50%の再現はできなかった。

### 8.2 論文の数値

参照フォルダのMaらの論文と[著者公開arXiv](https://arxiv.org/abs/2508.01633)を確認した。平均38.84%の改善はBD-rateであり、global scaling・学習voxelization・pruning・点編集を用いる。現在のqs固定、微少局所編集の1条件でのbit差分と同じ指標・操作空間ではない。

ローカルのTIP版PDFの実験節ではG-PCC(TMC13-v23)、D1/D2 PSNRでRD評価し、平均BD-rateはD1で−38.84%、D2で−27.90%。Surrogateを先に10 epoch学習し、固定してから前処理Networkを30 epoch学習する。今回はG-PCC用pretrained Surrogateが見つからず初期化され、オンライン更新される点も異なる。

従って「G-PCCへencoderを切り替えれば、現行のまま−40～−50%になる」という期待を支える証拠はない。ただし、指標差があることは今回の実bit増加を正当化する理由ではない。今回の候補・教師の不整合は独立して解決すべき問題。

## 9. 原因の優先順位と、次に修正するなら

| 優先 | 指摘 | 証拠の強さ |
|---|---|---|
| 1 | G-PCCに対応しない候補価値と、それを維持するH/proxy蒸留 | 式を確認。3点群×3単独操作＋jointすべて実bit増。後半Actor proxy系normがActual系の約23倍 |
| 2 | pool生成の空間偏り・方向不足・狭い編集表現 | コードと3点群のsource範囲で確認。−50%への影響量は未測定 |
| 3 | 継承Octree特徴の確率分母不整合・entropy飽和 | コードの2経路不一致、3点群で100%飽和を確認。修正後学習性能は未測定 |
| 4 | qs=2に未適合なgeometry閾値・cap・正値圧縮guard | 全編集Stepのcap/guard、無編集Chamfer実測で確認 |
| 5 | 探索後の最小正Amount固定とActual対比の制限 | E111–122の480 Step、E80–95のAmount探索0を確認 |
| 6 | plot/Actor/fixed validationのRD係数不一致、順位ログの誤命名 | 現行式で確認。性能への単独寄与は未測定 |

**最優先の1点は、G-PCC用の候補価値と教師を実符号量に整合させること。** 単にノイズや操作上限を大きくすることではない。

今後の修正案としては、現行poolと改善poolに対し、同一入力・同一qs・同一codecで少数の差分Actual検証をまず行う。親occupancyの整理、連続したsubtree、隣接context変化を評価に含め、負のgainも表現し、誤ったproxyをActualより優先する蒸留・H安全基準を見直す。同時にpoolの空間偏りと確率分母不整合を回帰テスト対象にする。これは提案であり、今回実装した変更ではない。

## 10. この監査で言える範囲

- **確認済み:** 実G-PCC経路、正の圧縮差分、探索集合の変化、パラメータ更新、候補価値式、pool空間偏り、entropy特徴飽和、proxy蒸留、cap/guard条件。
- **強く疑われる説明:** 改善に結びつかない候補/Hを出発点に、ごく小さい操作へ縮小して悪化を抑えているため、+1.7～+2%付近で見かけ上安定する。
- **未確認:** 各原因を単独修正したときの改善率、全poolの最良plan、全382 Episode後の結果、別の過去−50%runの再現、D1/D2の複数RD点でのBD-rate。
- **実施していない:** ソース/設定/checkpoint変更、学習再開、既存processへの介入、−50%達成の保証。
