# Octree幾何圧縮の追加研究: 大きな改善に必要なもの

2026-10-17。基準commit `deb783be385bd764d1bbb87e683655b2d9399ebb`。

## 要旨

**SparsePCGCを含む既存訓練コード・args・checkpoint・実行中processは変更していない。新しい独立実験だけを追加した。**

今回、新たに確認できたこと:

1. OctAttention自身の条件付き確率で葉occupancyを選ぶと、総操作率約0.15%でActual −1.02～−1.45%。従来の幾何・密度priorより有効な候補評価が可能だった。
2. 注意window内の相互作用と整数CDFまで再評価すると、同じ約0.15%で−1.560% / −1.303%。D1相当MSEと最大点距離を維持し、幾何損失も低下した。
3. **そのままG-PCCに転用しても同じ改善にはならない。** 同じ点配置で、OctAttentionが改善しG-PCCが悪化する例もある。
4. 量子化幅を変えずに丸め方を変えるだけで、OctAttentionでは約−30%も出た。しかしvoxel集合の約21%が変わり、現在の微小操作予算内の成果ではない。
5. **微小操作・現行codecを維持した−50%は未達。** 大きな改善のためには、codec固有の文脈を扱う候補価値と、多数の葉の符号量をまとめて下げる仕組みを分けて研究する必要がある。

## 1. 条件・検証範囲

- 入力: longdress 1051、loot 1000、soldier 0536、追加確認にredandblack 1451。
- 既存G-PCC encoder/config、既存OctAttention checkpointを使用。いずれも再学習・上書きしていない。新規計算はCPU。
- 比較する幾何を揃えるため、共通gridはG-PCCと同じ `round(raw/2)` → bbox正規化。OctAttentionにはその整数座標をqs=1で渡す。物理座標ではqs=2に相当する。
- **これは両codecのnative量子化を混ぜた比較ではない。** native OctAttentionのraw-min-first量子化とは異なる。longdressのbaselineは共通grid 188,680 bits、native OA qs=2は188,072 bitsだった。
- 圧縮損失 `C = 100*(edited_bits/baseline_bits - 1)`。負値が改善。G-PCCは全stream bytes、OAは実算術stream bytesであり、CEの推定値ではない。
- 操作率の分母は未編集unique voxel数。Adjustは1点移動を1操作、占有cell差分は別に数える。複数点bundleは実点数で課金する。
- Gは元座標での対称平均二乗Chamfer。D1用の最大方向MSEと両方向最大点距離を併記。D2・法線・知覚評価やBD-rateの再現は未実施。

OAの新しい実験器と既存teacherの算術bitsは、実フレーム・小点群で一致した。baselineとwindow法の出力について算術symbolのroundtripとOctreeからの座標集合復元も確認した。ただし**OA roundtripはencoder側で計算した確率を使っており、独立した自律decoderの全実行ではない**。G-PCCのwindow法出力は実decoderで座標一致を確認した。

OA streamにはモデル・座標復元等の外部情報を含まない。現行teacherと同条件だが、アプリ全体の配信payloadとは区別する。

## 2. 主な実験: codec自身の確率で候補を評価する

### 2.1 方法

最深層の親occupancyについて、現在のcheckpointから `p(mask | ancestors, previous context)` を取得する。

`候補gain = log2 p(編集後mask | 編集前のcontext) − log2 p(元mask | 編集前のcontext)`

単点Add/Prune、同じ親内のAdjustを全葉から生成し、gainの高い候補を共同選択した。安全な代替近傍を残すため、Prune/Adjust後も元入力の各点の最近傍距離を増やさない。Addは元入力に近い空voxelのみ。bboxは保護し、親occupancyを空にはしない。

これはNetworkの学習成果ではなく、既存codecを教師にした独立した候補探索である。候補選択時のcontextはまだ固定しているので、gainは**最終Actual gainではない**。編集した点群全体を再符号化して確かめた。

### 2.2 微小操作での結果

総操作率約0.15%、同じ編集を両codecで圧縮:

| 入力 | Add / Prune / Adjust数 | OA C | G-PCC C | G: 未編集→編集 |
|---|---:|---:|---:|---:|
| longdress 1051 | 31 / 30 / 257 | −1.445834% | −0.394036% | 2.178329→2.177156 |
| loot 1000 | 52 / 37 / 235 | −1.159101% | −0.094094% | 2.173170→2.172152 |
| soldier 0536 | 71 / 27 / 344 | −1.022195% | +0.066771% | 2.189966→2.188988 |

OA平均−1.209044%。全3フレームでD1用MSE・最大点距離を維持した。単純に全Stepをno-opへ置き換えた結果ではない。

総操作率約0.30%ではOA −2.510070 / −2.181578 / −1.834783%。しかしG-PCCでは−0.404686 / −0.077489 / +0.293048%。**OA向けの候補価値をG-PCCの価値として流用してはいけない。**

## 3. 候補間の相互作用を正しく評価するwindow法

現在のOA実装は1024-node windowごとにTransformerを実行する。葉の親を空にしなければ、上位Octree・葉の位置・sequence長を変えずにoccupancyだけを書き換えられる。

window境界の編集を除外し、各windowに候補数1/2/4/8/16/32のmenuを作った。それぞれについて**編集後contextでwindow全体の確率を再計算**し、実算術coderと同じ整数CDFからrateを求める。その後、総操作予算を制約とするmultiple-choice knapsackでwindowの組合せを選んだ。

| 入力 | 総操作率 | Add / Prune / Adjust | OA bits | OA C | G-PCC C |
|---|---:|---:|---:|---:|---:|
| longdress 1051 | 0.150080% | 37 / 36 / 245 | 188680→185736 | −1.560314% | −0.404686% |
| redandblack 1451 | 0.150103% | 48 / 47 / 203 | 187208→184768 | −1.303363% | −0.427953% |

| 入力 | G: 未編集→編集 | D1用MSE: 前後共通 | 両方向最大点距離: 前後共通 |
|---|---:|---:|---:|
| longdress | 2.178329→2.177170 | 1.5005765055 | 1.7320508076 |
| redandblack | 2.175526→2.174435 | 1.5011143986 | 1.7320508076 |

### 「候補scoreがあるだけ」でない検証

- windowで予測したCDF gainは2,941.612484 / 2,439.246388 bits。
- 最終点群をもう一度全sequenceで評価したCDF gainとの差は約1.2e−11 / 8.6e−12 bits。
- 実算術streamの削減は2,944 / 2,440 bits。CDFの理想符号長とstream長は終端・byte丸めにより少し違う。
- 編集座標集合、操作数、幾何評価、Actual streamまで変化を確認した。

ただし、各windowのmenuは元contextでのgain順prefixであり、全mask・全組合せの最適解保証はない。**window内評価が正確であることと、探索全体が最適であることは別。**

候補最適化だけで約67 / 63秒、window forward 330 / 312回が必要だった。通常訓練へそのまま追加すると重い。これはoffline教師・診断用であり、実運用では蒸留等が必要になる。

また、G-PCCにはこのwindow独立性を仮定できない。G-PCCの近傍context・adaptive coding状態は別物である。

## 4. 大きな数値が出る条件と、その限界

### 4.1 丸め規則を変えるとOAでは約−30%

qs=2を保ち、banker's roundingからhalf-up/floorへ変更した。D1用MSEは同じだが、point集合自体は大きく変わる。

| 入力・規則 | OA C | 同じ点群のG-PCC C | voxel変更の最小操作率 | G: 元→編集 |
|---|---:|---:|---:|---:|
| loot・half-up | −30.039665% | +6.320917% | 21.555748% | 2.173170→2.177762 |
| loot・floor | −30.180696% | +6.381801% | 21.486823% | 2.173170→2.178372 |
| soldier・half-up | −29.175640% | +8.327769% | 21.422293% | 2.189966→2.195227 |
| soldier・floor | −29.557527% | +8.183100% | 21.096515% | 2.189966→2.195321 |

0.15%操作と比べて約140倍の変更をしている。global量子化を1パラメータで変えられても、「1操作で済んだ」と数えてはいけない。D1同等でもGはわずかに悪化しており、全品質同一の主張もしない。

重要なのは、**同じOctree表現でも、各codecが予測しやすい配置は違う**こと。この結果は大きなrate差が存在する証拠だが、現在の微小操作制約内で−30%出せた証拠ではない。

### 4.2 8-child maskをまとめて編集する

255通りの非空maskから、品質制約を満たすAdd/Prune/Adjustの複合置換を評価した。各変更voxelを操作数に含め、候補1個を操作1個とは数えていない。

| 入力 | 実総操作率 | OA C | G-PCC C |
|---|---:|---:|---:|
| longdress | 0.150080% | −1.445834% | −0.394036% |
| redandblack | 0.150103% | −1.145250% | −0.387675% |
| longdress | 3.000184% | −14.318423% | −0.212993% |
| redandblack | 3.000050% | −12.721679% | −1.394623% |
| longdress | 14.448739% | −20.784397% | +7.087327% |
| redandblack | 14.182239% | −18.729969% | +2.381432% |

大きなOA改善は測れたが、操作量が大きい。これを現在の設計が達成した性能や、微小操作の解決として採用しない。全条件でD1用MSE・最大点距離を維持し、Gは低下したが、D2等は未検証。

### 4.3 可逆な軸入替・反転

逆変換すれば物理点集合は完全に同じ。Add/Prune/Adjustとは別のcoding transformであり、transform IDとinverse wrapperが必要。48通りの逆変換を単体テストした。

G-PCCは48候補、OAは6軸順序を探索。IDに1byteを加算した最良のC:

| 入力 | G-PCC | OA |
|---|---:|---:|
| longdress | −0.649627% | −3.926224% |
| loot | −0.520286% | −0.277655% |
| soldier | −0.656577% | −0.401987% |

選択にcodecを繰り返し呼ぶoracle結果で、Networkが汎化して選べた結果ではない。ID headerはrate計算上の1byteを見積もったもので、配信用wrapperは実装していない。既存codecのstream単独をそのまま世界座標で読む条件とは違う。

### 4.4 品質を保って点数だけ減らしても大幅改善しない

元入力の各点の最近傍距離を完全維持するPruneを増やした。3入力で30%削除しても、G-PCC Cは+7.907 / +6.991 / +10.724%。Gは約2.18→1.86～1.87へ低下した。

したがって、**操作量やPrune量を大きくするだけでは解決しない。** 非効率なoccupancyを作らないことが必要である。

## 5. なぜ微小操作で−50%は依然難しいか

### 実測rateの支配項

longdressではOAの73,905内部node中、最深層が55,435 node。最深層のCEは151,132.94 bitsで、全CEの**79.86%**だった。これはCEによる層別診断であり、stream bytesを層別に実測したものではない。

最も高コストの葉318個（voxel数の0.15%）の元CEを全部0にしても、その自己符号量は全CEの**2.97%**にしか相当しない。0.30%相当の636個でも5.31%。

これはfrozen-contextでの診断で、祖先・近傍や多数の後続contextを変える効果まで制限する数学的不可能性証明ではない。それでも、少数の葉の自己コストを改善するだけで−50%に至ると期待する根拠は弱い。大半の葉へ良い影響が広がる構造を見つける必要がある。

### 現在のAmount/Gate表現との不一致もある

新しい良いOA planはAdjustを多く使う。longdressの単純条件付き法はAdjust 257/318=80.8%、window法は245/318=77.0%。「常に4:4:2程度が良い」という根拠は今回得られていない。

現行コードでは、noiseless・既定share 0.4/0.4/0.2・3操作有効の場合、gate residualはpairwise log-odds差を最大2程度に制限し、Fine平均も既定±0.35で制限する。pool不足・丸め・他の安全制約を除いたAdjust share上限は概算でgateのみ64.9%、Fineを含めても78.8%。[gate計算](../models/modules/structure_actuator.py:5240)

探索ノイズ中や設定変更時までこの上限を適用する話ではない。またwindow法の77%は範囲内なので、この制限だけで失敗を説明するものでもない。ただし、**総量を増やさずに配分を変える学習**を正しく許しているかは再検証すべきである。

## 6. 文献から見た、大きな改善への別ルート

前処理論文は、global scaling・学習voxelizationとcodec固有surrogateを組み合わせ、G-PCCの平均D1 BD-rate −38.84%を報告している。固定qs・0.15%編集とは別の条件。[Maらの前処理論文](https://arxiv.org/html/2508.01633v1)

2026年の *Octree-based Learned Point Cloud Geometry Compression: A Lossy Perspective* は、葉をbit単位で一部だけ伝送し、残りをdecoderで予測する方式を提案し、平均D1 BD-rateでG-PCC比−49.17%、OctAttention比−74.50%を報告する。比較G-PCCはTMC13v14であり、本実験のv23とは異なる。**これは既存codecへの微小な前処理だけではなく、decoder側も変更する別方式。** 報告値を本実装の達成値とは扱わない。[Zhengらの論文・方式と実験条件](https://arxiv.org/html/2603.28095v1)

この方向には、多数の葉で伝送しないbitsを作るという意味で、大きなrate削減の仕組みがある。ただし、そのとき予測を誤るvoxel数と品質を今回の小さい許容範囲へ抑えられるかは、別途検証が必要。

## 7. 推奨する研究設計

| 維持条件 | 次に研究するもの | 現時点の判断 |
|---|---|---|
| codec・微小操作予算を維持 | codec固有の条件付きrate、共同葉候補、context差分教師 | OAで約−1～−2%を実証。−50%は未証明 |
| 物理点群を維持、wrapper追加可 | 可逆座標変換をNetworkで選択 | 数%の可能性。多数codec呼び出しの削減と汎化が課題 |
| 操作量の拡張を許可 | 学習voxelization・複数scaleのRD最適化 | 大きな差は実在するが、元の微小操作要件とは分離 |
| 新しいcodec/decoderも許可 | 非葉は保護し、葉の部分伝送＋予測 | −50%級を狙う構造的な別研究。今回未実装 |

### 現行構成を伸ばすなら、最優先はcodec別のrate教師

- **OA:** frozen checkpointと実整数CDFから候補教師を作る。window法のような高価な探索をofflineで行い、Where/Amount/Gateの共同方策へ蒸留する。候補poolは全葉から構成し、運用時は少数候補・1回程度のActual確認へ圧縮する。計算費用が本当に減るかは別途測る。
- **G-PCC:** OA教師やoccupancy histogramを流用しない。同梱encoderの `encodeOccupancyFullNeihbourgsNZ` / adaptive OBUF / `SchroContext.probability` 等の実contextに対応した教師を作る。実装するなら別の研究用encoder・別ファイルで計測し、元のG-PCC binaryもSparsePCGC経路も保持する。[同梱G-PCC encoder](../../compress/octree/G-PCC/tmc3/geometry_octree_encoder.cpp:815)
- **G-PCCの相互作用:** adaptive contextが後続へ及ぶため、OAのwindow分離をそのまま仮定しない。まず小数のcounterfactualをfull Actualで測り、符号・順位の一致を検証してからsurrogateを使う。
- **共通方策:** 3操作を別々のtop-Kとして固定配分で合成せず、同じoccupied集合と総操作予算の下で共同決定する。総操作率と変更cell率、個別割合を別記録する。検証ではno-opを含む実際の効果を示す。

まず「0.15%程度の予算・画質を維持して、未使用系列でも安定してActual負値」を再現し、その後に大きい削減を狙う。4点以上のRD曲線を同じD1/D2区間で比較し、固定qsでのCとBD-rateを両方報告する。現在の結果から−50%収束を保証する設計変更は提案できない。

## 8. 成果物・限界

- [codec別の独立実験器](../experiments/octree_codec_research.py)
- [window相互作用・CDF最適化](../experiments/octree_window_research.py)
- [teacher一致・rate集中度の検証](../experiments/octree_teacher_validation.py)
- [全trial CSV](../experiments/results/20261017_octree_codec_research/all_trials.csv)、[結果集計](../experiments/results/20261017_octree_codec_research/summary.json)、[hash manifest](../experiments/results/20261017_octree_codec_research/manifest.json)

保存trialは268行（再測定・baselineを含む）、別途window検証2例とteacher照合を保存。window結果の編集座標はNPZにも保存し、`(canonical_voxels + origin)*qs` で物理座標へ戻せる。

独立実験の初期run `octree-oa-leaf-20261017` は候補評価の重複計算を直すため自分の実験processだけ中断した。そこでは保存済みbaseline・対照値のみ利用し、葉の実験は別runで再実行した。実行中のユーザー訓練processは触っていない。

追加単体テスト6件でteacher bits一致、causality、window外確率の不変性、整数CDFとstream長の対応、48変換の厳密逆変換、削除の幾何制約を確認した。G-PCC/OctAttentionとも長期訓練は今回していない。したがって、学習・汎化・収束改善の実証とは呼ばない。

**まとめ: 小さい予算で改善する具体的な経路は前回より明確になった。OAでは実context評価により約−1.3～−1.6%を実証した。一方、G-PCCに同じ尺度は通用せず、−50%には「多数の葉のrateを下げる仕組み」と操作・品質制約を両立する追加研究が必要である。**
