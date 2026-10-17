# Context-aware Voxel編集のActual codec oracle研究

2026-10-17。監査対象commit: `fdc313b84acaf00585ce12e33745012693de657f`。

既存ファイルを変更せず、独立CPU実験を実施した。以下は指定された「原因 → 仮説 → 実験方法 → 数値結果 → 原因分析 → Networkへ実装する場合の具体案」の順で報告する。学習済みNetworkのcheckpointは読み込まず、本訓練への組込み・再学習・再起動も行っていない。

## 1. 原因：現在の経路で確認した制限

### 1.1 実装を再監査したデータフロー

```text
argsのcodec選択・configure_geometry_codec
  → geometry_octreeのcodec別量子化・full-cloud context
  → geometry_octree_proposals
  → geometry_joint_proposals：単点候補、sibling/subtree bundle、操作別pool
  → actuator：実行可能性filter、Proposal subset、Actor候補score、Amount/Gate/Fine
  → geometry_joint_policy：共通状態でpriorを更新し、requested_counts内で選択
  → occupied集合編集・codec用座標へ復元
  → loss._encode_actual_batch
      G-PCC：_GPCCActualEncoder → GPCCGeometryEncoder → native tmc3 → stream bytes
      OA：_OctAttentionActualEncoder → Octree sequence → Transformer → 整数CDF → 算術stream
```

確認した主要箇所:

- [geometry_joint_proposals.py](/home/maejima/MasterEx/myNet/models/utils/pointcloud/geometry_joint_proposals.py:15)：初期総率・操作share・reserveからpool数を決め、親をsampleし、priorで候補を順位付けする。
- [geometry_joint_policy.py](/home/maejima/MasterEx/myNet/models/utils/pointcloud/geometry_joint_policy.py:24)：3操作を共通stateへ適用し、影響候補のpriorを再計算。ただし操作別 `requested_counts` とProposal maskが選択範囲を制限する。
- [geometry_octree_rate.py](/home/maejima/MasterEx/myNet/models/utils/pointcloud/geometry_octree_rate.py:14)：既定empiricalはlevel・orientation別のoccupancy分布。**非空Leafの変更では、別parentのnative予測確率が変わる効果を表現していない。** `neighbour_binary` は6近傍等を使う別の近似で、native OBUFではない。
- [structure_actuator.py](/home/maejima/MasterEx/myNet/models/modules/structure_actuator.py:5640)：Actor logit、Critic、不確実性、Proposal headを扱う。Criticを使うsubset分岐はあるが、geometry joint selectorの最終scoreへnative per-node確率が直接渡されるわけではない。
- [compression.py](/home/maejima/MasterEx/myNet/models/utils/loss/compression.py:853)：実行後点群をcodecへ渡す。候補priorと、選ばれたplanに対するActual教師値は別経路。
- [gpcc_tmc3.py](/home/maejima/MasterEx/myNet/models/utils/compression/gpcc_tmc3.py:98)：絶対座標をroundしてbbox正規化し、geometry-only native符号化を実行。
- [geometry_codec.py](/home/maejima/MasterEx/myNet/models/utils/config/geometry_codec.py:102)：OAは明示的なoverrideがなければ `octattention_actualcode=True`。**args定義のFalseだけを見て、現在の通常設定が算術符号化をしないと判断するのは誤り。** native OA量子化はraw-min-first。

これは現在のディスク上コードの監査であり、長時間稼働中processがどのcommitで起動したかまで断定するものではない。

### 1.2 候補生成と予算の実測

同じ入力から現行factoryを独立したローカルargsで再実行した。Networkの学習済みsubsetではなく、初期設定のpool監査である。

|入力|Add / Prune / Adjust pool|非空1-parent候補|拡張合法maskと一致する候補|
|---|---:|---:|---:|
|longdress 1051|848 / 848 / 424|1,672|722|
|redandblack 1451|795 / 795 / 398|1,582|653|

今回探索した合法maskは境界除外前で552,319 / 540,631候補。5% Context planが使ったmaskのうち、現行初期poolにもあるものは **59/9,291＝0.635% / 73/8,865＝0.823%** だった。候補生成側にも大きな違いがある。

また通常の幾何ルートは `geometry_octree_max_edit_ratio=0.0099`、既定Amount binsは0.05 / 0.15 / 0.30 / 0.60 / 0.90%。今回の3～5%実験は**通常訓練の上限を変えず、独立実験だけで広げた条件**である。

### 1.3 G-PCCのper-node確率を取得できるか

既存wrapper、native `--help`、`geometry_octree_encoder.cpp`、`entropydirac.h`を確認した。既存API/CLIにper-node確率のdump口は見つからなかった。

native内部では、近傍・既符号化bitからOBUF contextが変化し、確率がthresholdで補正された後にarith coderへ渡る。推論bit・直接符号化等もある。encoder側で単純にmask確率を読むだけではnative costを再現できない。

今回、codecのcopy buildやdebugger instrumentationは行わず、**全点群のnative Actual encode差分だけをG-PCC教師値にした**。empirical / neighbour prior、OA scoreをG-PCCのActual確率・per-node costとは呼ばない。将来instrumentするなら、experiments以下のcopyでnode識別と補正後・適応更新前の確率をtraceし、元binaryとのstream一致検証が必要である。

## 2. 仮説

1. 編集Node自身のgainが正でも、未編集Nodeのcontext損失でtotal gainが負になる。
2. 元contextでの自己gainより、編集後contextを含むtotal整数CDF gainの方がActual順位をよく説明する。
3. 0.15～0.30%という小予算だけでは、編集可能性を過小評価する。1～5%で改善幅は拡大し得る。
4. 親削除・複数parent・新parent作成は非局所contextを変えるが、Node数減少だけではrate改善を保証しない。
5. 自己gainを犠牲にしてcontextを改善する候補を提案できれば、単なるscore差替え以上の余地がある。ただしcodec間で同じ候補価値とは限らない。

## 3. 実験方法

### 3.1 条件・品質・符号量

- 入力はlongdress 1051 / redandblack 1451。主比較はG-PCCと同じ `unique(round(raw/2))` の共通grid。OAへはその整数座標をqs=1で渡す。別途native OAの `round((raw-raw_min)/2)` でも対照実験した。
- `C = 100 × (edited_bits / baseline_bits − 1)`。負が改善。G-PCCはnative全stream、OAは現行teacherと同じ実算術payload。OAのモデル・node数等の外部情報を含む完全配信payloadではない。
- Node costは実coderと同じ16-bit整数CDFの `−log2(freq/65536)`。これはNodeごとの条件付き符号長であり、共有arith stateの各Nodeへ出力byteを一意に割り当てた値ではない。最終Actualは別にstream bytesを測る。
- Add / Pruneは各1点、Adjustは各軸最大1 voxelの移動を1操作。占有cell差分率は `(Add+Prune+2×Adjust)/N` として別記する。予算は `round(N×率)` なので表示率に1 voxel以内の丸め差がある。
- 全ての採用planで **D1相当最大方向MSE、対称二乗Chamfer、両方向最大距離がno-op baseline以下**を要求。品質を外れたcandidateも診断用には符号化したが成功扱いしない。D2・属性・知覚・BD-rateは未評価。
- baselineからの品質維持条件であり、探索の全更新でGも単調低下するという条件ではない。

### 3.2 同じ候補集合で比較した4方式

|方式|使用するscore|
|---|---|
|現行prior|現行のempirical gain・geometry proxy・tanh logitを拡張maskへ適用|
|Node自身のみ|元contextで、編集symbolだけ置換した整数CDF gain|
|Context込み|編集後の影響windowを再forwardした総整数CDF gain|
|Actual oracle|同じ品質条件・予算内の同じ有限plan集合を実符号化して最小streamを選択|

「現行prior」はActor/context head＝0、noise＝0の統制比較。**学習済みNetworkのscoreや、現行policyのrequested_counts充足動作を再現したものではない。** 新しい複合maskに対するgeometry proxyも独立実験で一般化した。checkpointを読む必要のある比較は実施していない。

OAのNode / Context scoreをG-PCCにも転用してActualを測ったが、その2列は**cross-codec転用の診断**であり、G-PCC自身のNode / Context確率ではない。G-PCC自身のtrue per-node scoreは取得していない。

### 3.3 探索空間

1. 単一操作、複数操作の8-child mask、物理的に隣接する2 parent、親削除、4³ / 8³ source領域を含む**計152 candidateレコード**を個別に測定。一部は同じ編集が異なるsource領域として重複するため、152種類の独立な点配置とは数えない。
2. Leaf maskの共通poolから、windowごとに操作上限1 / 4 / 16 / 64 / 192 / 384のmenuを生成。編集後contextでmaskを再提案する反復も追加。no-opを含む同一menuから、prior / Node / Contextの3目的でbudget DPを解いた。
3. 各目的のplanを両codecでActual encodeし、同じ有限full-plan集合上のoracleを比較。予算は **0.15 / 0.30 / 1 / 3 / 5%**。
4. G-PCCはさらに、現状態の混合操作windowを置換するたびに**全点群をnative encode**する探索を実施。OAのwindow独立性をG-PCCへ仮定していない。
5. 凍結OAの入力embeddingに対するtotal CE勾配を候補生成のヒントに使用。**CEは提案専用**で、採択は必ず再計算した整数CDF gain、最後にActual streamで確認。optimizerもcodec weight更新もない。
6. 固定Leaf構造に限らない対照として、4³ / 8³領域内のrounding tieをfloor / half-upへ共同変更。原入力の最近傍被覆を保ち、new parent作成も許可した。1 / 3 / 5%の計24 planをActual測定。

Leaf最適化はboundary rowを除外して推論windowを独立化した。各windowにも逆方向geometry条件を課す保守的filterがあり、**別windowとの幾何誤差相殺や全mask組合せを探索したわけではない**。DPは生成済みmenu内のCDF目的についての最適化で、全voxel配置の大域最適性は保証しない。

## 4. 数値結果

### 4.1 1操作で「自身改善・周囲悪化」と「周囲も改善」の両方を確認

longdressの、どちらもAdjust 1操作・Node数不変・mask population不変の例。

|測定値|自身改善・total悪化|自身と周囲が改善|
|---|---:|---:|
|mask: 前→後|10111011→11110011|10110111→11110101|
|Node自身のsoftmax確率: 前→後|0.065694→0.094495|4.378e−8→0.362255|
|Node自身のCDF bits: 前→後|3.933574→3.409179|16.000000→1.470508|
|自身の削減|+0.524395 bits|+14.529492 bits|
|未編集Nodeの削減|**−6.985543 bits**|**+3.963055 bits**|
|影響Node数: CDF cost / 確率|304 / 345|52 / 57|
|合計CDF削減|−6.461148 bits|+18.492546 bits|
|実OA stream削減|−8 bits＝悪化|+24 bits＝改善|
|同じ編集のG-PCC削減|−16 bits|+8 bits|
|対称Chamfer|2.178329270|2.178319831|
|D1相当MSE|1.500576505|1.500576505|
|最大距離 GT→rec / rec→GT|1.732051 / 1.732051|1.732051 / 1.732051|

悪化例では、未編集mask 207の確率が0.023162→0.001630へ落ち、このNodeだけで3.821729 bits悪化した。改善例では、未編集mask 171が0.037019→0.187109となり2.336768 bits改善した。操作種・編集点数・自己確率だけでは区別できない。

悪化例の影響距離中央値は14.56 grid、改善例は8.60 grid。window内位置はそれぞれ0.663 / 0.944だった。ただしサンプル全体に普遍的な位置・密度規則があるとは結論しない。たとえばredandblackのcontext改善群はwindow前方にも存在した。

品質条件を満たす単一操作のうち、「自身改善・context悪化」は20 / 14件、「自身もcontextも改善」は6 / 4件。これは意図的に層化した有限sampleであり、全候補中の発生確率ではない。

### 4.2 同じcandidate集合における順位相関

品質条件を満たすcandidate全体、longdress n=51 / redandblack n=45のSpearman相関。

|scoreとActual削減の相関|longdress|redandblack|
|---|---:|---:|
|現行prior → OA Actual|0.373|0.490|
|Node自身のみ → OA Actual|0.562|0.364|
|Context総CDF → OA Actual|**0.980**|**0.977**|
|OA Context総CDF → G-PCC Actual|0.544|0.062|

Context総CDFとOA Actualの高相関は、同一coderのCDFを使うことから予想される整合性であり、学習済みQθの精度ではない。G-PCCでは単一candidateのActual差分が0 bitsの割合も33.3 / 22.2%あり、byte単位のteacher量子化も存在した。

### 4.3 操作予算を1～5%へ拡大した4方式比較：OA Actual C

|入力・予算|現行prior|Node自身のみ|Context込み|Actual OA oracle|
|---|---:|---:|---:|---:|
|longdress・0.15%|−1.022%|−1.416%|−1.565%|−1.565%|
|longdress・0.30%|−1.476%|−2.463%|−2.739%|−2.739%|
|longdress・1%|−1.573%|−6.390%|−6.975%|−6.975%|
|longdress・3%|−1.573%|−13.996%|−15.705%|−15.705%|
|longdress・5%|−1.573%|−18.622%|**−22.039%**|**−22.039%**|
|redandblack・0.15%|−0.885%|−1.073%|−1.303%|−1.303%|
|redandblack・0.30%|−1.363%|−1.944%|−2.269%|−2.269%|
|redandblack・1%|−1.953%|−5.175%|−5.970%|−5.970%|
|redandblack・3%|−1.953%|−11.982%|−13.790%|−13.790%|
|redandblack・5%|−1.953%|−16.145%|**−19.324%**|**−19.324%**|

全セルは品質条件を満たす。**各方式・予算ごとの実操作率、cell差分率、G、D1、両方向最大距離は[全50行の数値表](/home/maejima/MasterEx/myNet/experiments/results/20261017_context_oracle_summary_v2/tables.md)に併記した。**

予算は編集目標ではない。現行prior目的のDPは、予算を広げても実操作率0.3577 / 0.4624%で止まった。これは実行中NetworkのAmountを測定した値ではなく、今回の統制比較の選択結果である。

### 4.4 同じplan集合をG-PCCで符号化した結果

中2列はOA-derived scoreで選んだplanの転用結果。G-PCC自身のper-node scoreではない。

|入力・予算|現行prior選択|OA Node選択|OA Context選択|Actual G-PCC oracle|
|---|---:|---:|---:|---:|
|longdress・0.15%|−0.389%|−0.389%|−0.341%|−0.389%|
|longdress・0.30%|−0.256%|−0.373%|−0.389%|−0.389%|
|longdress・1%|−0.181%|−0.186%|−0.298%|−0.389%|
|longdress・3%|−0.181%|+0.112%|−0.703%|−0.703%|
|longdress・5%|−0.181%|+0.165%|−1.230%|−1.230%|
|redandblack・0.15%|−0.373%|−0.257%|−0.403%|−0.403%|
|redandblack・0.30%|−0.478%|−0.347%|−0.408%|−0.478%|
|redandblack・1%|−0.468%|−0.252%|−0.609%|−0.609%|
|redandblack・3%|−0.468%|−0.906%|−1.797%|−1.797%|
|redandblack・5%|−0.468%|−1.425%|−2.673%|−2.673%|

oracleは少量planも選べる。longdressでは1%上限でも約0.15%のplanを選んだ。候補集合のoracleであって、全配置の最適解ではない。

### 4.5 Contextを考慮した提案の作り直し

凍結OAの入力embedding勾配から候補maskを再提案し、総整数CDFが改善したwindowだけ採択した。

|入力|実操作率 / cell差分率|Add / Prune / Adjust|OA C|同じ点群のG-PCC C|G: baseline→編集|D1|最大距離: 両方向|
|---|---:|---:|---:|---:|---:|---:|---:|
|longdress|4.9913% / 7.3181%|2494 / 3152 / 4930|**−22.713589%**|−1.118211%|2.178329270→2.146671502|1.500576505|各1.732050808|
|redandblack|4.9987% / 7.2165%|2392 / 3129 / 4403|**−20.063245%**|**−2.884906%**|2.175526326→2.146526521|1.501114399|各1.732050808|

longdressのCDF gain内訳は、seed planで「自身47,273.586、未編集側−5,694.904」、追加探索後は「自身46,692.030、未編集側−3,840.905」bits。**自身の削減は減ったが、周囲の悪化を大きく抑えたためtotalとActualが改善した。** これは今回の中心仮説を直接支持する。

ただし5%操作は約7.2～7.3%の占有cell差分を伴う。現在の通常上限0.99%と同じ編集量で−20%出せた、とは主張しない。codec weightの前後hashは両実験とも一致した。

### 4.6 G-PCC Actualを直接使った追加探索

混合操作windowの各置換を全点群Actual encodeで判定した。少量seedではlongdress −0.4313%（150 unique encode）、5% seedでは **−1.2780%**（96 encode）、redandblackは **−2.6785%**。品質はいずれもbaseline以下。

longdressの最良G-PCC点群は148,320 bits、G=2.147103330、D1=1.500576505、最大距離は両方向√3。redandblackの全実験中の最良G-PCCは前節の凍結OA由来候補で154,312 bits、−2.884906%だった。Actual探索の近傍・初期値依存性もあり、native oracleの全配置最適性は保証していない。

### 4.7 複数parent・上位構造・new parent作成

- 隣接parentの共同maskと、1～3階層相当のsource領域を個別に評価した。longdressの1点Pruneでは、3個の内部Nodeが消え、「編集側+0.492412、未編集側+52.768403＝total +53.260815 bits」、Actualは56 bits削減。未編集costが変わったNodeは66,965個。
- 一方で同じ上位構造変更がtotal −29 / −38 bits等に悪化する例もあった。sequenceが詰まり、BPTT window境界も変わるので、これは空間近傍だけの効果ではない。単発gainを操作数倍して期待してはいけない。
- 品質安全な複数階層patchのnode数優先planは1%でOA −0.920 / −1.184%、G-PCC −0.234 / −0.161%。候補を広く組むと実操作率約2.4～2.5%でG-PCC +2.5～+3.5%へ悪化した。no-opより悪いplanは採用しない。
- 局所voxelizationはnew parentを実際に作成した。5%上限・floor・4³領域のlongdressは実3.8086%、329親を作成・2966親を削除、OA −3.621%、G-PCC **+2.066%**。G=2.155201611、D1=1.500576505、最大距離は両方向√3。
- redandblackの同条件は実3.6947%、335親作成・2781親削除、OA −4.812%、G-PCC −0.010%。G=2.156380890、D1=1.501114399、最大距離は両方向√3。局所voxelizationのG-PCC最良は別の約1% planで−0.351 / −0.821%。

Node数・親数・規則的roundingへの変更はActual改善の代用にならなかった。ただし試したvoxelizationは2種類のtie-breaking templateに限られる。

### 4.8 native OA gridとbaselineの対照

共通gridの結果だけに依存しないよう、longdressでnative OAのraw-min-first gridを新規生成して再実験した。元rawを既存OA teacherへ渡したbaselineの算術bitsと一致した（188,072 bits）。

5%上限では、現行prior −1.3867%、Nodeのみ −17.1254%、Context込み／Actual OA oracle **−20.6517%**。Gは2.176165250→2.144361261、D1=1.499599776、両方向最大距離=√3。したがって「共通gridを使ったからだけ−20%出た」という結果ではない。

G-PCCのbaselineも測った。raw integer gridの557,528 / 590,776 bitsに対して、qs=2 baselineは150,240 / 158,896 bits、既に **−73.052 / −73.104%**。ただしGは0→約2.18へ悪化しているので、これは品質維持の成功ではない。今回のLossで求めているのは、このqs=2 baselineからの**追加**削減であり、rawからの総削減率とは分母が異なる。

## 5. 原因分析・30～50%の可能性に対する結論

### 5.1 「1%前後しか余地がない」はOAについて誤り

少量編集だけで結論するべきではなかった。今回、OAはLeaf maskだけでも約5%操作で19～22%、追加探索で20～23%のActual削減を確認した。Context scoreはNode自身だけのscoreより5%条件で約3.2～3.4ポイント改善した。

一方、現行priorのcold-start順位、初期poolの狭さ、通常のAmount/changed-cell上限は、この大きい探索範囲と異なる。**候補外の配置をQθだけで選ぶことはできない。** 同時に、今回の統制prior比較を実行中Networkの性能・勾配・収束原因そのものと読み替えてはいけない。

### 5.2 編集自身の確率最大化では不十分

同じAdjust、同じmask population、自己確率上昇でも、未編集Nodeの損失によりtotalの符号が逆転した。単なるmask類似度・density・Node数ではその影響を説明できなかった。入力embeddingから周囲の影響を提案へ反映すると、自己gainを犠牲にしてもActualをさらに削減できた。

ただし良い候補の全てに「周囲も改善」が必要という意味ではない。周囲が多少悪化してもtotalが十分改善する候補は有効である。

### 5.3 G-PCCは別問題として残る

OA total scoreとG-PCCの相関は系列によって大きく異なり、G-PCCの最良追加削減は今回 **1.28～2.88%**。native Actualによる近傍探索を追加しても30%には近づかなかった。empirical priorをnative確率へ読み替えることもできない。

G-PCC向けQθの教師はnative全stream差分にする必要があるが、評価器を学べば自動的に50%削減できるという証拠はない。nativeで良い共同配置をさらに発見する研究と、見つかった価値を学ぶ研究は分けるべきである。

### 5.4 30～50%級の配置は存在するか／Leafでは不足か

**今回調べた有限pool・menu・局所探索・品質条件では、30～50%級のplanは見つからなかった。全配置空間に存在しないと証明したわけでもない。**

- 実際に確認できた水準はOA約20～23%、G-PCC約1～3%。
- Leaf-levelだけでOAに大きい余地はあるため、「Leafでは何をしても約1%」とは言えない。
- より広いpatch / topology / 局所voxelizationを試したが、単に広くするだけで30%へ届くという証拠もない。
- 未探索には、全maskの複雑な共同組合せ、windowを跨ぐ幾何誤差相殺、別のparent生成配置、より高度なvoxelizationが残る。
- 2フレームのoffline oracleであり、多系列・複数RD点での汎化、固定操作率での大域最適性、学習Networkによる再現は未確認。

したがって「30～50%を見込んで本訓練へ直ちに新方式を入れる」段階ではない。一方で、特にOAではPθ/Qθを独立に検証するだけのActual改善余地が確認できた。

## 6. Networkへ実装する場合の具体案と価値判断

### 6.1 Context-aware occupancy proposal Pθ

価値はある。既存Proposal headがpoolをsubset化するだけでは、pool外のmaskを作れない。

- 整数占有集合・祖先・近傍配置・raw点の被覆条件を入力に、8-child mask / 4³・8³ patchの編集後配置を共同提案する。
- codecの系列順・window位置・量子化原点も特徴に持つ。3操作別quotaを先に固定するのではなく、最終配置から実行可能なAdd / Prune / AdjustとAmountを分解する。
- no-opと少量planを必ず含める。Amountを強制的に増減するscheduleで結果を作らない。
- 今回の500k超の候補全列挙や多数forwardはoffline教師生成用。通常訓練へ同じ計算をそのまま追加しない。

### 6.2 周辺context込み総Δbitsを予測するQθ

**OAでは有力な優先研究課題。G-PCCでは必要性はあるが、30～50%を達成する十分条件ではない。**

- Qθは `(元配置, 共同編集後配置, codec/config, 文脈)` からtotal Δbitsと不確実性を予測する。既存Criticの入力・target拡張で実現できるかも比較し、重複headを無条件に増やさない。
- OAは整数CDFのself / unchanged-context / topologyを補助targetにできる。G-PCCは取得できていないper-node確率をtargetにせず、全点群Actual差分でtotalを学ぶ。
- Proposer/Qθはencoder側なので将来Nodeの元配置も参照できるが、**codec自身のcausal確率モデルやdecoderは変更しない**。
- geometry予測だけで安全と判断せず、実行可能性とD1/G/max距離を検証する。scoreが不確実・gain非正ならno-opへ戻す。
- 評価指標はQのMSEだけでなく、候補順位、選択後Actual regret、geometry違反率、no-opに対する改善率。train/testは系列・RD点で分け、oracle label作成に使ったフレームを汎化成功と数えない。

次の段階は、**独立したPθ/Qθのoffline学習・held-out検証**が妥当。少量編集・現行上限を維持した検証と、約5%編集の検証は別々に報告する。今回の約20%改善を現行0.99%上限のNetwork性能として扱わない。本訓練への組込みは実施していない。

## 検証・成果物・制限

- [集計JSON](/home/maejima/MasterEx/myNet/experiments/results/20261017_context_oracle_summary_v2/summary.json)：比較、per-node/context内訳、native探索、voxelization、native OA grid、品質、pool overlap。
- [全50行の比較・品質表](/home/maejima/MasterEx/myNet/experiments/results/20261017_context_oracle_summary_v2/tables.md)。その他全candidateにもraw JSON内でG/D1/max距離を記録。
- [SHA256 manifest](/home/maejima/MasterEx/myNet/experiments/results/20261017_context_oracle_summary_v2/manifest.json)：274ファイル。主実験のper-node CDF差分NPZも保存。
- 独立コードは `experiments/context_oracle_*.py`、集計は `experiments/summarize_context_oracle.py`。実行引数は各新規resultsディレクトリの `arguments.json`。
- CPU環境は `/home/maejima/miniconda3/envs/myOA/bin/python -B`、`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`。
- 関連28テスト合格。部分window再計算と全sequenceのモデル・stream hash一致、no-op、整数CDFのfloor、budget DP、操作数・幾何を検証。
- G-PCCの採用full planはnative decoderで座標一致を確認。OAはlargest各score planと追加探索planで、fresh全sequence encodeのstream hash一致を確認した。
- OAの独立確率による自律decodeも、既存decoder関数本体を変更せずCPU wrapperから実行した小control（64 Node / 119 voxel）でsymbol・座標一致。ただし**全フレームの自律OA decodeは未実施**。full-frame roundtripの多くはencoder計算済み確率を使う検証である。
- train.py / models Python、codec source/config、G-PCC binary、OA codec checkpointを含む15,614ファイルの前後hash一致。追加探索のcodecモデルstate hashも一致。既存訓練processは継続し、既存ログ・checkpoint・設定は上書きしていない。
- raw実験は全完走。集計の初回のみ相対pathのmanifest書出しで停止したため、元成果物を上書きせず新しい`summary_v2`へ再集計した。正式な結果索引は上記v2。
