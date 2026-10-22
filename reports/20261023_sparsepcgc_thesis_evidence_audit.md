# 修士論文前のSparsePCGC実測監査：符号量・構造特徴・損失・RD

調査日：2026-10-23。対象は**現行コードと、明示した既存381 checkpointの固定Validation出力**。実行中訓練processのメモリ内重みを測定したものではない。

## 結論

| 優先項目 | 今回の確認範囲 | 判定 |
|---|---|---|
| 1. 符号量削減の分解 | 4 frame × Baseline/MINE、6階層×8 Stage、計384 Stageを実符号化・実復号 | 入力点数減少だけでは説明できない。固定した元CDFでの占有記号変更が大きな利得。未編集記号のContext変化は4件とも悪化側 |
| 2. 構造特徴と実Codec | 実Network forwardから40chを取得。実整数CDF NLLへ尺度・座標を対応付け | NLL proxyとの相関はほぼ0。17chが今回の4 frameで定数。さらに特徴群間のvoxel行順不一致を実測 |
| 3. 損失・勾配 | 現行ソースとSHA一致の独立8 Step計測を再集計。式と勾配置換を再読 | Actual/Proxy信号の接続はあるが、単純な加重平均の最小化とは異なる。原因診断器・Structure Policy予測器は現設定で迂回 |
| 4. RD性能 | 学習済み出力1 frame、6設定×2系列、実stream・自律復号・MPEG D1 | D1 BD-rate −0.347016%。曲線交差あり。D2は正式GT法線がなく未計算。全データセットの性能証明ではない |

**論文執筆前の最優先事項は、構造特徴のvoxel行対応の是正と検証。その後で特徴ablation・複数系列のD1/D2 BD-rate評価を行うべきである。今回だけで「40特徴が有効」「予測確率改善が主因」「品質を維持して数%改善」と結論しない。**

## 0. 条件・保護・再利用の区別

- Network：`/data/maejima/log/20261020/MyNetwork_train/pretrained/043722_SparsePCGC/381.pth`。SHA256 `bb0feb80e9a230dd716b85cc246e7acd77c2c7f0cf76065c894274108caa49c2`。
- 対象：UVGのBlueBackpack 0100/0101、BlueSpin 0100/0101。既存の固定Validation分割。同系列内holdoutであり未知系列汎化とは呼ばない。
- MINE：完了済み`mature_current_s21_storage_v2/validation_000`で**実際にNetworkが生成し、Actual encodeへ渡した点群**を再利用。独立editorで改善案を新規選別したものではない。
- 追加計測：現在のNetworkで同じsnapshotを再評価して構造特徴を取得。固定Validation C=−1.9051239039501278、G=0.024276533164083958は既存結果と完全一致。今回、新規訓練はしていない。
- Codec：SparsePCGC dense_lossy、AE=0/SR=1（native bit depth 9、m=8）をStage分解に使用。CPU、PyTorch 1.10.0+cu113、MinkowskiEngine 0.5.4、OMP/MKL/PyTorch各2 thread。
- Codec計測は独立subprocess内で既存関数をwrapper化。本体Codecソースは編集しない。算術encoderの元処理を呼び、その入力確率・記号・返却streamを観察する。
- `train.py`、`test.py`、`test_BDrate.py`、modelsおよびSparsePCGCの計167 sourceを開始時SHAで保護し、終了時の差分0。既存checkpointとruntime sidecarも過去の記録SHAと一致。既存ログ・実行中processを変更していない。
- 大きな成果物は `/data/maejima/log/independent_experiments/20261023_thesis_audit_owVPnq/`。repoから `experiments/results/20261023_thesis_evidence` をsymlinkしている。

## 1. 実際に符号量が減る原因

### 1.1 測定したもの

各native lossless階層内の8 Stageで、候補座標、占有記号、sigmoid確率、**実算術coderと同じ16-bit整数CDFへの変換結果**、浮動小数NLL、整数CDF NLL、返却された実算術stream bytesを取得した。

整数CDFの量子化は独自の近似式だけで済ませず、実際の `torchac.torchac._convert_to_int_and_normalize` を呼ぶ。NLLは実際に符号化された記号のCDF区間幅に対する `−log2(width/65536)`。非占有記号も含める。NLLと実bytesは同じ値と仮定しない。

streamをunpackし、**復号した**粗い点群が入力粗点群と一致することを検査。その復号結果からSR復元し、最終復号PLYを保存した。encoder側の粗点群をdecoderへ横渡しして代用しない。

SparsePCGCが占有予測を算術符号化へ使うこと自体は原論文・公式実装とも整合する。ただし以下の数値は論文値の引用ではなく今回のnative実測である。[原論文](https://arxiv.org/abs/2111.10633)、[公式実装](https://github.com/NJUVISION/SparsePCGC)

### 1.2 4 frameの結果

候補数は、全符号化階層・Stageの占有／非占有候補の合計。元の点数や占有点数と区別する。

| frame | 入力点数 前→後 | 符号化候補数 前→後 | 整数CDF NLL 前→後 [bit] | 完全wrapper stream 前→後 [bit] |
|---|---:|---:|---:|---:|
| BlueBackpack 0100 | 345000→344593 | 228336→227768 | 63187.291→61657.366 | 66680→65168 |
| BlueBackpack 0101 | 344613→344208 | 226896→226296 | 62666.041→61348.356 | 66200→64896 |
| BlueSpin 0100 | 179535→179442 | 119552→119336 | 30132.222→29563.978 | 33344→32768 |
| BlueSpin 0101 | 179437→179343 | 119584→119496 | 29885.907→29365.832 | 33096→32576 |

完全streamには、既存BD評価adapterの占有streamと、必要な点数・座標系などを実際にserializeしたside recordを含む。今回のside recordは各520 bit。学習用logical bitや論文containerと同一の定義ではない。

BlueBackpack 0100の階層別集計（各行に8 Stage）：

| 符号化階層 | 候補数 前→後 | 整数CDF NLL 前→後 [bit] | 算術payload 前→後 [bit] |
|---|---:|---:|---:|
| 0：SR後の最も細かい符号化階層 | 175376→174888 | 45071.405→43750.887 | 45112→43792 |
| 1 | 41312→41232 | 13597.897→13409.233 | 13632→13456 |
| 2 | 9024→9024 | 3339.990→3319.247 | 3376→3360 |
| 3 | 2056→2056 | 876.825→876.825 | 904→904 |
| 4 | 448→448 | 231.149→231.149 | 280→280 |
| 5 | 120→120 | 70.025→70.025 | 104→104 |

**階層2は候補数が同じでも符号量が変わっている。** したがって「対象数が減っただけ」ではない。

### 1.3 編集記号・Context・候補増減の分解

同じ階層・Stage・座標の候補を対応付け、共通候補を「占有記号変更」と「記号不変」に分けた。共通候補では、元のCDFで編集後記号を評価してから、編集後CDFへ変更する順序で分解する。追加／削除候補は別計上する。

これは**厳密な符号量会計**だが、実現可能な独立介入の因果実験ではない。CDFを先に変更する別順序なら交互作用の配分も変わる。

BlueBackpack 0100について、差を「編集後−編集前」とした結果：

| 項目 | ΔNLL [bit] |
|---|---:|
| 符号化候補の消滅・新規発生 | −55.786 |
| 元CDFを固定した占有記号変更 | −2753.194 |
| 記号を変更した候補のContext変化 | −9.693 |
| **記号不変の候補のContext変化** | **＋1288.749** |
| 合計 | −1529.925 |

対応する実stream差は−1512 bit。差約17.925 bitは算術終端・丸め等を含むpayload差で説明され、今回container/sideのサイズは前後同じだった。

未編集記号側で整数CDF確率が変化した候補は147583。4 frameすべてで、未編集記号のContext効果は悪化側（＋1288.75 / ＋1502.18 / ＋961.48 / ＋886.54 bit）。

したがって、今回支持される説明は、**「固定codecにとって高コストだった占有記号を変更する利得が、未編集側のContext悪化を上回った」**である。「周囲の予測確率が一斉に改善したから減った」とは言えない。

参考として、各Stageの `N × 平均NLL` を対称に分解すると、候補数項は全ΔNLL利得の約9.8 / 12.2 / 9.4 / 4.0%。ただし残りをそのまま「確率予測が良くなった割合」と呼んではいけない。平均NLLの変化には占有記号の変更・候補集合の構成変化も含まれる。

### 1.4 学習用Actualの4 bit重複を発見

現行`encoder_multiple._normalize_result`はnative coderの点数情報を含むlogical値を保持する。一方、`sparsepcgc_teacher_worker._build_encoder`内の互換wrapperは、その値を`main_bits`とみなし、さらに点数情報を加算する。

AE=0/SR=1では：

```text
BlueBackpack 0100 Baseline
物理main stream              66160 bit
native coderのlogical値       66164 bit（+4）
保存済みtraining teacher値    66168 bit（さらに+4）
今回の完全wrapper stream     66680 bit（66160+実side 520）
```

8件全てがこの関係と一致した。本体集計は今回修正していない。同一設定の前後bit差は変わらないが、絶対bit数と改善率は一致しない。微小な定数誤差であり、大きな性能差の主因とは言えない。**過去の「decoder-complete Actual」という名称だけで、論文用の完全物理bitstreamと同一視してはいけない。**

## 2. 構造解析モジュールと実Codec

### 2.1 実NLLへの対応方法

現行Networkの実Validation forwardから、入力xyz、40ch features、cause_targets、occupancy_nll_proxy、canonical voxel座標を保存した。raw leafに直接NLLを割り当てず、SR=1に合わせて2×2×2内で特徴を平均し、最初の符号化階層の**占有候補**へ座標照合した。空候補や上位階層まで含む総NLLとは別の解析である。

metadata座標を基準に対応付けた結果：

| frame | 対応した占有候補数 | occupancy_nll_proxy対実NLL Pearson | Spearman |
|---|---:|---:|---:|
| BlueBackpack 0100 | 88322 | −0.00334 | 0.01182 |
| BlueBackpack 0101 | 88120 | −0.00344 | 0.01447 |
| BlueSpin 0100 | 45602 | −0.00359 | 0.02592 |
| BlueSpin 0101 | 45584 | −0.00444 | 0.03307 |

**今回の条件では、occupancy_nll_proxyを実NLLの有効な近似とは扱えない。** 一方、single_proxyのPearsonは約＋0.376～＋0.401、mean_occは約−0.392～−0.384、sparse_proxyは約＋0.356～＋0.371だった。占有の疎密に関連する特徴には関連性があるが、編集価値や因果的重要性を証明したものではない。

### 2.2 40ch中17chが今回の全4 frameで定数

- 格子位相・中心化位相・snap変位の9ch
- density / curvature / anisotropyの3ch
- quant_merge / quant_density / quant_residualの3ch
- row_existとbit_entropyの2ch

整数grid入力では位相・snapが0なのは自然である。幾何3chは点数が既定の2048を超えるため計算を省略し0。`bit_entropy`は今回すべて1で、実符号化エントロピーの変動を表していなかった。

これは「17chを全条件で削除してよい」という意味ではない。入力量子化、点数、Codec設定が変われば有効になる可能性がある。40chの学習上の有効性を確定するには、後述の行対応を直した後で再学習を含むablationが必要。

### 2.3 特徴群間のvoxel行順不一致を検出

今回の実forwardでは、40chのxyz 3chは入力xyzの順序に100%一致する一方、child_idはmetadata座標の順序に100%一致した。両座標は集合としては同じだが、同じ行の座標一致率は0 / 0.000290 / 0.000557 / 0.001115%。

child_idを入力xyzから再計算すると一致率は約12.5%しかない。8子のIDが偶然一致する水準であり、単に座標系のスケールが違う問題ではない。

つまり、この4回の計測では**同じfeature rowに、あるvoxelの位置特徴と別voxelの占有特徴を結合している状態**を確認した。形状一致だけの検査では検出できない。下流のどの判断へどれだけ悪影響が出たかは、修正対照実験なしには断定できない。

次に修正するなら、Networkのnode順序変更後、SparsePCGCにもcanonical voxel keyに基づくmetadata整列を適用し、xyz、占有情報、Morton key、repair unit、候補source indexを一括して対応付ける。単に1つの座標配列だけをsortする修正は不十分。今回は調査依頼のため本体を変更していない。

相関CSVにはmetadata基準とinput_xyz基準の両方を保存した。特にmixed cause scoreの相関を単純な点単位の真値対応として扱わない。

### 2.4 原因スコアとActual削減量

現設定の原因スコアは予測器出力ではなく、疑似cause_targetsへカテゴリ重みを掛けて再正規化した値。4 frameの**点平均**スコアと実削減bitsのPearsonは、node＋0.875、context＋0.971、sparse＋0.988、lowprob−0.977などだった。

ただし**N=4、系列は2つ、点数・contentも異なる**。この高い相関を有意な予測精度や特徴重要度とは扱わない。1 frameのActual削減量を全voxelへ複製して巨大な標本数とする解析もしていない。

個々の候補・patchの原因スコア対Actual Δbitsは、個別counterfactual encode教師が不足しており未確認。今回保存したaligned candidate NLL savingも、総Actual bit削減量と同じものではない。

## 3. 損失は何を学習させているか

### 3.1 現行設定の役割

| 項目 | 主な設定・役割 | 注意 |
|---|---|---|
| 圧縮主目的 | w_com=3、compression_primary | native codecは微分しない。表示値と微分可能代理経路を区別 |
| 幾何 | cp_lambda_geom=150、閾値0 | 編集後点群とGTの幾何loss。最終復号D1/D2そのものではない |
| 幾何のbackward | cp_geometry_backward_scale=0.036 | forward係数150と別。直接幾何勾配の局所係数は、balance前なら5.4 |
| node / single補助 | cp_lambda_nodes=.03、single=.05 | 正規化された補助項でありnative各Stage bitsではない |
| actuator・編集量 | cp_lambda_actuator=.05、w_actuator=1、cp_lambda_op=0 | tail側balanceも存在。Actor RDにはchanged_ratioのtie-breakもある |
| 原因診断L_attr | w_attr=0、direct_octree_structure_targets=true | 学習した原因分類器ではなく疑似原因を直接利用する経路 |
| Structure PolicyのL_policy | w_policy=0、上記direct設定 | 操作種別の疑似教師を直接利用。Actual Actor方策lossとは別物 |
| Actual方策・Critic | policy_weight=.1等、同一frame/anchor比較 | 教師は3*C+150*max(G,0)+2*changed_ratio。Cだけのgradientではない |
| Proxy / Entropy | candidate distill等、entropy=.001 | Actor/Whereへの補助信号。native CDF NLL教師と同一ではない |

圧縮primaryの補助blockには自動balance（target ratio=.75）があり、各項の表示係数だけで実効寄与を判断できない。またActual方策lossなどを `10*(loss-loss.detach())` として加えるため、**表示Lossへの寄与0でもbackwardへの寄与がある**。

したがって「全体Lossは単純に(3C+150G)/153」「プロットの変化だけで各moduleの学習量が分かる」とは説明できない。詳細は`compression_primary_loss.py`、`train_flow.backward_only_scaled_loss`、`Network.discrete_policy_loss`、`train.py`の追加項を分けて記述すべきである。

### 3.2 実勾配と更新の再確認

完了済み独立8 Stepの訓練source SHAは**現行ファイルと全件一致**。今回はその計測を再集計したもので、新しい8 Step訓練を行ったと偽らない。以下はStep 2/4/6/8のsource gradient L2の最大値。sourceごとの最大が同じStepとは限らない。

| module | Actual RD/critic由来 max L2 | Candidate Proxy由来 max L2 | downstream圧縮由来 max L2 | 実更新が非ゼロのStep |
|---|---:|---:|---:|---:|
| Candidate Actor | .260926 | .042646 | 0 | 8/8 |
| Candidate Critic | .298455 | 0 | 0 | 7/8 |
| Plan Critic | 2.295075 | 0 | 0 | 8/8 |
| Proposal | .103826 | 0 | 0 | 1/8 |
| Add Where | .054253 | .001773 | .010363 | 8/8 |
| Prune Where | .071474 | .002259 | 16.307239 | 8/8 |
| Adjust Where | .033267 | .009739 | .002319 | 8/8 |
| Add Amount | 18.566546 | 0 | .015238 | 8/8 |
| Prune Amount | 133.497768 | 0 | .155236 | 8/8 |
| Adjust Amount | 27.341621 | 0 | .008065 | 8/8 |
| Feature Adapter | 2.8931e−8 | 5.3199e−11 | 6.7424e−8 | 8/8 |

ただし、これらのraw gradientが全てそのままoptimizerへ入るわけではない。決定headにはpolicy-only復元がある。

例：Step 4のPrune Whereは合成backward後L2=3.217633、policy系勾配置換後=.002259、clip後も=.002259。直接圧縮／幾何STEの大部分は最終決定更新へ残らない。一方、Step 4のFine Amount 3 headは既存clipにより約.12999 / .97268 / .19236となる。RMS、各source、restore・balance・clip前後の全値をCSV保存した。

**「圧縮の代理勾配が一度非ゼロ」だけでは、最終更新へ寄与した証拠にならない。** またActual codecを微分しているわけではなく、Actualに基づく比較教師・Critic回帰・score-functionによる更新である。比較groupやanchor測定StepによりActual方策gradientが0になる場合もある。

Point Transformer本体はencoder_0grad=trueで意図的に凍結。構造解析は主に固定統計処理。これらの勾配0を一律に不具合と扱わない。一方、AdapterとActuator/Actor/Criticには実更新が確認できる。長期性能改善まで保証する結果ではない。

## 4. 実復号点群を使うRD評価

### 4.1 プロトコル

- Original：BlueBackpack 0100、345000点、9-bit geometryのresolution=511。
- MINE：上記381 checkpointの固定Validation出力をそのまま再生。新たなNetwork推論回数は0。この評価は**保存済み学習済み出力のreplay pilot**であり、現在の`test.py`の推論設定まで実証したものではない。
- 各系列を同一6設定で実encode/decode。AE/SRは(1,0),(0,1),(1,1),(0,2),(1,2),(0,3)。入力MINEはレート間で固定し、都合の良い編集へ切り替えない。
- 全12測定で復号成功。Original GTと最終decodedをMPEG pc_errorで比較。bpp分母は両系列ともOriginalの345000。
- Rateはoccupancy.binとserializeしたgeometry.side.binの実bytes。学習用logical bitや元入力を含む中間ファイルの大きさをrateにしない。
- D1は対称pc_error mseF。重複処理・resolution等を固定。D2にはGT法線を指定するのが正規のtool利用である。[MPEG公式metric](https://github.com/MPEGGroup/mpeg-pcc-dmetric)

### 4.2 6点の実測

Chamferは両方向平均二乗距離の**和**、最大距離は両方向の最大値。いずれもOriginalと最終decodedの比較。学習用Gと混同しない。

| AE/SR | Baseline bits / D1 PSNR | MINE bits / D1 PSNR | Chamfer 前→後 | 最大距離 前→後 |
|---|---|---|---:|---:|
| 1/0 | 98200 / 70.4423 dB | 97248 / 70.2775 dB | .141386→.146068 | 1.7321→1.7321 |
| 0/1 | 66680 / 67.9023 dB | 65168 / 67.7684 dB | .253649→.261226 | 1.7321→2.0000 |
| 1/1 | 29192 / 63.8368 dB | 28992 / 63.8223 dB | .619962→.624406 | 4.1231→3.7417 |
| 0/2 | 21312 / 61.8897 dB | 21120 / 61.8868 dB | .968397→.971517 | 4.0000→4.0000 |
| 1/2 | 9256 / 56.6556 dB | 9240 / 56.6930 dB | 3.078693→3.089979 | 8.3066→8.3066 |
| 0/3 | 7424 / 55.0643 dB | 7408 / 55.0452 dB | 4.474545→4.498217 | 8.3666→8.3666 |

符号量は6設定とも減ったが、D1 PSNRは5設定で低下。特に訓練代表のAE0/SR1では−2.2675%のrate削減に対して約−.1339 dB。**同じ設定でrateが減ったことを、同等品質の改善と読み替えてはいけない。**

### 4.3 BD-rate

AkimaでPSNRに対するlog10(bpp)を補間し、共通区間55.0643～70.2775 dBだけを積分した。外挿・PSNR一致許容幅・測定点の補正はしない。

- **D1 BD-rate：−0.3470161510%**。
- 曲線交差あり。全品質域で改善したという意味ではない。
- 同じ6点のPCHIP感度確認：−0.3914948110%。差約.04448 percentage points。これは密な実測点による補間誤差検証の代わりではない。
- FAU-LMS bjontegaard 1.3.0の未変更wheelと100乱数曲線を新たに比較。Akimaの最大差1.4211e−14 percentage points。[参照実装](https://github.com/FAU-LMS/bjontegaard)
- **D2 PSNR / D2 BD-rate：未計算**。対象GTにはx/y/z/RGBしかなく、正式なnx/ny/nzがない。勝手な法線・D1代替を使用しない。
- 1 frame・2系列内holdoutのpilotなので、全データセットの平均BD-rateや未知系列汎化とは呼ばない。10～15点の密な実測も未実施。

## 5. 論文前の次の優先作業

1. **voxel行identityを修正・検証**。同一rowのxyz/occupancy/key/候補indexを一致させ、ランダムな入力順序変更に対するequivarianceと、固定Validation前後を検証する。本調査だけでは性能改善を保証しない。
2. **学習用rate集計を整理**。4 bit重複とlogical/physical/side情報を分け、従来ログとの比較条件を固定する。物理streamを論文評価の正とする。
3. **整列後に40chのablation**。まず定数・重複・scale不一致を区別。その後、再学習したfeature群別ablationと、patchごとのActual counterfactual labelで有効性を調べる。現在の相関だけで削除・採用を断定しない。
4. **学習目的と最終RDを接続**。現在のGは編集直後の品質であり、SR後のD1/D2の保証ではない。最終decoded品質を使う検証または教師を導入する価値を検討する。表示Loss、Actor RD教師、Validation objectiveは別々に記述する。
5. **正式RD評価を拡大**。承認したGT法線、独立系列、複数frame、6点＋密な点でD1/D2 BD-rate。`test.py`の推論policyが訓練・今回のValidation policyと同じかも確認し、identity-controlを提案手法実測と混同しない。

## 6. 成果物・検証

追加したのは`experiments/thesis_audit/`の独立コードと本レポート、resultsへのsymlinkのみ。以下はrepo内symlinkから参照できる。

- [全Stageの候補数・NLL・算術bits](../experiments/results/20261023_thesis_evidence/native_analysis/stages.csv)
- [対応候補の符号量分解](../experiments/results/20261023_thesis_evidence/native_analysis/aligned_decomposition.csv)
- [4 frame集計](../experiments/results/20261023_thesis_evidence/native_analysis/summary.json)
- [特徴相関・全channel統計・行identity](../experiments/results/20261023_thesis_evidence/feature_analysis/summary.json)
- [各特徴の相関CSV](../experiments/results/20261023_thesis_evidence/feature_analysis/correlations.csv)
- [原因スコアと実bitsの4 frame相関](../experiments/results/20261023_thesis_evidence/plan_correlations/summary.json)
- [損失設定と勾配・更新集計](../experiments/results/20261023_thesis_evidence/loss_analysis/summary.json)
- [全source勾配とrestore/clip前後](../experiments/results/20261023_thesis_evidence/loss_analysis/all_source_and_routing_gradients.csv)
- [6点RDのExcel](../experiments/results/20261023_thesis_evidence/rd_frame0/results.xlsx)・[RD評価JSON](../experiments/results/20261023_thesis_evidence/rd_frame0/summary.json)
- [検証と補間感度](../experiments/results/20261023_thesis_evidence/verification_v2/summary.json)・[実テスト出力](../experiments/results/20261023_thesis_evidence/verification_v2/tests.txt)・[成果物SHA256](../experiments/results/20261023_thesis_evidence/verification_v2/manifest.json)

分解のidentity、CDF端点、BD-rate関連の**39テスト合格**。最初の`native/`は別importクラスを観測してStageが0件、`native_v2/`はprivate CDF関数のimportで停止したため不採用。`native_v3/`のみを集計した。最初のverificationは「保存teacher値=native logical値」という仮定が4 bitずれて停止し、これが重複加算の発見につながった。測定値を合わせる補正は行わず、物理値を保持し、`verification_v2`で差の由来を記録した。

**到達点：符号量削減の実測分解と1 frameのD1 RD評価は成立。構造特徴の因果的重要性、行対応修正後の性能、正式D2、全系列・密点BD-rateは未解決。論文の結論ではこの範囲を超えないこと。**
