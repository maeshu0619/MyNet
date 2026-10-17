# Add / Prune / Adjustを同時に扱うOctree圧縮の追加研究

2026-10-17。基準commit: `deb783be385bd764d1bbb87e683655b2d9399ebb`。

## 結論

**3操作を別々に最適化して最後に足すのではなく、最終occupancy配置を共同提案し、編集後の点群全体で符号量・品質を評価する設計は重要である。ただし、それだけで−50%を達成できるという証拠は得られていない。**

今回、独立したCPU実験を追加した。4フレーム、総操作率約0.15%ではG-PCC −0.570～−0.884%、同じ編集点群のOctAttention −0.692～−1.344%。約0.30%ではG-PCC −0.705～−1.315%。採用planは幾何Gを低下させ、D1相当MSE・両方向最大点距離を維持した。

SparsePCGCを含む既存訓練コード、args、checkpoint、codecソース、実行中訓練processは変更していない。追加したのは `experiments/` の独立コード・実験成果物と本レポートだけ。最終確認でtracked差分は0だった。

## 1. 現在の設計で既に共同化されている部分と、残る制限

現在のコードを再確認した結果、「既存コードが3操作を完全に独立選択している」という説明は正しくない。

- [geometry_joint_policy.py](/home/maejima/MasterEx/myNet/models/utils/pointcloud/geometry_joint_policy.py:24) は3操作の候補を連結し、共通Octree状態を編集ごとに更新して限界gainを再評価している。
- 一方、同ファイル80行付近では操作別の `counts[op]` を超える候補を無効化する。共同選択であっても、自由な総予算だけの探索ではない。
- [geometry_joint_proposals.py](/home/maejima/MasterEx/myNet/models/utils/pointcloud/geometry_joint_proposals.py:44) の基本候補は単点Add / Prune / Adjustと一部bundle。全葉の任意の複合maskを列挙するわけではない。
- 例えばlongdressと既定の初期設定を式に代入すると、pool予算は848 / 848 / 424、基本候補を作る親は1,952 / 55,435、約3.52%。これは初期設定の計算例であり、実行中checkpointの全Stepの実測値ではない。
- [geometry_octree_rate.py](/home/maejima/MasterEx/myNet/models/utils/pointcloud/geometry_octree_rate.py:1) はempirical / neighbour-binary priorであり、native G-PCCの実際の確率モデルそのものではない。既定は [args.py](/home/maejima/MasterEx/myNet/models/utils/config/args.py:2087) の `empirical`。

G-PCC実装は近傍配置、既に処理した子、dynamic OBUF、適応確率を使う。周辺maskの出現頻度を改善してもActualが改善するとは限らない。今回、G-PCC内部の全bit確率を直接traceしたわけではないため、その値を測定済みとは主張しない。

## 2. 実験条件と「同時」の意味

- 入力: longdress 1051 / loot 1000 / soldier 0536 / redandblack 1451。
- 共通grid: `unique(round(raw / 2))`。G-PCCのqs=2と同じ整数gridを使い、OctAttentionには整数座標をqs=1で渡す。native OAの別の量子化順序を混ぜない。
- `C = 100 * (edited_bits / baseline_bits - 1)`。G-PCCは全stream bytes、OAは実算術stream bytes。CE推定値をActualとは呼ばない。
- 操作率の分母は未編集unique voxel数。Add / Pruneは1点、Adjustは各軸最大1 voxelの移動を1操作とする。bundle内の変更も全点数で課金し、相殺・重複を検査する。
- Gは元物理座標での対称平均二乗Chamfer。D1相当の最大方向MSEと両方向最大距離も評価する。D2、属性、知覚評価、BD-rateは今回の検証対象外。

今回の基本単位は「操作種別」ではなく**編集後の8-child occupancy mask**。各葉で255通りの非空maskを検証し、削除位置と追加位置を一体で決める。その後で実行可能なAdd / Prune / Adjustへ分解する。合計操作予算のみを設け、操作別の割当は固定しない。

元入力の各点が編集後maskでも同じ距離以内に被覆されるかを検査する。従来の独立実験のように「単独で削除可能な点」の集合へ先に固定しない。追加先voxelが削除部分を救う候補も残す。bbox保護、追加点の距離制約も設け、最後に点群全体のG・D1・最大距離で採否を判定する。

候補poolにはOA gainが負・0の候補も残す。OA確率、mask出現頻度、近傍maskとの一致、点数削減、およびそれらの混合から11種類の**共同plan**を作り、G-PCCで点群全体をActual encodeする。全mask組合せの大域最適探索ではない。

## 3. 候補生成の実測

| 入力 | 合法mask候補数 | 削除部分に追加先の被覆が必要な候補 | OA単独gainが非正の候補 |
|---|---:|---:|---:|
| longdress | 552,319 | 107,164 | 480,918 |
| loot | 588,370 | 127,658 | 518,751 |
| soldier | 795,429 | 136,600 | 689,015 |
| redandblack | 540,631 | 108,243 | 469,964 |

合計2,476,749候補。生成時間はCPUで約10～15秒/frame。全候補をActual encodeしたわけではない。11 profile × 2操作予算 × 4フレームの88共同planを比較し、8勝者の再符号化、24操作family ablation、4baselineを含めて主実験には124レコードがある。

## 4. 主結果: 幾何品質を維持した共同plan

### 総操作率約0.15%

各フレームで、品質条件を満たした11 planのうちG-PCC bitsが最小のものを示す。同じ点配置をOAでも圧縮した。

| 入力 | Add / Prune / Adjust | G-PCC bits: 前→後 | G-PCC C | OA C | G: 前→後 |
|---|---:|---:|---:|---:|---:|
| longdress | 103 / 57 / 158 | 150240→148912 | −0.883919% | −1.344075% | 2.178329→2.178008 |
| loot | 118 / 91 / 115 | 144536→143712 | −0.570100% | −1.163508% | 2.173170→2.172827 |
| soldier | 106 / 226 / 110 | 215664→214416 | −0.578678% | −0.691992% | 2.189966→2.189958 |
| redandblack | 30 / 136 / 132 | 158896→157648 | −0.785419% | −0.867484% | 2.175526→2.175020 |

平均G-PCC −0.704529%、OA −1.016765%。全4件でD1相当MSEはそれぞれ1.5005765 / 1.5003201 / 1.5013116 / 1.5011144のまま。両方向最大点距離は全て√3のまま。

単独の削除では被覆条件を満たさず、追加先を含むため成立したmask変更は7 / 5 / 11 / 26個あった。これは候補scoreだけでなく、最終voxel集合まで変更して検証した結果である。

ただし0.15%の勝者で、1個のmask変更が2操作以上に分解されたものはsoldierの1個だけだった。他3件は全て1操作maskだった。0.30%ではlongdress 1個、soldier 7個、他2件は0個。**複合maskを候補として許したことと、大量の複合maskが実際に有効だったことは区別する。** 今回の結果を強い3操作synergyの証明とはしない。

### 総操作率約0.30%

| 入力 | Add / Prune / Adjust | G-PCC C | 同じ点群のOA C |
|---|---:|---:|---:|
| longdress | 276 / 121 / 239 | −1.315229% | −2.285351% |
| loot | 320 / 134 / 195 | −0.819173% | −2.027325% |
| soldier | 212 / 513 / 159 | −0.704800% | −1.214575% |
| redandblack | 120 / 298 / 178 | −1.173094% | −1.431563% |

平均G-PCC −1.003074%、OA −1.739703%。こちらもG低下、D1相当MSE・最大点距離維持。ただしG-PCC向けに選んだplanであり、OAの最良結果ではない。前回のOA window法のlongdress −1.560%等を今回一律に更新したわけではない。

### best-of-Nの選択バイアス

上記はoffline codec oracleであり、学習済みNetworkの推論精度ではない。固定profile `combined` だけでも0.15%で全4件の品質を維持し、G-PCC −0.761 / −0.565 / −0.579 / −0.785%だった。ただしprofile自体は研究中に検討したものである。

事後的な内部leave-one-sequence-out確認では、他3系列でprofileを選び、残り1系列で適用した。平均−0.527858%。soldierでは選ばれたprofileがG条件を外れ、no-opへ戻した。**4系列の内部確認であり、独立した大規模汎化検証やNetwork学習成功とは扱わない。**

## 5. 共同化の効果を分解した結果

### 5.1 幾何品質では追加先と削除部分を切り離せない

longdressの共同planについて、削除部分だけを適用して追加先を全て取り除くと、最大GT→rec距離が√3→√6、D1相当MSEが1.5005765→1.5006131になった。共同planなら元の値を維持する。lootでも最大距離は削除部分だけだと√5、共同planでは√3。

これはAddまたはAdjust移動先による被覆の補完を確認したもの。単に「点が減ればよい」「移動元だけで幾何コストを決めればよい」では不十分である。

### 5.2 ただしG-PCCの3操作間rate相互作用は今回大きくなかった

共同planを先に生成した後、Addだけ / Pruneだけ / Adjustだけも符号化する診断を行った。共同削減bitsから、3単独削減bitsの和を引いた差は、longdress / loot / soldier / redandblackで **0 / 56 / 64 / 8 bits**。

非加法性は存在するが、今回の操作率・配置では小さい。「3操作を同時にしなかったことだけが−50%未達の最大原因」とは言えない。ablationは診断用であり、提案を3操作独立で最適化した結果ではない。

### 5.3 OAでは編集されていない周囲のcontextがgainを打ち消す

実算術coderと同じ16-bit整数CDFから、gainを次のように分解した。

`全gain = 元contextで編集symbolだけ置換したgain + 編集後contextへ再計算したgain`

| 入力 | 編集symbol自身のgain | context変化のgain | 合計CDF gain | 実stream削減 |
|---|---:|---:|---:|---:|
| longdress | +3123.867 bits | −587.846 bits | +2536.021 bits | 2536 bits |
| loot | +3083.148 bits | −967.066 bits | +2116.082 bits | 2112 bits |

contextによる損失のほぼ全ては未編集row側だった。整数CDFでの符号長が変わった未編集rowは42,628 / 43,458。差はwindow内の文脈伝播を含み、全rowが大幅に悪化したという意味ではない。CDF合計と実streamには終端・byte丸めの差がある。

longdressでは、編集前symbolの元context確率中央値0.0001853に対し、編集後symbolの元context確率中央値0.28257、再計算後0.28377だった。候補自身は高確率でも、**周囲を含めたtotal rateが最終評価対象**である。

なお、longdressの葉maskの単純な周辺分布エントロピーは310,659 bits、G-PCCの全streamは150,240 bits。これらは同じ符号長ではない。経験分布エントロピーをnative codecの残り削減可能bitsと読むことはできない。

## 6. 他のアプローチで何が起きたか

### 点数削減・近傍一致

longdress 0.15%で単純な点数削減profileはGを下げてもG-PCC **+0.580405%**。近傍mask一致profileはG-PCC **+1.746539%**でGも悪化。密度・規則性・点数だけでは圧縮価値を判断できない。

### 親ノード削除・隣接親への共同再配置

12,000親/frameを調査し、元親を全削除して周囲の保持voxel・追加先で被覆する共同patchを生成した。longdress 3,050候補、redandblack 2,948候補。そのうち追加先による被覆救済が必要な候補は582 / 551。

longdressで303葉親を消したnode優先planは、品質を維持してOA −0.309519%だがG-PCC **+0.085197%**。救済patchを優先すると、134 / 129救済patchを実行できたがG-PCC **+0.623003 / +0.584030%**。OAも+0.008480 / −0.038460%に留まった。

したがって、親削除や相互補完そのものは実装・実行できるが、**ノード数減少をそのままrate減少とするscoreは不十分**だった。全8件のtopology planで品質条件を満たし、両codecで再符号化した。

### G-PCC Actualを使う共同block探索

longdressの最良planを初期状態に、混合操作を含む12 blockの削除・別共同planとの置換を2巡した。48試行、31種類のproposalを評価したが追加改善の採択は0。単一操作を順番に調整した実験ではない。**この有限近傍での停滞であり、大域最適の証明ではない。**

### 幾何を保つ境界anchor・符号化原点の変更

境界の追加voxelによりbbox正規化後のgrid配置を全体的に変える経路も調査した。longdressではx軸だけ距離条件を満たすanchorがあった。共同planの1操作を置き換え、総318操作を維持するとG-PCC −0.921193%、追加の原点情報に仮に1 byte課金した推定net値は−0.915868%。G・D1・最大距離は基準以下だった。

しかし同じ点群はOA **+0.432478%**。全体grid変化もcodec固有である。原点情報を新しいwrapperへ実際にserializeした実験ではないため、推定overhead込みの値を完成codec性能とはしない。

## 7. 幾何lossと−50%目標についての重要な制約

今回の整数入力・固定qs=2では、baselineの丸めvoxelは各raw点にとって既にgrid上の最近傍である。別の整数grid点を追加しても、それより近くはできない。

`raw→rec MSE` のbaselineは1.5005765 / 1.5003201 / 1.5013116 / 1.5011144。これはこの固定gridでの下限である。D1相当MSEをbaseline以下に保つには、全raw点が少なくとも1個の同距離最近傍voxelで被覆され続ける必要がある。

したがって、今回の厳しいD1条件でGを下げる余地は主に逆方向 `rec→raw` にある。逆方向MSEは0.6777528 / 0.6728498 / 0.6886544 / 0.6744119。未占有grid点のrawへの距離二乗はこの整数条件では少なくとも1なので、**Add単独は平均逆方向誤差を上げる**。削除・移動と一体で品質を評価する理由がここにもある。これは固定grid・今回のbaselineについての議論で、可変量子化や任意連続座標の一般論ではない。

OAについては、実coderの整数CDFが各symbolに最低1/65536の確率を与えるため、元contextを固定した1 symbolの符号長は最大16 bits。longdressの318操作で変更する葉rowは最大318なので、**編集symbol自身だけの削減上限は5088 bits、baseline188680 bitsの約2.70%**である。

−50%には94,340 bits削減が必要。固定topology・固定原点のこの経路では、残り約89,252 bits以上をcontext全体の改善等で得る必要があるが、実測context効果は逆に約588 bitsの悪化だった。この条件付き評価は、微小な葉編集だけで−50%を期待する難しさを示す。親削除・原点変更や他codecを含めた万能な不可能性定理ではない。

論文との目標も揃える必要がある。Maらの前処理はglobal scaling・学習voxelization・pruningを含むRD最適化で、報告38.84%は平均BD-rateである。固定qs・0.15%操作での単一rate差と同じ指標ではない。[原論文](https://arxiv.org/html/2508.01633v1)

Zhengらの大きな改善は葉bitの一部を送信せずdecoder側で予測する方式を含む。既存codecを変えない微小前処理へ、その改善率をそのまま移せるわけではない。[原論文](https://arxiv.org/html/2603.28095v1)

## 8. 次に優先する研究設計

最優先は、**codec固有の「共同編集後の総Δbits」を学ぶ評価器と、被覆制約付き共同occupancy提案の組合せ**。単純な出現頻度priorや孤立候補gainだけではなく、周辺contextへの影響まで教師にする。

1. 状態は編集後の占有集合。最終mask / 小領域patchを共同提案し、Add / Prune / Adjustは実行計画への分解として扱う。3操作別の固定quotaや、単独Prune可否による先行除外を避ける。
2. G-PCCではActual joint-plan差分、OAでは少なくとも編集後window全体の整数CDF差分で教師labelを作る。元gridの近傍だけでなく、祖先変化・原点変化も区別する。
3. Amountは総予算の上限だけを設け、no-opと少量planも選択肢に残す。特定割合への強制減衰・強制増加はしない。
4. 先に独立した多数系列・複数RD点で実現可能性を評価する。今回の88 plan・4系列だけで新評価器の汎化や−50%を保証しない。原点変更、可変量子化、decoder変更は別実験枠で操作数・品質・side情報を明記する。

今回の確認範囲では、「探索noiseを増やすだけ」「3操作同時化だけ」「点数やノード数を減らすだけ」のいずれも−50%を裏付けなかった。**目標未達を解決済みとは報告しない。**

## 9. 成果物・検証・再現

- [集計JSON](/home/maejima/MasterEx/myNet/experiments/results/20261017_octree_joint_research/summary.json): 全主結果、操作family相互作用、context分解、各追加実験、内部CV。
- [SHA256 manifest](/home/maejima/MasterEx/myNet/experiments/results/20261017_octree_joint_research/manifest.json): 40ファイル、一致確認済み。
- 主実験: `experiments/octree_joint_mask_research.py`、raw結果 `experiments/results/20261017_octree_joint_masks_v2/`。
- 追加: `octree_joint_context_audit.py` / `octree_joint_topology_research.py` / `octree_joint_actual_search.py` / `octree_joint_phase_research.py`。
- 再集計: `experiments/summarize_octree_joint_research.py`。既存成果物を上書きしないため、出力先が存在すると停止する。

実行環境は `/home/maejima/miniconda3/envs/myOA/bin/python -B`、`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`。主実験は4フレームを `--frames`、`--budgets .0015 .003` へ指定。追加実験の対象・試行数は各節とraw JSONを参照。

既存8件 + codec関連6件 + 新規8件、合計22テスト合格。全255×255のmask変換の操作数、非局所移動をAdjust1個と数えないこと、共同被覆、親を跨ぐ救済、重複排除、予算・bbox、teacher一致・因果mask・整数CDFを検証した。

候補生成だけでも約10～15秒/frame、OA全体のActual評価はおおむね15～23秒/回、G-PCC encodeは約0.8～1.2秒/回（decode付きは別）。今回の探索をそのまま通常訓練へ入れるのは重い。offline教師生成・評価器への蒸留を想定した研究用コードであり、既存訓練への高速な差替えを完成させたとは主張しない。

G-PCC採用planはnative decoderで整数座標一致と再encode stream hash一致を確認。OAは実算術encode、encoder計算済み確率を使うsymbol decode、Octree座標復元を確認。ただし独立した自律OA decoder全体の検証ではない。既存third-party PLY readerのResourceWarning等は残るが、テスト失敗ではない。

最初のpilot `20261017_octree_joint_masks/` は「最近傍割当が0の葉も存在する」というケースで停止したため成果に数えない。空割当を正しく扱う修正とテストを追加し、`_v2` で4フレーム全て完走した。訓練コードの修正ではない。
