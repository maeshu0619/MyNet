# G-PCCの圧縮改善が出ない原因と、微小voxel編集の独立実験

実験日: 2026-10-16～17。既存ソースの基準commit: `a805ba082e2cb6916cf78d30ddc3e05170d692c6`。

## 結論

**操作量を増やさなくても、圧縮損失を負値にできた。ただし今回の微小操作で−50%は達成していない。**

直接確認できた問題は、(1)候補priorと実G-PCC符号量の不一致、(2)改善する編集をほとんど含まないpool、(3)編集後のbbox原点の変化を局所Octree差分が扱わない点である。単にノイズ・損失重み・操作量を増やせば解決するという証拠は得られていない。

今回追加・編集したのは `experiments/` 以下の独立実験と本レポートだけ。`train.py`、既存モデル、args、codec設定、checkpoint、実行中の訓練processは変更・停止・再起動していない。実験はCPUで行った。

## 1. 論文の−40～−50%と現在の指標は同じでない

参照論文は平均D1 BD-rate −38.84%、D2 BD-rate −27.90%を報告する。複数量子化・画質条件にまたがるRD曲線の比較であり、同じqsでbitsを半減するという意味ではない。手法にはglobal scalingと学習voxelizationも含まれ、今回の0.1%前後の編集予算と同じ制約ではない。[論文本文](https://arxiv.org/html/2508.01633v1)

今回の定義は現行teacherと同じ:

`C[%] = 100 × (編集後のG-PCC全stream bits − 未編集bits) / 未編集bits`

負値が改善。qs=2、geometry-only、同じtmc3・encoder.cfgで編集前後を比較する。bppの分母を編集後点数に変えた見かけの改善ではない。

この条件で−50%を目指すこと自体は研究目標として設定できる。しかし、論文のBD-rateだけから「微小編集で同じqsのbitsが半減するはず」とは判断できない。今回BD-rateやD2の再現実験はしていない。

## 2. 実験条件と検証

- 入力: longdress 1051/1052、loot 1000、soldier 0536、redandblack 1450/1451。
- encoder: 同梱G-PCC `release-v23.0-rc2`。通常実験では設定を変更せず、planar/QTBT等の変更は別の対照実験に隔離した。
- 操作割合の分母: **未編集・qs=2量子化後のunique voxel数**。raw点数比ではない。0.0015 fraction = **0.15%**。Adjustは1点移動を1操作、占有cellの差分は2として別記録した。
- 幾何損失G: 元入力座標での `mean NN²(GT→再構成) + mean NN²(再構成→GT)`。両方向の最大点距離、D1用の最大方向MSEも記録した。
- 新規実験はNetworkを学習・評価したものではない。明示的な編集規則やcodec探索の診断であり、best-of-NをNetworkの性能と呼ばない。
- 点編集trialは372行（baseline・再測定・対照を含む）。別途、量子化12条件、codec設定12条件。合計432回の実符号化を実施した（再確認を含み、小点群の単体テストは除く）。
- baselineと各runの最良planを実decodeし、整数voxel集合の一致を確認。単体テストでは現行teacherとのbits一致も確認。全点平行移動・点順序変更でbitsが変わらないことも確認した。

物理座標は元のgrid originとqsを使って復元して幾何評価した。bitsは現行teacherと同じ正規化後のstreamであり、アプリ側の外部offset等の伝達コストまで測ったものではない。

OctAttention/SparsePCGCの追加実験は行っておらず、G-PCCの結果をそれらへ一般化しない。

## 3. 最新訓練ログの確認

`20261016/MyNetwork_train/G-PCC/174254.txt` の46,219,413 bytes時点を取得した。snapshotのhashと集計は `aggregate.json` に保存した。

| 観測Actual Step | 負値 | 0 | 正値 | 平均C |
|---:|---:|---:|---:|---:|
| 2,200 | 0 | 459 | 1,741 | +1.509492% |

再開時のEpisode CSVでは54 Episode完了。最初の10 Episode平均C=+1.821254%、直近10 Episode平均=+1.425317%。悪化を小さくしているが、編集によるbit削減には至っていない。

ログ上のbackendは `gpcc_surrogate`、Actual encoderもG-PCC、qs=2、SparsePCGC teacher/cache=False。少なくとも「G-PCC指定なのにSparsePCGCで圧縮している」という説明は支持されない。

開始時にはSurrogate checkpoint未検出、`pretrain_steps=0`、Surrogate backward有効と記録されている。現行Surrogateはscalar予測MLPであり、論文の事前学習済みautoregressive occupancy surrogateとは異なる。**ただし、その後online学習しているため「現在も未学習」「勾配が流れていない」とは断定しない。** その因果寄与を分離する訓練ablationは今回実施していない。

## 4. 同じ点数を削除しても、選ぶ場所で符号が逆転した

現行候補生成＋joint selectorを、Actor残差0・探索0で呼び出してpriorの初期品質を調べた。学習済みcheckpointの再生ではない。

まず、**Prune単独の削除数を完全一致**させた比較:

| 入力 | 削除数 | 操作率 | 現行priorのPrune C | 品質＋bbox制約付きPrune C |
|---|---:|---:|---:|---:|
| longdress 1051 | 127 | 0.059938% | +0.154420% | −0.079872% |
| loot 1000 | 129 | 0.059673% | +0.154979% | −0.066419% |
| soldier 0536 | 176 | 0.059726% | +0.189183% | −0.070480% |
| longdress 1052 | 128 | 0.059668% | +0.147992% | −0.100423% |

これはAmountを増やした効果ではない。元入力への距離を確認し、疎な点を優先しながら代替近傍を残し、bboxを維持する編集規則へ変えた結果である。今回の4入力では、同じ削除数でもWhereの選び方に改善余地があった。

次は総操作率約0.15%での4フレーム平均:

| 編集規則 | 平均C |
|---|---:|
| ランダムPrune | +1.890325% |
| 密な箇所の兄弟occupancyを埋めるAdd | +0.720923% |
| singletonを隣の既存親へ移すAdjust | +1.603698% |
| 現行priorのAdd/Prune/Adjust joint | +0.763345% |
| 疎な箇所のPrune（品質制約なし） | −0.319004% |

この表は**総操作数だけ**を揃えており、各操作への配分は違う。jointは概ねAdd/Prune/Adjust=0.06/0.06/0.03%。純Pruneとjointの差をWhereだけの効果とは解釈しない。Whereを切り分けたのは上の同数Prune比較である。

### priorが良いと判定したplanが、Actualでは悪かった

総操作率約0.30%の現行joint plan:

| 入力 | proxyの予測削減bits | Actualの削減bits | Actual C |
|---|---:|---:|---:|
| longdress 1051 | +3,268.36 | −2,096 | +1.395101% |
| loot 1000 | +3,396.65 | −2,224 | +1.538717% |
| soldier 0536 | +4,361.38 | −3,320 | +1.539432% |
| longdress 1052 | +3,283.84 | −2,136 | +1.411205% |

予測gainの大きさだけでなく**符号も不一致**。現行の `GeometryOctreeRate` は経験的occupancy priorで、G-PCCの実CPM・算術符号化を再現していない。3操作の相互作用を扱うこと自体は必要だが、不正確なrate尺度を共同最適化しても実bitsは減らない。

## 5. 改善した候補の大部分がpoolに入っていない

品質制約付き0.15% Pruneで実際に使った削除点に対する、現行poolの包含数:

| 入力 | 改善planの削除点数 | 単点Pruneとして存在 | bundleの削除点を含めた包含数 |
|---|---:|---:|---:|
| longdress 1051 | 318 | 2（0.629%） | 10（3.145%） |
| loot 1000 | 324 | 1（0.309%） | 7（2.160%） |
| soldier 0536 | 442 | 2（0.452%） | 8（1.810%） |
| longdress 1052 | 322 | 3（0.932%） | 5（1.553%） |

bundleに含まれる点を単独で選べるとは限らないため、右列も完全再現可能率ではない。bboxの最小座標を外へ広げるAdd/Adjust候補は、これらのpoolでは0個だった。

現行生成器は全親を無制限に保持せず、親の間引きとproxy順位によるpool縮約を行う。[生成器](../models/utils/pointcloud/geometry_joint_proposals.py:44)

**poolに存在しない候補はActorやノイズでは選べない。** ただし、これは「現在のpoolには負値planが一切ない」という全探索による証明ではなく、今回見つけた改善planを大部分再現できないという実測である。

## 6. 1点で大きく変わる原因: 編集後の原点再設定

現行Actual encoderは毎回 `coords -= coords.min(axis=0)` を行う。[Actualの正規化](../models/utils/compression/gpcc_tmc3.py:98)

一方、priorの `patch()` は編集前の座標系上で、変更点と祖先のmaskを更新する。編集後bboxに合わせて残り全点を再配置しない。[局所差分](../models/utils/pointcloud/geometry_octree_rate.py:81)

最小座標を決める点を消すと、実codecでは他の全点のchild slotや親の組合せも変わり得る。局所的な数ノードの変更として評価することと一致しない。

redandblack 1450での対照実験（未編集160,048 bits）:

| 編集 | bits | C | 編集後bbox最小座標 |
|---|---:|---:|---|
| 品質制約付き301点Prune | 162,760 | +1.694492% | (0,0,1) |
| 上からz最小点を1点だけ戻す | 159,736 | −0.194942% | (0,0,0) |
| 上からx最大点を1点だけ戻す | 162,760 | +1.694492% | (0,0,1) |

z最小点の復元だけで3,024 bits変化した。変更点自身の局所影響も含むので「3,024 bitsすべてが原点だけのコスト」とは厳密分離できないが、コード経路と対照から、大域的な符号化位相の影響を無視できない。

同様にlongdress 1051では、境界外側へ近接する2点を追加するだけで150,240→148,624 bits、C=−1.075612%、操作率0.000944%となった。ただしGは2.178329→2.178389、再構成→GTの最大点距離は1.732→3.0。別フレームへの固定適用では悪化もあり、万能な品質維持手法ではない。

この効果は**現行teacherのbbox再正規化を含む結果**。任意のG-PCC入力正規化やOctAttentionでも同じ効果になるとは限らない。

## 7. bboxと幾何品質を維持した固定規則の結果

実装した実験規則は、元入力への距離を使い、削除点に対する代替近傍を残し、各軸のbbox端点を保護する。これは手設計の前処理で、Networkの学習成果ではない。圧縮後bitsを見て削除点を選ぶbest-of-Nでもない。

| 入力 | 削除数 | 約操作率 | 未編集bits→編集bits | C | G: 未編集→編集 |
|---|---:|---:|---:|---:|---:|
| longdress 1051 | 318 | 0.15% | 150240→149840 | −0.266241% | 2.178329→2.175661 |
| loot 1000 | 324 | 0.15% | 144536→144160 | −0.260143% | 2.173170→2.170149 |
| soldier 0536 | 442 | 0.15% | 215664→215272 | −0.181764% | 2.189966→2.187024 |
| longdress 1052 | 322 | 0.15% | 151360→150920 | −0.290698% | 2.177186→2.174436 |
| redandblack 1450 | 301 | 0.15% | 160048→159744 | −0.189943% | 2.172207→2.170099 |
| redandblack 1451 | 298 | 0.15% | 158896→158400 | −0.312154% | 2.175526→2.173629 |

平均C=**−0.250157%**。全6フレームで、両方向の最大点距離1.732051とD1用の最大方向MSEを維持し、Gは低下した。D2・法線品質・知覚品質の同一性までは検証していない。

redandblack 1450は、最初の未使用フレーム検証で前述の+1.694%になった失敗例であり、隠さず保存した。その結果からbbox保護を追加した。1451はその後の未使用フレーム確認だが、隣接フレーム1例であって大規模なblind評価ではない。

## 8. −50%を「達成したことにしない」ための対照

- longdress 1051の今回の微小操作で最も小さいCは**−1.586794%**。636 Prune＋2 Add、総操作率0.301104%。平均Gは低下するがGT→再構成の最大点距離が1.732→7.348に増えた。品質保持の代表結果には採用しない。
- qs=2→4では3フレームで約−65.5～−66.4%。しかし物理voxel集合の変更には少なくとも約80.7%相当の操作が必要で、Gも約2.18→6.75へ悪化した。今回の微小編集という要件を満たさない。
- 兄弟を1点へ大きく間引く対照では、約73.8%削除してC約−34.5～−36.8%。これも採用しない。
- codec設定だけの対照では、planar有効化は約+0.02～+0.03%、IDCM無効化は差0、QTBT無効化は約+12.8～+17.1%。今回試したスイッチでは−50%の未使用改善は見つからなかった。設定変更は実験process内のみで、元cfgには反映していない。

**微小編集で−50%が不可能と証明したわけではない。** 今回試した方法・データで確認できた範囲は小幅な改善である。−50%を正当化するには、同じ編集予算・画質で大きなgainが出る別の構造編集を、まずActualで発見する必要がある。

## 9. 次に訓練側へ反映するなら

最優先は、操作上限や損失係数の拡大ではなく、**Actualと整合したG-PCC向け候補価値・候補poolへの変更**。

1. bboxが変わる編集について、Actualとproxyが同じ原点・Octreeを評価するようにする。安易な原点固定で別の符号化問題へ変えず、方針を明示する。
2. 今回のような品質を保つ疎領域候補をpoolに含め、同じ操作数でActualのgain符号が合うか検証する。Add/Prune/Adjustの配分変更とWhereの改善を分離して測る。
3. empirical histogramのgainを真のbitsとして信頼せず、実codecの近傍・符号化順序・祖先変化を扱う局所価値を校正する。3操作共同評価は維持する。

今回、既存訓練への実装変更や長期訓練はしていない。勾配が0か、係数変更で改善するかの因果実験もしていないため、その点を「解決済み」とは報告しない。

## 再現用ファイル

- [独立voxel編集実験](../experiments/geometry_micro_edit_probe.py)
- [候補poolの包含率監査](../experiments/geometry_pool_audit.py)
- [量子化の対照](../experiments/geometry_quantization_control.py)、[codec設定の対照](../experiments/geometry_codec_control.py)
- [全trial CSV](../experiments/results/20261016_gpcc_micro_edits/all_trials.csv)、[集計・ログsnapshot](../experiments/results/20261016_gpcc_micro_edits/aggregate.json)
- `gpcc-micro-bboxguard-20261016/` 以下に品質・bbox制約付き結果のNPZを保存。物理座標は `(canonical_voxels + original_grid_origin) * qs` で復元できる。
- 単体テスト8件通過。既存tracked fileの `git diff --exit-code` は0。

固定規則の再実行例（既存訓練を起動しない）:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/maejima/miniconda3/envs/myOA/bin/python -B \
experiments/geometry_micro_edit_probe.py \
  --frames /data/maejima/data/ground/8i/longdress/longdress_vox10_1051.ply \
  --out /tmp/gpcc-micro-recheck-new-directory \
  --modes qualityguardbbox --budgets .0015
```

`--out` は新規ディレクトリに限る。操作数を揃えたPrune比較は `--modes matchedprune --budgets .0015`、元のprior比較は `--modes prior --budgets .0015`。
