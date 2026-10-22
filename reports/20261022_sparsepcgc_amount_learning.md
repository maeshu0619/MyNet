# SparsePCGC Amount探索・学習経路の修正と実測

実験日2026-10-22、再開後の確認・追記2026-10-23。開始時HEAD: `59e8a6429ab00d0fdba33bf9e8e555ffb992f76b`。

## 結論と適用状態

**Amountの探索幅とActual RD教師からの学習経路は改善した。ただし、短時間の固定Validationでは圧縮率の非劣化を確認できなかった。このため新方式は明示的opt-inとし、通常起動時は従来動作を維持する。最終圧縮性能まで解決済みとはしない。**

新方式を使用する場合は、通常のSparsePCGC訓練コマンドへ次を追加する。

```bash
--sparsepcgc_amount_exploration true
```

`ana_den6_online`かつ操作量学習が有効なSparsePCGCルートに限定する。他codecには適用しない。実行中processの再設定・再起動は行っていない。

## 1. 原因の再確認

1. 既存の総量はno-op＋5つのpositive bin。今回の実験設定では0.15 / 0.20 / 0.25 / 0.275 / 0.30%。binの外側は各操作AmountのFine残差に依存する。
2. Fineの元のlog探索標準偏差は0.08。学習された残差のforwardはWhereとの整合性などのconfidenceに弱められる。初期confidenceも小さく、Amountの変更が実行量へ反映されにくい。
3. `Amount`比較groupでは従来、coarse binだけを変更し、Fineをanchorへ固定していた。実際のStep 6で3つのFine Amount headのActual勾配は全て0だった。Fine groupでは従来から非ゼロの場合もあるため、「常にAmountの勾配が0」という意味ではない。
4. 学習済み旧checkpointにはraw Amount出力が約−45～−55のものがあった。既存のbounded ratio経路はその領域で実際のforward感度が小さい。ゼロ勾配へ倍率を掛けても解決にならない。
5. coarse binの`argmax(logits + scale × Gumbel)`に対して、教師用log確率が`log_softmax(logits)`だった。scaleが1でない場合、実際のサンプル分布は`softmax(logits / scale)`である。

## 2. 新方式の設計

操作kの要求割合の概略は次式。丸め、Pool容量、重複・競合排除、安全上限は後段で従来どおり適用する。

```text
r_k = coarse_bin × Gate_share_k × exp(legacy_fine_mean_k + δ_k + σ_k(t) × ε_k)
δ_k = span × softsign((raw_k − frozen_reference_raw_k) / span)
span = log(2), ε_k ~ Normal(0,1)
```

- 基準は同じ入力特徴へ適用する**固定checkpoint headの重み・bias**。現在の出力を毎Step detachして基準にする方式ではない。
- 初期化／旧checkpoint読み込み直後は全入力で`δ=0`。旧forwardを維持しながら、raw=−55でも新残差のrawに対する真の微分は1。学習後はforward自体が変化する。STEによる勾配のみの増幅ではない。
- 追加残差は概ね0.5～2倍の学習範囲。総量を特定割合へ強制増減するscheduleは追加していない。coarseのclass数・既存binの意味も維持する。
- Fineの初期log標準偏差を0.08→0.30へ拡大。単一操作のnoise倍率の約95%範囲は、他の探索倍率が1なら約0.56～1.80倍。これは正規分布の区間でありhard範囲ではない。
- 既存の探索schedule・adaptive倍率を使用し、既定の前半25%探索窓の終了までにnoiseは0になる。評価時も0。新たな操作率の強制減衰はない。
- `Amount`比較ではcoarseとAdd / Prune / Adjustの3つのFineを**同時に**探索。比較相手も同一frame・anchorの同じ条件で再評価する。Fine groupでは従来どおりcoarseを固定して3つのFineを探索する。
- WhereAll比較では量を固定して位置を比較する仕組みを維持。全要因を同時に乱してWhereへAmountの効果を誤配分する変更はしない。WhereのActor／Proposal／既存Actual学習経路も維持する。
- binの方策log確率を実サンプル温度に整合させる。no-opで実行されなかったFineへActual creditを与えない。
- Actual codecは微分不能。圧縮／幾何の実測値を含む既存RD比較教師から方策勾配を流す。圧縮bitstreamから直接autogradする方式ではない。

新しい学習可能パラメータは追加していない。3 headの基準weight / biasとversionをbufferとして追加（実測構成で約692 bytes）。旧checkpointはメモリ上だけで移行し、新checkpointでは固定基準も保存・復元する。

## 3. 実験条件と比較の整合性

- 独立したUVG / SparsePCGC dense_lossy、m=8、seed 21と22。モデルはGPU、Actual teacherは既存CPU codecで実行。既存訓練processには接触しない。
- seed 21は修正前後32 Step、seed 22は24 Stepを比較。各状態を同じ固定4 holdoutフレームでActual評価。最終382 Episodeの代わりではない。
- seed 22の修正前は既存の独立64 Step実験の最初24 Stepとsnapshot 24を再利用。111 sourceのSHA256が今回の修正前snapshotと一致することを確認。
- 両seedとも共通の初期state tensorがbyte一致、入力frame・source voxel hashの順序が一致、初期Actual Validationが完全一致。追加された7 bufferだけは旧モデルに存在しない。既存NaN診断sentinelもbyte比較した。
- 圧縮重み、幾何重み、optimizer、clip上限、安全上限、Codecの設定は変更していない。
- 最初の試作でpre-calibration ratioを誤って利用し、初期Validationが変化した。これを不採用とし、最終ratioを保持するV2へ修正。回帰テストと再実験を実施した。`after_s21`は不採用試作、最終比較は`after_v2_s21/s22`のみ。

## 4. 実行された操作量

分母は編集前のvoxel数。全訓練Stepの実際の選択数を集計。単なる要求量やnoise flagではない。

| seed / Step | 総操作率の正の最小～最大: 修正前→新方式 | 総操作率std: 前→後 | no-op: 前→後 |
|---|---|---|---|
| 21 / 32 | 0.149564～0.278892% → 0.115855～0.296046% | 0.048344→0.062759 percentage points | 1→2 |
| 22 / 24 | 0.198978～0.268027% → 0.167692～0.315837% | 0.050413→0.081778 percentage points | 1→3 |

seed 21の各操作の実行割合:

| 操作 | 修正前の正の最小～最大 [%] | 新方式の正の最小～最大 [%] |
|---|---:|---:|
| Add | 0.023413～0.192944 | 0.021927～0.224894 |
| Prune | 0.046685～0.222859 | 0.012867～0.213320 |
| Adjust | 0.003469～0.097828 | 0.000559～0.019632 |

総量の探索幅は増えたが、**各操作の観測幅が必ず増えるとは限らない**。Adjustはこのseedでは狭くなった。Gate、no-op、学習の変化も含む実行分布であり、全ての幅をnoise単体の効果と解釈しない。合計操作率・変更voxel率の既存0.99%上限への違反は0。

## 5. 勾配と実更新

seed 21のAmount比較Step 6。Actual列は圧縮・幾何を含むRD教師からの勾配。clip後は全損失を合成・復元・balanceした実際のgradientであり、Actual単独の分離値ではない。

| module | Actual L2 前→後 | 新方式Actual RMS | 新方式clip後L2 | 新方式実更新L2 |
|---|---:|---:|---:|---:|
| Add Amount | 0→110.2742 | 14.6062 | 0.582562 | 8.58717e−6 |
| Prune Amount | 0→35.8619 | 4.7500 | 0.189451 | 7.63187e−6 |
| Adjust Amount | 0→149.6168 | 19.8172 | 0.790399 | 4.48325e−5 |
| Coarse Amount | 182.3258→94.4126 | 1.48575 | 1.000000 | 1.15547e−4 |

3 Fine headの共通clip normは1、coarseは別のclip groupという既存設計を維持した。後段global clipは修正前後ともprobe 8回中0回。巨大なraw L2をそのままoptimizerへ渡しているわけではない。実更新率`||Δθ||/||θ||`は新方式で0.07528 / 0.05346 / 0.44097 / 2.3600e−5。初期Fine重みのノルムが非常に小さいため、割合だけで更新過大／過小を判定しない。

Where比較Step 2の新方式Actual L2はAdd 0.0542525、Prune 0.0715050、Adjust 0.0332666。実更新L2は0.00495815 / 0.00432239 / 0.00613563で、修正前とほぼ同じ。Where経路を壊してAmountだけを学習させた結果ではない。

Fine AmountのActual非ゼロprobe数は同じ8 probeで3→5。全Stepの勾配を非ゼロにすることは狙わない。比較教師のないanchor、Where比較、no-opではFineのActual creditが0になる場合がある。各損失・restore前後・balance前後・clip前後・更新の全値はCSVへ保存した。

## 6. 固定Validation: 改善と未解決を区別

Actual Cは負が圧縮改善、Gは小さい方が良い。noiseなし、同じ4フレームの平均。

| seed / 評価Step | 修正前Actual C [%] | 新方式Actual C [%] | 修正前G | 新方式G |
|---|---:|---:|---:|---:|
| 21 / 0 | −2.140586 | −2.140586 | 0.024636642 | 0.024636642 |
| 21 / 24 | −2.140376 | −2.100223 | 0.024567200 | 0.024282391 |
| 21 / 32 | −2.134260 | −2.097225 | 0.024599288 | 0.024233994 |
| 22 / 0 | −2.152769 | −2.152769 | 0.024601584 | 0.024601584 |
| 22 / 24 | −2.106734 | −2.009180 | 0.024637376 | 0.024172716 |

両seedともGeometryは改善したが、圧縮改善率は低下した。探索・学習が量へ効いていることは確認できる一方、**圧縮非劣化や最終性能向上は未確認**。単に「学習が安定した」「最終圧縮率を維持できた」とは結論しない。

旧学習済みcheckpointの初期forwardも4フレームで再符号化し、C=−1.9051239039501278、G=0.024276533164083958が旧コードと完全一致。これは初期互換性の証拠であって、再訓練後の性能保証ではない。

既存Actual方策教師は`3*C + 150*G + 2*変更voxel割合`。一方、今回の既存固定Validationの`fixed_objective`は`C + cp_lambda_geom * max(G-cp_tau_geom,0)`で、今回cp_lambda_geom=150。両者を同じJと誤記・混同しない。重みも式も本作業では変更していない。連続Amountの学習自由度が増えると圧縮／幾何の選好や信用割当の不完全さも顕在化する。どれが今回のValidation低下を支配したかは、head rollbackや長期比較なしには断定しない。

### 再開後の追加分析: 教師の分解と更新方向

保存済みの同一frame / anchorに限って、教師差が`3*ΔC + 150*ΔG + 2*Δchanged_ratio`と一致するかを再計算した。最大の丸め差は7e−7未満。今回追加の訓練や既存成果物の上書きは行っていない。

| 実験 | 有効Fine / Amount比較 | variantのRD改善 / 悪化 | CとRDの優劣が逆 | Actual勾配と逆向きの個別head更新 |
|---|---:|---:|---:|---:|
| seed 21 修正前 | 7 | 3 / 4 | 0 | 2 / 9 |
| seed 21 新方式 | 7 | 3 / 4 | 0 | 4 / 15 |
| seed 22 新方式 | 5 | 2 / 3 | 1 | 3 / 12 |

最後の列は非ゼロActual勾配が計測されたAmount headの実更新だけを対象とする。Adam履歴により、瞬間の個別head勾配に逆らう更新が残る。一方、3 head合計の一次近似では、対象となる全StepでActual**方策損失**の降下方向だった。これはcodec bitsが下がることの証明ではない。

したがって、今回の範囲では「教師の圧縮／幾何の足し算が壊れている」「更新全体がActual方策損失に逆行している」という説明は支持されなかった。探索した良い／悪い量の経験は増やせるが、それだけでnoiseなしValidationの圧縮性能へ汎化するとは限らない。Geometry優先の比較は1件あるものの、seed 21には優劣逆転がないため、**Geometry重みだけを今回の圧縮低下の主因とは断定できない**。

再計算は`experiments/amount_learning/analyze_tradeoff.py`、[集計](/home/maejima/MasterEx/myNet/experiments/results/20261022_amount_learning/tradeoff_20261023/summary.json)、[各比較の分解](/home/maejima/MasterEx/myNet/experiments/results/20261022_amount_learning/tradeoff_20261023/pairs.csv)、[更新方向](/home/maejima/MasterEx/myNet/experiments/results/20261022_amount_learning/tradeoff_20261023/update_directions.csv)へ保存した。

## 7. 変更ファイル・再現

- `models/utils/training/sparsepcgc_amount_policy.py`: 固定基準、旧checkpoint移行、真の連続残差、sampling温度に整合したbin log確率。
- `models/modules/structure_actuator.py`: SparsePCGC分岐への接続、3 Fine＋coarse共同Amount探索・credit。既存の後段ratio補正、safe Pool、実行制約は保持。
- `models/utils/config/args.py`: opt-inと探索σ・残差span・共同探索の設定。標準OFF。
- `tests/test_sparsepcgc_amount_policy.py`: 互換load、他codec排除、勾配、実voxel集合変化、後段補正保持、探索減衰、評価determinism。
- `experiments/amount_learning/`: 開始時snapshotと独立集計。
- `experiments/sparsepcgc_direction/run.py`: 独立実験のopt-in、選択snapshotだけを評価するオプション。

訓練・勾配計測の再現例（`--out`は未作成の新規ディレクトリを指定）:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/maejima/miniconda3/envs/myOA/bin/python -B \
  experiments/sparsepcgc_direction/run.py \
  --out /data/maejima/log/independent_experiments/amount_new_run \
  --amount-exploration on --seed 21 --steps 32 --validation 4 \
  --probe-steps 2,4,6,8,12,16,24,32 \
  --seed-cache experiments/results/20261022_actual_direction/cold_current_s21/cache
```

主比較集計は[summary.json](/home/maejima/MasterEx/myNet/experiments/results/20261022_amount_learning/summary_v3/summary.json)、[勾配CSV](/home/maejima/MasterEx/myNet/experiments/results/20261022_amount_learning/summary_v3/gradients.csv)、[実更新CSV](/home/maejima/MasterEx/myNet/experiments/results/20261022_amount_learning/summary_v3/updates.csv)、[操作割合CSV](/home/maejima/MasterEx/myNet/experiments/results/20261022_amount_learning/summary_v3/decisions.csv)、[Validation CSV](/home/maejima/MasterEx/myNet/experiments/results/20261022_amount_learning/summary_v3/validation.csv)。大きな成果物は`/data`へ保存し、repo内のresultsからsymlinkしている。

### 最終コードの再確認

- 最終コードで標準OFFと明示ONをそれぞれ8 Step訓練し、Actual Validationまで完走。8 Step全ての入力hash、編集後voxel hash、操作数が、それぞれ修正前／新方式V2の対応する先頭8 Stepと一致した。
- 2026-10-23に関連6 test moduleを再実行し、**107テスト合格**。テスト件数を決め打ちして成功扱いせず、実際のsubprocess終了コードと出力を保存するよう最終検証scriptも修正した。
- 既存111 sourceのうち変更はactuatorとargsの2つだけ、他109 sourceは開始時SHA256と一致。各独立訓練run中のsource/checkpoint変更検出も0件。
- 証跡は[最終確認JSON](/home/maejima/MasterEx/myNet/experiments/results/20261022_amount_learning/final_checks_resumed_20261023/checks.json)、[実テスト出力](/home/maejima/MasterEx/myNet/experiments/results/20261022_amount_learning/final_checks_resumed_20261023/tests.txt)、[成果物SHA256 manifest](/home/maejima/MasterEx/myNet/experiments/results/20261022_amount_learning/final_checks_resumed_20261023/manifest.json)。

## 8. 保持した制約と今後

- `train.py`、`network.py`、loss、Codecソース、既存checkpoint・既存ログは変更していない。G-PCC / OctAttentionの実訓練比較は行っていないが、分岐の不適用と既存関連テストを確認した。
- safe Poolに存在しない位置の新規提案や、候補容量を超える操作は許可していない。安全上限や幾何制約の撤去ではない。
- 探索窓終了後のActual比較教師不足、Criticの信頼性・候補間credit、Adam履歴による瞬間勾配と更新方向の不一致は、今回解決したとはしない。
- 次は独立環境でFine Amount / coarse / Whereのrollbackを行い、同一初期重み・複数seed・より長い学習でActual CとGのPareto比較をする。圧縮非劣化を確認するまで新方式を標準ONにはしない。特定Voxel割合やloss値を人工的に補正しない。
