# SparsePCGC Actual教師・勾配経路の項目別修正と実測

2026-10-22。開始時HEAD: `87b8219279dd614e39ddb1c79129d4ea6afb511e`。

## 結論

**同一frameのActual比較、Proposalの更新、Prune Whereの減衰、Amountのclip干渉、Adapterの初期ゼロ更新、学習済みAmountのFP32飽和、最近のActual教師取得の偏りを修正した。初期状態と学習済みcheckpointの両方で、対象headへActual教師から勾配が届き、Adamで更新されることを確認した。**

ただし「圧縮効率と幾何品質の両方が改善した」「長期収束まで解決した」とは結論付けない。初期24 Stepの固定ValidationではCが0.01818ポイント悪化し、Gは約1.04e-5低下した。学習済みcheckpointからの24 StepではCを維持し、Gが約1.29e-5低下した。いずれも2 frame・1 seedの短時間診断であり、有意な性能改善の証拠ではない。

Candidate Criticが得るのは**同一anchorの候補集合の対比教師**である。複合planのActual scalarから、各候補の独立した因果Δbitsを一意に復元することはできない。この点は未解決の識別上の制約であり、各候補をActual encode済みとは報告しない。

## 1. 保存・比較条件

- 開始時のtracked未コミット変更は`args.py`の8i→UVG選択2行。保持した。既存のOct20監査コード・結果・レポートも保持した。
- [開始時Git状態](../experiments/results/20261022_credit_fixes/00_baseline_source/status.txt)、[開始時差分](../experiments/results/20261022_credit_fixes/00_baseline_source/worktree.diff)を保存した。開始時snapshotには今回の診断wrapper自体は既にuntrackedとして載っているが、訓練ソース修正前である。
- 旧版対照は保存した5ファイルをread-only import loaderで実行した。実行中に作業ツリーが変わった対照のintegrity欄は空ではないが、**実行に使った凍結snapshotのSHAは全件一致**している。最終版2 runは`changed=[]`。
- NetworkはCUDA、SparsePCGC Actual teacherはCPUに固定した。GT Actual cacheは各runで新規作成し、編集後とGTのbackendを統一。旧来のGPU教師とのbitwise同一性までは主張しない。
- 静的候補・固定構造・固定特徴cacheのみ独立出力先へコピー。既存学習済み出力や編集後Actual結果を教師として流用していない。
- 診断process内だけCPU teacherの非同期overlapを無効化した。初期化ready応答との競合で失敗した試行があったためであり、圧縮設定や本体codec実装は変更していない。
- seed=21、UVG、AMP無効、382 Episodeの元scheduleを保持。最終比較は4条件×24 main optimizer更新。新方式は連続2-viewなので、同じplan評価/更新Step数ではunique frame数が異なる。GT cache、no-opの有無を含めたActual encode呼出し回数まで厳密に同じ予算とは呼ばない。この差を隠して完全同一入力列とは呼ばない。
- 学習済み状態は既存`043722_SparsePCGC/381.pth`とruntime sidecarのread-only warm-start。503 parameter keys、missing/unexpected/shape mismatchはいずれも0。**既存のmore-trainingはAdam stateを復元しないため、完全なoptimizer resumeではない。**
- 固定ValidationはBlueBackpackの0100/0101、signature `a92d8679cfa4d386382ec548312e4344aac449272eac0f0087b358ffdcba7744`。現在のtrain分割は各系列先頭100 frame。旧043722ログにもこの集合について`train_count=100, overlap=0`が記録されている。ただし同系列内分割であり、未知系列への汎化実験ではない。

既存checkpoint・既存ログ・codecソース・他人のprocessは変更していない。実験用checkpoint/log/cacheだけ新規出力先へ保存した。全実験終了後、診断用CUDA processは終了している。

## 2. 現在の勾配経路

`Actual C,G → detachしたRD教師 → 同一frame/anchor/group比較・replay・Critic回帰 → policy-only勾配保存 → 全loss backward → 決定headのpolicy-only復元 → Where/Fine Amount/coarse Amount/Gate別clip → global clip → Adam`

Actual codec自体は微分しない。主Actual RD教師は既存どおり

`J = 3*C + 150*max(G - cp_tau_geom, 0) + edit_tiebreak_weight*changed_ratio`

である。`w_com=3`、`cp_lambda_geom=150`は変更していない。以下の「Actual勾配」は**圧縮単独ではなく、このActual RD・Actual Critic教師のsource**を指す。

hard sort/int planと整合しない`L_downstream`や通常GeometryのSTEを無条件に決定headへ混ぜる修正は行わなかった。更新前の復元は維持し、代わりにActualの比較教師を成立させた。したがって、そのpathwise寄与が復元後に0になること自体は残る。Adapterなど非決定headのpathwise学習は残る。

## 3. 指定順に行った確認・修正

### 3.1 Actual比較教師

修正前は初期frame/groupが毎Step変わり、frame baseline、同一anchorの比較plan、replayが揃わなかった。最初の4 Stepはpolicy coreが0。初期6 Stepの厳密pairは0/6、24 Step対照でも0/24だった。

訓練window生成後だけ、各frameをanchor→variantの順で2回提示するDataset wrapperを追加した。base datasetの`files/all_files`とValidation集合は書き換えない。pending比較の既存優先処理により、他要因を固定した同一frame/anchor/group比較を完了する。

項目1だけの6 Step確認でStep 2/4/6に比較が成立。RD差はそれぞれ1.04263 / 0.0943441 / 2.51225。最終版は初期・学習済みとも12/24 pair。残り12 Stepは新しいanchorの測定なので、教師未成立を偽の非ゼロadvantageで埋めない。

`--sparsepcgc_matched_frame_repeats 2`が新しい既定、1で従来順へ戻せる。他codec・test経路は1回のまま。unique frameあたりの実行plan数/更新回数は約2倍になる。1 Stepあたり完成planは1個であり、多数候補を追加Actual encodeする変更ではない。探索Step見積りも2-viewに合わせ、Episodeに対する既存の探索減衰を維持した。

### 3.2 Candidate Critic

専用Actual replayが成立しない間の微小勾配は、主に候補Q統計→Plan feature→Plan Criticの間接経路だった。教師条件を満たすと既存の候補集合比較lossは既に接続されていた。新しい候補ごとの偽labelやloss倍率は追加していない。

項目1直後のStep 2でActual source L2は約0.28377、専用replay非ゼロは1/6。最終初期runでは5/24（20.83%）、Where比較viewに限ると5/5。最終学習済みrunでは2/24（8.33%）、Where比較viewに限ると2/2。**以前の長期8.81%という集計と分母・状態が異なるため、8.33%を長期改善とは解釈しない。**

既存group Critic lossがActual sourceの診断辞書から漏れていた箇所を分類し直した。lossの二重加算はしていない。

### 3.3 Candidate Actor / Proposal

Proposalは候補logit・候補pool選択へ使われる現役経路だった。専用Actual replayのコードは存在したが、対照が成立せず更新がなかった。教師成立後、項目1のStep 2だけでActor Actual L2=0.0566824、Proposal=0.0521414を観測した。新しいheadやloss倍率は不要だった。

最終初期24 StepではProposalはActual教師のある5 Stepで更新し、対照は0/24。学習済み最終版でも2 StepでActual教師と実更新を確認。教師がないStepに、無理にProposalを更新する処置はしていない。

### 3.4 Add / Prune / Adjust Where

Pruneには旧STE向けの0.005倍率とparameter hookの1/6がActual方策にも掛かっていた。SparsePCGC ana_den6訓練だけ、この旧減衰を外した。境界付きforward値はこの変更では同じ。他codecの旧処理は保持した。

同じpaired 6 Step条件で、Prune WhereのStep 2 Actual L2は`2.02667e-4 → 5.94990e-2`、実更新L2は`0.00234416 → 0.00427077`。単に1200倍になるとは限らない。hookの逆伝播位置と複数sourceが違うため、norm比をスカラー倍率と同一視しない。

WhereAllでAdd/Prune/Adjustの位置を同時に対比する既存設定は保持した。Pruneだけを順番にActual最適化する設計にはしていない。

### 3.5 Amount・再検証で発見した追加問題

初期Actualゼロはmatched Fine/Amount教師で解消した。次に、Fine 3 headとcoarse selectorの共有clipを分けた。各上限1、global上限10は変更せず、Fine 3操作は一緒のgroupを維持した。

実例では「coarseがFineを潰す」だけではなく、Step 4の大きなFine勾配がcoarseを縮めていた。分離前のcoarse最終L2約0.00564に対し、分離後は約0.80050。テストでは逆方向（coarse=100、Fine各0.1）の干渉も除去できることを確認した。

学習済みcheckpointで再検証すると、raw Amount logitが約−55.38 / −52.07 / −45.63だった。`tanh(raw/6)`のFP32丸めで当該Prune frameのderivativeが厳密に0になった。

同じtanhを、正側`1-2*sigmoid(-2x)`、負側`2*sigmoid(2x)-1`で計算する小さいscalar関数へ変更した。**STEで1を返す修正ではなく、同じ解析的関数の尾部derivativeを保持する数値安定化**である。テスト上forward差は最大約1.2e-7以内、train/testの式も同じ、他codecは元のtanh。極端なlogitの本来小さい勾配を人工的に大きくしてはいない。

安定化前後の学習済みpaired Step 2で、Prune AmountのActual L2は`0 → 1.96772e-8`、実更新L2は`0 → 3.96788e-6`。古い実装でも別frame/Stepには非ゼロがあるため、「旧版24 StepすべてPrune更新0」とは言わない。

さらに学習済み24 StepでFine=22、WhereAll=2、Amount/Gate=0となる教師取得の偏りを発見した。過去の累積coverageではAmount/Gateを十分試した扱いでも、新しい更新への教師はなかった。

pending pair完了を最優先に保ち、**最近開いた比較groupのcoverage**をSparsePCGCだけに追加した。未観測groupの初回測定と、既存min-shareから決める最大待ち時間を管理する。操作割合や報酬を指定するものではない。runtime sidecarへ保存し、明示warm-startでStep時計が巻き戻った場合だけ古い時計を捨てる。

最終学習済みrunは最初の8 StepでFine / WhereAll / Amount / Gate各pairを測定。Step 6のcoarse Actual L2=0.182395、Step 8のGate=79.4632（いずれもclip前）。以前の24 Stepに出たValidation no-op偏りも最終runでは再発しなかった。ただしこれを長期のno-op崩壊を防ぐ証明とはしない。

### 3.6 Point Transformer Feature Adapter

encoder凍結は意図した設計である。一方、Adapterは学習対象なのに、auto warmupが全探索期間へ延びて初期fusionが極小になっていた。勾配切断ではなく数値スケールによるFP32実更新ゼロだった。

SparsePCGC ana_den6の**auto** warmupだけ上限100 Stepを追加した（`--sparsepcgc_feature_auto_warmup_cap`）。明示warmup値、residual gate上限0.1、feature LR、固定encoderは変更していない。他codecのauto warmupも変更していない。

最終初期Step 2の更新直前L2は`2.33445e-15 → 6.49824e-10`、RMSは`4.82175e-17 → 1.34220e-11`、実更新L2は`0 → 3.16453e-6`。最終24 Stepで24/24に実更新を確認した。元のforwardと全Stepでbitwise同じ設計変更ではなく、固定されたbounded residualを早めに学習へ導入する変更である。

### 3.7 Plan Critic / Proxy / Geometry / Gate

- Plan Criticの回帰と重みは維持した。初期global clipは0/24、学習済みは旧版22/24→最終21/24。学習済みでは依然Plan Criticがglobal normの大部分を占める。clipを解除せず、他headのActual信号と実更新が残ることを計測した。長期の干渉を否定したわけではない。
- ProxyとEntropyは競合する。最終初期Actorのcosine範囲は約−0.994～−0.501、学習済みでは−0.993～−0.380。探索EntropyがProxy順位への集中に逆らうこと自体は目的どおりなので、見た目のnormを合わせるためには変更しなかった。
- 旧Geometry補助は別frameの系列baselineを使い、変えていない決定groupにも信用を与え得た。native経路は同一frame/anchor/group比較に限定した。
- 試行でGeometry-onlyとActual RDのcoarse勾配cosine約−0.975を観測した。最終版はGeometryが既にActual RDに入っている場合、第二の正規化Geometry-only方策教師を重ねない。**通常Geometry lossとActual RDの150Gは維持する**。RD側Geometry重みが0の場合だけ、測定済み同一anchorの制約付き補助を利用する。これは補助方策目的の変更であり、forward全体を厳密に同一にする最適化ではない。
- GateはActualのmatched group教師を取得する。新しい固定操作配分、割合の強制減衰、loss倍率の追加はしていない。

## 4. 最終比較表

勾配列は**初期24 Step内のActual source L2最大値**。同じStep・同じplanの比ではなく、教師が届く範囲を比較するためのenvelopeである。更新率列は同run内の最大`Δθ L2/θ L2`（比）。Amountは初期parameterが極小なので†の相対値だけで発散・改善を判断しない。全Stepの絶対更新・RMSは付録CSVにある。

| モジュール | 修正前の問題 | 原因 | 修正内容 | 修正前勾配 | 修正後勾配 | 更新率（後） | 解決状況 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Actual比較教師 | 厳密pair 0/24 | 別frame/groupでanchor未成立 | 連続2-view | — | pair 12/24 | — | 教師取得を確認 |
| Candidate Critic | 専用教師0/24、間接のみ | matched集合比較が不成立 | 教師を成立、source分類を修正 | 5.76e-4 | 0.422574 | 2.63e-4 | 集合教師5/24。個別因果寄与は未識別 |
| Candidate Actor | 弱い系列補正 | Actual対照不足 | 既存比較/replay経路を利用 | 7.65e-4 | 0.240351 | 3.22e-4 | Actualと更新を確認 |
| Proposal | 更新0/24 | 比較replay未成立 | 既存Actual経路へ教師供給 | 0 | 0.0773920 | 4.35e-4 | 更新5/24 |
| Add Where | Actualが弱い | 同一frame教師不足 | matched WhereAll | 1.39e-4 | 0.106567 | 4.40e-4 | Actualと更新を確認 |
| Prune Where | 極小 | 旧0.005×1/6減衰 | nativeのみ旧減衰を除外 | 1.97e-8 | 0.146811 | 5.61e-4 | Actualと更新を確認 |
| Adjust Where | Actualが弱い | 同一frame教師不足 | matched WhereAll | 0.00147234 | 0.872346 | 6.05e-4 | Actualと更新を確認 |
| Add Amount | 初期6 StepでActual 0 | Fine対照不足・共有clip | matched Fine、clip分離、安定tanh | 0.0156053 | 27.3098 | 2.40† | clip後更新。最大Δθ≈4.90e-5 |
| Prune Amount | 初期Actual 0、成熟時丸め0 | 対照不足・FP32 tanh飽和 | 同上 | 0.115058 | 106.024 | 0.923† | 成熟時のゼロ更新も改善 |
| Adjust Amount | 初期Actual 0 | 対照不足・飽和/clip | 同上 | 0.00962998 | 82.5857 | 2.32† | clip後更新。最大Δθ≈4.87e-5 |
| coarse Amount | 成熟24 StepでActual 0 | 累積coverageで新教師が後回し | 最近の教師coverage、Fine別clip | 2.04082 | 113.589 | 4.56e-5 | 成熟Step 6でもActual 0.182395 |
| Feature Adapter | 初期実更新0 | 長すぎるauto warmup | native auto cap100 | 1.47e-13 | 1.95e-7 | 7.41e-5 | 最終初期24/24更新 |
| Plan Critic | 大きいが直接教師あり | 全Step RD回帰 | loss/重み維持 | 2.13907 | 2.17482 | 9.47e-4 | 正常更新。成熟clip干渉は要継続観測 |
| Gate | 比較教師の取得偏り | 古い累積group coverage | 最近の教師coverage | 0.230939 | 12.9549 | 4.47e-4 | matched教師と更新を確認 |

Actual source最大値を大きくすることが目標ではない。Fine raw L2が大きいStepでも、実際に使われるgradientは上限clipを通り、Adamの実更新を別に確認した。

### 学習済みcheckpointでの代表的な再計測

| 対象（最終run） | Step | Actual source L2 | 最終gradient RMS | 実Δθ L2 | 更新率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prune Amount | 2 | 1.96772e-8 | 1.28511e-9 | 3.96788e-6 | 3.61910e-5 |
| Prune Where | 4 | 4.99235 | 0.00678123 | 0.00324728 | 3.87899e-4 |
| coarse Amount | 6 | 0.182395 | 0.00198381 | 0.000223065 | 4.46957e-5 |
| Gate | 8 | 79.4632 | 0.00745714 | 0.000542107 | 7.59814e-5 |
| Adapter | 2 | 5.16106e-12 | 2.70599e-9 | 0.000364545 | 8.58738e-5 |

教師がないanchor Stepや、別groupだけを変えたStepのActual gradientが0になるのは正常である。momentumによる実更新があるだけでも、Actual由来とは判断しない。

例えば成熟Prune Where Step 4では、`L_downstream`単独L2=14.3266、通常Geometry単独=16.4366、全loss backward後=4.21523、policy-only復元後=5.01779、Where clip後=0.999910、global clip後=0.601698、実更新L2=0.00324728だった。異なる方向の和なのでnormは加算しない。この例でもSTEを残すことで大きな数字を作ったのではなく、復元後のActual/方策sourceから更新している。

## 5. 固定Validationの性能・操作分布

Cは既存full-cloud Actual bitstream percent指標、Gは既存訓練のGeometry loss。**GをD1/D2 PSNR、またはcodec復号後の品質と呼ばない。** teacherは既存設定どおりskip-decodeであり、今回BD-rate評価はしていない。

| 開始状態・24更新 | 旧版C [%] | 最終C [%] | 旧版G | 最終G | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| 初期状態 | −2.238504334 | −2.220324505 | 0.02568104025 | 0.02567068301 | Cは0.018179829ポイント悪化、Gは約1.04e-5低下 |
| 381 checkpoint warm-start | −2.135117187 | −2.135117187 | 0.02541132644 | 0.02539843228 | C維持、Gは約1.29e-5低下 |

| 条件 | frame | Add / Prune / Adjust（実操作数） |
| --- | --- | --- |
| 初期旧版 | 0100 / 0101 | 218/589/43、217/594/43 |
| 初期最終 | 0100 / 0101 | 212/595/42、212/594/42 |
| 成熟旧版 | 0100 / 0101 | 123/530/25、123/529/25 |
| 成熟最終 | 0100 / 0101 | 123/530/25、124/529/25 |

同じActual CでもGは少し変わる。これを損失値の補正や性能偽装で一致させていない。最終版に対してno-op禁止などの人工的な採択条件は設けていない。

## 6. 失敗・棄却・制約

1. `00_baseline`は6 Stepの勾配測定後、診断wrapperのValidation呼出しミスで終了。勾配は使えるが完走成功として数えない。別Validationのみの旧runはruntime復元前のwrapperであり、主性能比較には使わない。
2. `00_baseline24`はCPU async worker初期化競合で0 Step終了。除外した。4本の主比較はすべてCPU serialで統一した。
3. `07_geometry`は途中にruntime保存対応のsource変更が入った試行。証拠は残すが、変更なしで完走した`07_geometry_dedup`、最終16/17を重視する。Geometry-onlyとActual RDの逆向きだけで性能改善を断定しない。
4. 最近のgroup coverage追加前の成熟24 Step試行12/15では、Validationでno-op、C=G=0になった。**G=0を品質改善成功とは扱わない。** Feature gateをoffにしても、旧runtimeを移してもno-opのまま。一方、旧parameterへ新runtimeを移すと旧操作を維持した。この範囲ではruntime単独より更新後parameterに依存する退行だった。Amount/GateのActual教師がその24 Stepに一度もなかったことを確認し、最終17で両教師を取得・退行非再発を確認した。複数修正の全因果分解を完了したとは言わない。
5. 初期状態の圧縮性能は改善していない。従って「7項目すべて解決して訓練性能改善まで達成」とは報告しない。多系列・多seed・十分な期間・同一encode予算/同一uniqueデータ予算の両方での比較が必要。
6. 本来小さいAmount derivative、Plan Criticによるglobal clip、Proxy/Entropy競合は残る。非ゼロにするための倍率やnorm均等化は追加していない。極端なlogitでは安定式でもunderflowし得るため万能な非飽和化ではない。
7. 個別候補の因果Actual寄与を教師にするには、同一anchorから候補/bundleだけを替える追加codec比較が必要。1 scalarを全候補の正解として流用しない。追加encode予算と識別単位を決めずに、完全な候補価値学習の解決済みとはしない。
8. 成熟runでは最近のcoverage修正後もFineが16/24、WhereAllが4/24で、専用Candidate教師は2/24だった。各groupへ到達することは確認したが、長期の教師頻度を十分に増やした保証はない。過去の8.81%を上回ることだけを目的に、差がないplanへ偽の教師を作ったり、強制的にnormを増やしたりはしていない。
9. AMP/FP16/BF16経路の同等な尾部derivative・実更新は今回未検証。特に半精度へgradientを戻す箇所では再びunderflowし得るため、今回のFP32実測を半精度全般へ一般化しない。

## 7. 計測の妥当性・成果物

最終4 runの通常Adam clone更新は実更新に一致し、最大誤差0。勾配source分解のrelative error最大は約4.98e-7。勾配L2だけでなくRMS、routing前後、decision/global clip、parameter差分、方向cosine、source除去時のAdam更新差分を保存した。

source除去は観測済みclip係数・Adam stateを固定した条件付き反実仮想であり、一意の加法的寄与率ではない。値が1を超えても誤りとは限らず、足して100%にはならない。

- [集計JSON](../experiments/results/20261022_credit_fixes/summary/summary.json)
- [数値付録の表](../experiments/results/20261022_credit_fixes/summary/metrics.md)
- [各loss sourceのL2・RMS](../experiments/results/20261022_credit_fixes/summary/source_gradients.csv)
- [backward→routing→clip→実更新](../experiments/results/20261022_credit_fixes/summary/routing_stages.csv)
- [source寄与・条件付きAdam除去実験](../experiments/results/20261022_credit_fixes/summary/source_conditional_adam.csv)
- [方向cosine](../experiments/results/20261022_credit_fixes/summary/gradient_cosines.csv)、[clip発生](../experiments/results/20261022_credit_fixes/summary/clip_events.csv)、[Actual教師成立](../experiments/results/20261022_credit_fixes/summary/actual_teachers.csv)
- [raw JSONL SHA256](../experiments/results/20261022_credit_fixes/summary/raw_manifest.json)。各runのsource snapshot・実パラメータ差分vector・実験checkpointも保存。

追加コードは`models/utils/training/sparsepcgc_actual_credit.py`、診断wrapper、集計コード、tests。既存の変更は`train.py`、`network.py`、`structure_actuator.py`、`train_runtime.py`とargsの2設定追加に限定した。G-PCC/OctAttentionは既存分岐を保持し、専用codecファイル、既存Network形状、既存checkpointは変更していない。

関連151テスト合格。対象は新規11テストに加え、training stability、Point Transformer features、geometry codec routes、geometry joint policy、SparsePCGC Actual semantics。9変更/追加PythonファイルはAST parse成功、`git diff --check`成功。model/runtime保存復元、Actual forward/backward、optimizer実更新を確認した。他codecの本訓練を再実行したわけではない。

### 再実行例（新しい出力先が必須）

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/maejima/miniconda3/envs/myOA/bin/python -B \
  experiments/verify_sparsepcgc_credit_fixes.py \
  --out experiments/results/credit_verify_new --steps 24 --validation 2 \
  --seed-cache experiments/results/20261022_credit_fixes/00_baseline24_serial/cache
```

学習済み検証は上記へ`--more-training /data/maejima/log/20261020/MyNetwork_train/pretrained/043722_SparsePCGC/381.pth`を追加する。既存入力checkpointはread-only、新しい出力先へ実験モデルを保存する。条件は現在のargsに従うため、今回の再現にはrun内の`effective_args.json`・source snapshotも照合する。

**最終判定：主要な教師不足・数値ゼロ・更新経路の不具合は修復を確認。長期の圧縮率改善、候補個別の因果価値識別、全状態での勾配競合解消は未達/未検証として残す。**
