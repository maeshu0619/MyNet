# SparsePCGCのActual学習信号・勾配経路の追加監査

2026-10-20。結論は、**全モジュールの勾配L2を揃えるのではなく、比較可能なActual教師を作る頻度と、その教師が更新を支配できる条件を改善することが優先**である。初期Candidate Criticの小さいL2を、全期間の更新不良とみなす根拠はなかった。

## 実験と制約

- 既存`train.py`を独立processで6 Step実行した。UVG/BlueBackpack、Episode 1、seed=21、元の382 Episodeのscheduleを保持した。既存訓練の現在のcheckpointを追加計測したものではない。
- 既存train/models/args/codec/checkpoint/実行中processは変更していない。追加したのは`experiments/`の診断・集計コード、新規生成物、本レポート。
- 共有GPU負荷によるteacherのCPU自動退避を予備実行で観測し、既存GPU-GT cacheとの混在に符号量差が出た。最終実験はteacherをCPUへ明示固定し、GTも編集後も新規にnative encodeした。NetworkはGPU。GPU teacherとbitwise同一の測定とは主張しない。
- 追加backwardの前後でParameter.grad、保持した決定tensorのgrad、RNGを退避・復元した。Actual lossの計算式・選択方式・optimizer処理は変更していない。
- 今回の目的は勾配監査。既定どおり`skip_decode=True`で、現在の幾何lossは計測したが、最終復号点群のD1/D2 PSNRを追加評価した実験ではない。以下の改善案による品質維持は未検証。

## 1. ActualからWhere/Amountまで

```text
Actual codecのCと現在のG
  → detachedなRD教師（3*C + 150*G + 有効なtie-break）
  → 比較baseline／同一frame・anchorの対照／replay
  → log-probabilityの方策loss、候補集合の比較loss、Plan Critic回帰
  → 決定head用policy-only勾配を保存
  → L_total.backward()
  → 決定headのgradを保存済みpolicy-onlyへ復元
  → decision group別clip → global clip → Adam
```

Actual codecを微分しないことは正常。問題は、Actualの比較教師が成立していない間に、別のpathwise圧縮勾配も更新前に排除されることにある。ただしhardなargsort/int planと不整合なSTEの混入を防ぐ意図があるため、復元の無条件削除は推奨しない。

| Step | 探索group | sequence/group baseline既知 | global Actual補正 | 同一frameの厳密pair | Candidate専用replay |
| --- | --- | --- | --- | --- | --- |
| 1 | WhereAll | いいえ | 無効 | なし | なし |
| 2 | Fine | いいえ | 無効 | なし | なし |
| 3 | Amount | いいえ | 無効 | なし | なし |
| 4 | Gate | いいえ | 無効 | なし | なし |
| 5 | WhereAll | はい | 有効 | なし | なし |
| 6 | WhereAll | はい | 有効 | なし | なし |

初期4 Stepはframeもgroupも異なり、`global_credit_available=False`で`0*log_prob`を返す。Step 5・6ではsequence/group EMAとの比較が可能になり、Actual由来のWhere勾配が発生した。これは別frameを跨ぐ弱い補正であり、同じframeの反実仮想教師ではない。Amountはこの6 StepではActual方策由来勾配が0だったが、長期にも0とは断定しない。

## 2. 圧縮勾配の消失位置

代表としてStep 2。数値はL2。圧縮単独とjointは異なる量なので、単純な減衰率として比較しない。

| 対象 | L_downstream単独 | 全loss backward後 | policy-only復元後 | decision clip後 | global clip後 |
| --- | --- | --- | --- | --- | --- |
| Add Where | 0.0104761 | 0.0113279 | 0.00112877 | 0.00112877 | 0.00112877 |
| Prune Where | 0.0164377 | 0.00325719 | 6.08224e-6 | 6.08224e-6 | 6.08224e-6 |
| Adjust Where | 0.000220597 | 0.00241137 | 0.00240584 | 0.00240584 | 0.00240584 |
| Add Amount | 0.0152964 | 0.0403032 | 0.0555995 | 0.00812656 | 0.00812656 |
| Prune Amount | 0.148797 | 0.668211 | 0.519414 | 0.0759188 | 0.0759188 |
| Adjust Amount | 0.00816009 | 1.06752 | 1.07568 | 0.157224 | 0.157224 |
| 共通Amount selector | 0 | 6.73639 | 6.73639 | 0.984607 | 0.984607 |

Whereは操作別の位置/方向headを合算したL2。共通Candidate Actorは別group。

sourceを追跡すると、この決定head群では**L_downstreamと通常幾何lossのpathwise寄与は復元後0**となる。残るのはActual/Proxy/Entropy/別幾何方策など。従って`w_com`だけを増やしても、除去されるSTE経路を通した更新は増えない。なお`w_com`はActual RD教師にも使われるため、全経路に無影響という意味ではない。

## 3. Candidate CriticとPlan Critic

Candidate Criticの専用Actual replay損失は6 Stepとも不成立だった。観測したActual sourceのCandidate Critic勾配は、全6 StepでPlan Critic回帰lossからの勾配と一致した。

経路は`選択候補Qのmean/std → 18次元Plan feature → Plan Critic → Actual RD回帰`。Candidate Critic入力はdetachされ、ここからWhereへは流れない。両Criticの最終層はゼロ初期化され、Step 1はPlan最終層weight=0で入力側勾配も0。Step 2直前のPlan最終層weight L2は約8e-4であり、初期Candidate勾配は間接経路で減衰する。

Plan Criticは毎StepのActual RDとreplayで直接回帰する。重み0.1に外側backward倍率10が残り実効係数1。global Actor補正は0.1×0.05=0.005。専用Candidate replay係数0.4は、同一frame/anchor/group等の条件成立時だけ有効。損失重みだけでなく、教師成立条件と回帰位置が異なる。

| Step 2対象 | 更新直前L2 | RMS | 実更新のL2 | 更新率 Δθ L2 / θ L2 |
| --- | --- | --- | --- | --- |
| 共通Actor | 0.0197986 | 0.00116462 | 0.00110654 | 0.000319907 |
| Candidate Critic | 2.07411e-5 | 1.15586e-6 | 0.000419304 | 0.000118800 |
| Plan Critic | 1.89739 | 0.0456178 | 0.00283363 | 0.000607392 |
| Add Where | 0.00112877 | 8.56435e-6 | 0.00338756 | 0.000296121 |
| Prune Where | 6.08224e-6 | 6.85478e-8 | 0.000850130 | 0.000111656 |
| Adjust Where | 0.00240584 | 1.82538e-5 | 0.00501751 | 0.000498146 |
| 共通Amount selector | 0.984607 | 0.0154946 | 0.000178608 | 0.0000364801 |

RMSの分母はgroup内の全学習対象要素数。操作別Amountの初期パラメータはゼロ/極小で、相対更新率は未定義/非常に大きくなる。絶対更新量も併記し、比だけを発散と判断しない。Step 2のAdd/Prune/Adjust Amount実更新L2はそれぞれ4.8732e-5 / 4.88308e-5 / 4.88237e-5だった。

## 4. optimizer更新へのsource寄与

Adam state・Parameterをcloneし、観測したrouting/clip係数を固定したまま1 sourceだけ除去した更新との差を計算した。通常更新cloneは全6 Stepで実更新とbitwise一致、最大誤差0。元optimizer/stateは変更していない。

以下は`通常更新とsource除去更新の差のL2 / 実更新L2`。一意な寄与配分率ではなく、momentumや勾配方向の競合で1を超え得る。source間の値を加算しない。

| 対象 | Step | 圧縮pathwise除去 | 幾何pathwise除去 | Actual RD source除去 | candidate-local Proxy除去 | 別幾何方策除去 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 共通Actor | 2 | 0 | 0 | 0 | 1.72748 | 0 |
| Candidate Critic | 2 | 0 | 0 | 1 | 0 | 0 |
| Add Where | 2 | 0 | 0 | 0 | 1.13675 | 0 |
| Prune Where | 2 | 0 | 0 | 0 | 1.00474 | 0 |
| Adjust Where | 2 | 0 | 0 | 0 | 1.08821 | 0 |
| Add Amount | 2 | 0 | 0 | 0 | 0 | 0.511920 |
| Prune Amount | 2 | 0 | 0 | 0 | 0 | 0.279556 |
| Adjust Amount | 2 | 0 | 0 | 0 | 0 | 0.483110 |
| 共通Amount selector | 2 | 0 | 0 | 0 | 0 | 1.33823 |
| 共通Actor | 6 | 0 | 0 | 0.00334996 | 0.205861 | 0 |
| Add Where | 6 | 0 | 0 | 0.0968974 | 0.581583 | 0 |
| Prune Where | 6 | 0 | 0 | 0.00127024 | 0.133871 | 0 |
| Adjust Where | 6 | 0 | 0 | 0.0131756 | 0.655161 | 0 |

元のsource辞書はpolicy_loss全体を分類していない。operation/Amount/Fine local credit、追加Entropy、条件成立時のcritic pair等は残差`policy:unclassified`として監査し、CSVには寄与を別記した。上表のProxyはcandidate-local sourceのみである。Actual sourceもRD教師に幾何を含み、純粋な圧縮と幾何の完全分離ではない。source勾配の合成再構成誤差は最大相対2.47e-7。

## 5. 方向競合・clip

- Step 2のPrune Whereでは圧縮pathwiseと幾何pathwiseのcosineが約−1。しかし両方が後段で除去されるため、これだけを実更新競合の原因とは扱わない。
- 共通Actorのcandidate-local ProxyとEntropyのcosineは約−0.793。Entropyのノルムは小さい。ゼロ勾配のcosineは未定義として記録し、0とは扱わない。
- decision clipはAmountで5/6 Step、Where/Gateで0/6。global clipは0/6（上限10）。Amountはcoarse selectorと3操作Fineをまとめて上限1へclipする。Step 2は6.84→1、Step 3は21.81→1であり、Fineの小さい別sourceも巻き添えで縮小される。
- Whereのgroup clipには共通Actorと移動元headが含まれないため、Where clip未発生だけで全Where経路を保証しない。

## 6. 長時間ログによる前提の修正

既存processのログを読み取り専用でprefix snapshotした。監査Step数12,600、anchor再利用3,558、厳密group pair成立914、専用Candidate replay損失非ゼロ1,110。専用Candidate教師は全体の8.81%、WhereAll内では1,110/4,435=25.03%だった。探索group=noneの3,050 Stepでは、この専用replay損失は0だった。

**初期の「Critic勾配が極小」を全期間へ一般化しない。** 既存GPU訓練のsource probeでは次の値を確認した。いずれもsource単独backwardのL2で、更新直前値ではない。

| 実行中訓練のStep | Actual source→Actor | Actual source→Candidate Critic | Actual source→Plan Critic |
| --- | --- | --- | --- |
| 12550 | 0.00277848 | 1.81897802 | 8.56944378 |
| 12580 | 0.00216905 | 1.99106019 | 4.54781945 |
| 12600 | 0.00192873 | 0.62780035 | 6.04405833 |

Actualの経路は長期に常時切れているわけではない。ただし専用Candidate教師なしでもPlan経由の勾配は発生するので、大きい勾配だけで候補判別の学習成功とは扱わない。ログのフラグ・損失成立頻度は、現在のParameter.gradを新規に測定した値ではない。

## 推奨案のまとめ（今回は未実装）

| 問題箇所 | 原因 | 実測値 | 推奨修正案 | 期待できる効果 | 想定される副作用・注意 |
| --- | --- | --- | --- | --- | --- |
| 初期Actual Actor/操作教師 | 別frame/groupでbaseline未成立、厳密pairなし | 初期4 Stepは方策補正0、Step 5からWhereに非ゼロ | 同一frameのanchor→variantを短い2-view単位で処理。3操作共同WhereAllを維持し、他の変更因子を固定 | 因果比較可能なActual ΔRDを得る | 系列多様性低下。1 Stepのencode数を保っても1系列のコスト増。固定部分・policy versionの保存が必要 |
| Candidate Criticの候補教師 | 初期はPlan feature経由の間接回帰のみ | 専用loss0/6、S2 L2=2.07e-5でも実更新4.19e-4 | matched候補集合のQ差・不確実性を直接学習。plan scalarを各点にコピーしない | Plan難易度ではなく候補集合の相対価値を学ぶ | 候補間相互作用を単点に誤配分しない。別pairで順位・calibrationを評価 |
| WhereのActual寄与 | 弱いglobal補正と高頻度local教師の差 | S6 Actual除去時の更新変化はAdd9.69%、Prune0.127%、Adjust1.32% | 有効pair確保を先行。その後matched replay係数0.4→0.8等の1x/2x比較。跨frame global weight 0.05はまず据置 | 信頼できるActual比較信号の実更新への影響を増やす | 比較教師が誤ると増強も逆効果。最適倍率・品質維持は未確認 |
| 圧縮STEの増強が効かない | 決定headでpolicy-onlyへ復元 | S2 Prune圧縮単独0.01644、復元後の同source寄与0 | 復元を無条件撤去せず、Actual比較教師を増やす。STE混合は別ablation | hard planと不整合な勾配を避ける | STE復活はbiasedな学習になる。`w_com`はRD教師にも影響する |
| Fine Amountがcoarseに巻き添え | 共通clip groupをcoarse幾何方策が支配 | Amount clip5/6、最大21.81→1 | coarse/Fineの別clipを比較しglobal安全clip維持。別幾何方策0.55→0.275は0.5x試験候補。幾何loss重み150は保持 | Fine側の有効信号を残す | 幾何品質悪化の危険。0.5x単独では21.81→約10.9で依然clipするため、倍率低下だけを解決とは扱わない |
| L2だけに基づく一律増幅 | Adam・要素数・momentumを無視 | S2 Candidate更新率1.19e-4、Plan6.07e-4。長期Candidate勾配0.63〜1.99 | RMS、実更新率、教師成立率、Actual符号一致率で判断。倍率を揃えない | 不要な増幅や大きい更新を回避 | ゼロ初期headの相対更新率は未定義。追加backwardは低頻度診断に限定 |

0.5x/2xは今後の独立比較実験の候補であり、検証済みの最適値ではない。**今回、重み変更後の訓練・圧縮率改善・復号後品質維持を実証したとは報告しない。** 採用前に同じ初期state・frame・候補・codec backendでcontrolと比較し、Actual bits、GTに対する幾何品質、操作率、教師成立率を確認する。

## 成果物と検証

- [詳細レポート](/home/maejima/MasterEx/myNet/experiments/results/20261020_sparsepcgc_gradient_routing_cpu/summary/report.md)
- [routing各段階・RMS・実更新率CSV](/home/maejima/MasterEx/myNet/experiments/results/20261020_sparsepcgc_gradient_routing_cpu/summary/routing_stages.csv)
- [source寄与・条件付きAdam比較CSV](/home/maejima/MasterEx/myNet/experiments/results/20261020_sparsepcgc_gradient_routing_cpu/summary/source_contributions_and_conditional_adam.csv)
- [方向cosine CSV](/home/maejima/MasterEx/myNet/experiments/results/20261020_sparsepcgc_gradient_routing_cpu/summary/gradient_cosines.csv)
- [clip履歴CSV](/home/maejima/MasterEx/myNet/experiments/results/20261020_sparsepcgc_gradient_routing_cpu/summary/clip_events.csv)
- [実行中logのprefix集計・SHA256](/home/maejima/MasterEx/myNet/experiments/results/20261020_sparsepcgc_gradient_routing_cpu/summary/live_log_snapshot.json)
- [検証結果JSON](/home/maejima/MasterEx/myNet/experiments/results/20261020_sparsepcgc_gradient_routing_cpu/summary/validation.json)

6回の本体更新と6回の別Emulator更新を確認。Adam cloneと実更新の最大誤差0、source再構成相対誤差は最大2.47e-7。既存学習ソースSHA256は開始/終了で一致した。既存processは継続したまま。
