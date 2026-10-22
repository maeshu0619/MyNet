# SparsePCGC 実学習3 Stepの勾配監査（2026-10-20）

元のtrain.pyを独立processで実行し、既定seed=21、UVG/BlueBackpack、Episode 1の最初の3 Stepを計測した。既存processの現在の重みではなく、同じ現行設定からのfresh実行である。既存の学習ソース・args・codec・checkpointは変更していない。

追加backwardはretain_graph=Trueで行い、モデルのParameter.gradとretainされた決定tensorのgradを退避・復元した。既存のbackward hookは有効。勾配ノルムは全対象Parameter要素を連結したL2であり、勾配平均ではない。グループは重複するためノルムを足し合わせない。

## 損失の実測値

| 項目 | Step 1 | Step 2 | Step 3 |
| --- | --- | --- | --- |
| L_total | -1.16702 | -3.54909 | -1.67208 |
| L_downstream | -4.66189 | -6.94935 | -5.239 |
| L_geom | 0.0291423 | 0.0226805 | 0.0237968 |
| weighted_geometry_primary | 3.50107 | 3.40208 | 3.56952 |
| loss_bit | -2.73319 | -2.25357 | -1.64673 |
| loss_nodes | -0.00507366 | -0.00537698 | -0.00364115 |
| loss_single | -0.000949753 | -0.00267994 | -0.00498865 |
| surrogate_loss_for_grad | -4.51204 | -3.55276 | -2.33639 |

Actual codec圧縮率[%]（ログ表示精度）: -1.55603, -2.317057, -1.747202。Actual encode自体は微分不能。上記L_downstreamは後段で上書きされた重み・proxy適用済み変数であり、生のActual圧縮率ではない。さらにw_com=3を掛けない。

## 損失単独のbackward勾配L2（routing/clip前）

| 損失 | Step 1 | Step 2 | Step 3 |
| --- | --- | --- | --- |
| L_total | 1.09456 | 7.20155 | 21.9332 |
| L_downstream | 0.128487 | 0.151098 | 0.141137 |
| L_geom | 0.00374884 | 0.00365066 | 0.00362351 |
| weighted_geometry_primary | 0.0162135 | 0.0197136 | 0.0195669 |
| loss_bit | 0.944089 | 1.11307 | 1.03894 |
| loss_nodes | 0.944088 | 1.11307 | 1.03894 |
| loss_single | 0.472047 | 0.556535 | 0.519471 |
| surrogate_loss_for_grad | 1.93081 | 2.27634 | 2.12476 |
| L_discrete_policy | 1.08713 | 7.19063 | 21.9327 |
| policy_source:actual_compression_policy | 1.0793 | 2.20059 | 2.26982 |
| policy_source:geometry | 0 | 6.84576 | 21.8127 |
| policy_source:proxy | 0.129591 | 0.150579 | 0.147537 |
| policy_source:entropy | 3.3221e-05 | 2.89358e-05 | 3.01624e-05 |
| policy_source:efficiency | 0 | 0 | 0 |

L_attr / L_policy / L_actuator / full_cloud_amountとその記録済み重み付き項は全3 Stepで値0、requires_grad=False。L_com / L_com_objective / weighted_compression_primaryはこの計測位置ではL_downstreamと同一tensor。重複測定していない（loss_aliases.csv）。

policy_source:actual_compression_policyはコード中の名称であり、純粋な圧縮項の完全分離ではない。RD targetは3*C + 150*G（および有効なら編集tie-break）を含む。policy_source:geometryはそれとは別の幾何方策項である。

## optimizer直前のモジュール/head別勾配L2

| グループ | Step 1 | Step 2 | Step 3 | 状態 |
| --- | --- | --- | --- | --- |
| module:encoder | 0 | 0 | 0 | frozen / frozen / frozen |
| module:structure_analyzer | 0 | 0 | 0 | absent / absent / absent |
| module:point_transformer_feature_adapter | 1.43978e-16 | 2.33445e-15 | 2.04199e-14 | nonzero / nonzero / nonzero |
| module:cost_attributor | 0 | 0 | 0 | frozen / frozen / frozen |
| module:cause_aggregator | 0 | 0 | 0 | frozen / frozen / frozen |
| module:policy_module | 0 | 0 | 0 | frozen / frozen / frozen |
| module:actuator | 1.08713 | 2.42822 | 2.48473 | nonzero / nonzero / nonzero |
| module:network_only_codec_policy | 0 | 0 | 0 | disconnected / disconnected / disconnected |
| module:network_k_proposal_policy | 0 | 0 | 0 | frozen / frozen / frozen |
| module:single_plan_student | 1 | 1 | 1 | nonzero / nonzero / nonzero |
| module:k_proposal_set_loss | 0 | 0 | 0 | absent / absent / absent |
| module:single_plan_distillation_loss_module | 0 | 0 | 0 | absent / absent / absent |
| module:algorithmic_proposal_selector | 0 | 0 | 0 | disconnected / disconnected / disconnected |
| module:full_cloud_amount_selector | 0 | 0 | 0 | disconnected / disconnected / disconnected |
| actuator:move_voxel_head | 4.90073e-06 | 0.00175378 | 0.00405828 | nonzero / nonzero / nonzero |
| actuator:drop_head | 1.43206e-09 | 6.08223e-06 | 5.40107e-06 | nonzero / nonzero / nonzero |
| actuator:add_head | 2.15773e-07 | 0.000709723 | 0.000382233 | nonzero / nonzero / nonzero |
| actuator:add_voxel_head | 2.4538e-06 | 0.000877764 | 0.000807835 | nonzero / nonzero / nonzero |
| actuator:operation_gate_head | 0.0120847 | 0.176065 | 0.00441512 | nonzero / nonzero / nonzero |
| actuator:subtree_move_source_head | 5.15488e-06 | 0.00164691 | 0.0039495 | nonzero / nonzero / nonzero |
| actuator:den6_candidate_actor | 2.53622e-05 | 0.0197986 | 0.0167049 | nonzero / nonzero / nonzero |
| actuator:den6_candidate_critic | 0 | 2.07411e-05 | 5.05244e-05 | zero / nonzero / nonzero |
| actuator:den6_candidate_proxy_critic | 0.129591 | 0.149245 | 0.146474 | nonzero / nonzero / nonzero |
| actuator:den6_candidate_proposal | 0 | 0 | 0 | disconnected / disconnected / disconnected |
| actuator:den6_plan_critic | 1.0793 | 2.20059 | 2.26982 | nonzero / nonzero / nonzero |
| actuator:drop_amount_head | 9.64327e-05 | 0.0759188 | 3.10489e-06 | nonzero / nonzero / nonzero |
| actuator:algorithmic_amount_selector_head | 0.00299833 | 0.984607 | 1 | nonzero / nonzero / nonzero |
| actuator:algorithmic_amount_residual_head | 0 | 0 | 0 | disconnected / disconnected / disconnected |
| actuator:add_amount_head | 3.91509e-05 | 0.00812656 | 1.07193e-06 | nonzero / nonzero / nonzero |
| actuator:move_amount_head | 3.63843e-05 | 0.157224 | 1.33764e-06 | nonzero / nonzero / nonzero |

frozen=requires_grad=False、disconnected=対象Parameter.gradが全てNone、zero=接続したgradはあるが全要素0、absent=そのgroupにParameterがない。structure_analyzer等のパラメータなし計算と、存在しないheadを同一の学習不良とは解釈しない。

single_plan_studentは別emulator optimizerで更新される。上記はmain clip時点に残るgrad。actual_parameter_updates.csvのoptimizer_ordinal=1がemulator、2が本体。Surrogateも別optimizerであり、15回の実更新を記録。各Stepに5回、clip後L2は約10。

## 解釈上の重要点

- 生の幾何pathwise勾配は約0.0036〜0.0037。重み・support balance・backward scale適用後は約0.016〜0.020。cp_lambda_geom=150だがcp_geometry_backward_scale=0.036のため、dL/dGは4.3249 / 5.4 / 5.4（ログ精度）。w_geom=10000はこのprimary objectiveの重みではない。
- 別の幾何方策項はStep 2で6.84576、Step 3で21.8127。したがって「幾何勾配は全経路で小さい」という判断は実測と矛盾する。Amount selector等に集中し、routing/操作別balance・clip後は大幅に変わる。
- 決定headはjoint backward後、保存されたpolicy-only勾配へ置き換えられる。L_geomやL_downstream単独の微分を、そのまま更新に使ったとは扱わない。
- 現行Whereは共通den6_candidate_actorに加え、操作別add_head/drop_head/move_voxel_head等にも全3 Stepで非ゼロ勾配を観測した。この3 StepではActorへのactual_compression_policy sourceは0で、Proxy/Entropy側の寄与を観測した。初期フレーム群での観測であり、後半・再訪frameも同じとは断定しない。
- 既定heuristic_guidance_online_grad_audit=Falseなので、通常ログのoperation_grad_norms=0は今回の実測ゼロを意味しない。この監査はParameter.gradを直接読み、操作別Where/Amountの非ゼロ勾配を確認した。
- Point Transformer feature adapterのgradは1e-16〜1e-14だが、3回とも実Parameter差分は0。事実上更新されていない。Candidate Proposalおよび別経路のselector類はこの3 Stepでは未接続。

## 検証

全3 Stepでemulatorとmainのoptimizer.stepとParameter差分を確認。probe errorなし。train.pyおよびmodels以下のPythonソースSHA256に変化なし。追加したのはexperiments以下の監査/集計コードと生成物のみ。

独立の既存訓練ログとの比較: {"reference": "/data/maejima/log/20261020/MyNetwork_train/SparsePCGC/043722.txt", "printed_precision_only": true, "equal": true, "compared_lines": 9}。一致はログの表示精度に限定し、全Parameterのbitwise一致を証明したものではない。

生成CSVは丸め前の実測値。measurements.jsonlが元データ。初回auditは入力ソース一覧のdirectory項目で停止し、v2は計測出力側の重複keywordで停止。結果は正常完走したv3だけを用いた。
