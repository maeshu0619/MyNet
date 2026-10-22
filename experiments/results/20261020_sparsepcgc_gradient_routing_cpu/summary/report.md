# SparsePCGC Actual creditとoptimizer更新の追加監査

2026-10-20。既存train.py/models/args/codec/checkpoint/processは変更せず、experiments以下の独立wrapperでEpisode 1の6 Stepを実行した。seed=21、UVG/BlueBackpack、元の382 Episodeのscheduleを保持。本訓練の現在のcheckpointを測定したものではない。

共有GPU高負荷で予備実行がauto→CPUへ退避し、GPUで保存済みのGT cacheとの混在に数bit差が出た。予備実行v2は最終計測に採用しない。最終runはteacherを明示CPUに固定し、GT Actual cacheを流用せず、GTと編集後の両方をCPU native codecで符号化した。Networkは通常のGPU実行。静的候補/構造/特徴cacheは既存生成物のコピーを独立出力先で使用。GPU teacherとbitwiseに同じ教師値とは主張しない。

## 1. 勾配経路と初期ゼロの原因

`Actual C,G → detached RD教師 → 方策/比較損失・Plan Critic → retained backwardで決定head勾配保存 → joint backward → 決定headをpolicy-onlyへ復元 → Where/Amount/Gate別clip → global clip → Adam`

Actual codecを微分しないこと自体は正常。score-function/比較教師から更新する。detached Actual scalarを強引につなぎ直す修正は提案しない。

| Step | 探索group | frame baseline既知 | sequence/group baseline既知 | global利用可能 | 厳密pair | Actor replay pair |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | WhereAll | False | False | False | False | 0 |
| 2 | Fine | False | False | False | False | 0 |
| 3 | Amount | False | False | False | False | 0 |
| 4 | Gate | False | False | False | False | 0 |
| 5 | WhereAll | False | True | True | False | 0 |
| 6 | WhereAll | False | True | True | False | 0 |

初期4 StepはWhereAll/Fine/Amount/Gateの別groupで、新しいframeである。frame+groupとsequence+groupのbaselineが未初期化なのでpolicy_core_rawは0*log_probになる。厳密pairには同じframe/anchor/group、再利用可能anchor、比較plan、RD差>=0.05等が必要だが、この6 Stepは異なるframeなので直接比較教師も成立しない。Step 5/6では同じsequenceのWhereAll baselineが既知になり、弱いglobal補正だけが成立した。これは厳密な同一frame反実仮想ではない。

| Actual方策由来の勾配L2 | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | --- | --- | --- | --- | --- | --- |
| head:den6_candidate_actor | 0 | 0 | 0 | 0 | 0.000134107 | 0.000154164 |
| head:den6_candidate_critic | 0 | 2.07411e-05 | 4.98193e-05 | 7.58951e-05 | 9.12705e-05 | 9.81867e-05 |
| head:den6_plan_critic | 0.776789 | 1.89739 | 2.1192 | 0.271816 | 1.54486 | 1.66841 |
| AddWhere | 0 | 0 | 0 | 0 | 2.73183e-05 | 8.52477e-05 |
| PruneWhere | 0 | 0 | 0 | 0 | 6.2633e-09 | 8.65186e-09 |
| AdjustWhere | 0 | 0 | 0 | 0 | 5.39829e-05 | 0.000202148 |
| AddAmount | 0 | 0 | 0 | 0 | 0 | 0 |
| PruneAmount | 0 | 0 | 0 | 0 | 0 | 0 |
| AdjustAmount | 0 | 0 | 0 | 0 | 0 | 0 |
| AmountSelector | 0 | 0 | 0 | 0 | 0 | 0 |
| Gate | 0 | 6.10339e-05 | 8.55923e-05 | 0.000236189 | 0.000113277 | 0.000127111 |

## 2. compression勾配はどこで失われるか

以下はStep 2。joint列は全損失を合成した勾配であり、圧縮単独列と同じ量ではない。決定head復元後は圧縮pathwise寄与が厳密に0になる。そのためw_comだけを増やしても、そのSTE経路を通してWhere/Amountの最終更新は増えない。

| group | compression単独 | joint backward | policy復元後 | decision clip後 | global clip後 |
| --- | --- | --- | --- | --- | --- |
| head:den6_candidate_actor | 0 | 0.0197986 | 0.0197986 | 0.0197986 | 0.0197986 |
| head:den6_candidate_critic | 0 | 2.07411e-05 | 2.07411e-05 | 2.07411e-05 | 2.07411e-05 |
| head:den6_plan_critic | 0 | 1.89739 | 1.89739 | 1.89739 | 1.89739 |
| AddWhere | 0.0104761 | 0.0113279 | 0.00112877 | 0.00112877 | 0.00112877 |
| PruneWhere | 0.0164377 | 0.00325719 | 6.08224e-06 | 6.08224e-06 | 6.08224e-06 |
| AdjustWhere | 0.000220597 | 0.00241137 | 0.00240584 | 0.00240584 | 0.00240584 |
| AddAmount | 0.0152964 | 0.0403032 | 0.0555995 | 0.00812656 | 0.00812656 |
| PruneAmount | 0.148797 | 0.668211 | 0.519414 | 0.0759188 | 0.0759188 |
| AdjustAmount | 0.00816009 | 1.06752 | 1.07568 | 0.157224 | 0.157224 |
| AmountSelector | 0 | 6.73639 | 6.73639 | 0.984607 | 0.984607 |
| Gate | 0.00310251 | 0.174194 | 0.176065 | 0.176065 | 0.176065 |

保存/復元はhardなargsort/int planと不整合なSTEの混入を防ぐ設計であり、無条件に復元を削除することも推奨しない。代わりに実codecの比較教師を成立させる。

## 3. Candidate CriticとPlan Criticの違い

Candidate Criticには初期6 Stepで専用Actual replay損失が成立していない。それでもStep 2以降に非ゼロになるのは、選択候補Qのmean/stdが18次元Plan featureに入り、Plan Critic損失から逆伝播するため。component:plan_critic_loss_weightedのCandidate Critic勾配とActual sourceの同勾配は今回全Stepで一致した。候補自身の独立したActual Δbits教師を学習できた証拠ではない。

両Critic最終層はゼロ初期化。Step 1のPlan最終層weight L2=0なので、Plan headには更新があるが、入力側Candidate Criticへは勾配が届かない。Step 2直前のPlan最終層weight L2は約8e-4であり、さらに候補Qのmean集約が挟まる。Candidate Critic入力特徴もdetachされるので、その間接経路からWhereへは流れない。

Plan Criticは毎StepのActual RDを直接回帰し、過去planもreplayする。係数は0.1、外側policy backward倍率10がこの回帰には残り、実効係数は1。一方global Actor補正の実効係数はpolicy_weight 0.1 × global_actual_credit_weight 0.05 = 0.005。専用Candidate replay係数0.4は比較pair成立時にだけ有効で、初期6 Stepでは不成立。選択頻度よりも教師の成立条件・回帰位置・初期化/集約の違いが支配的。

## 4. RMS・実パラメータ更新・source寄与

RMS=勾配L2/sqrt(group内全学習対象要素数)。更新率=||Δθ||/||θ_before||。ゼロ初期値では未定義、極小初期値では大きくなるため、操作Amountの更新率だけを発散とは解釈しない。

| Step 2 group | grad L2 | grad RMS | Δθ L2 | θ_before L2 | 更新率 |
| --- | --- | --- | --- | --- | --- |
| head:den6_candidate_actor | 0.0197986 | 0.00116462 | 0.00110654 | 3.45895 | 0.000319907 |
| head:den6_candidate_critic | 2.07411e-05 | 1.15586e-06 | 0.000419304 | 3.5295 | 0.0001188 |
| head:den6_plan_critic | 1.89739 | 0.0456178 | 0.00283363 | 4.66524 | 0.000607392 |
| AddWhere | 0.00112877 | 8.56435e-06 | 0.00338756 | 11.4398 | 0.000296121 |
| PruneWhere | 6.08224e-06 | 6.85478e-08 | 0.00085013 | 7.6138 | 0.000111656 |
| AdjustWhere | 0.00240584 | 1.82538e-05 | 0.00501751 | 10.0724 | 0.000498146 |
| AddAmount | 0.00812656 | 0.00107639 | 4.8732e-05 | 4.4968e-05 | 1.0837 |
| PruneAmount | 0.0759188 | 0.0100557 | 4.88308e-05 | 5.31576e-05 | 0.918605 |
| AdjustAmount | 0.157224 | 0.0208249 | 4.88237e-05 | 4.42523e-05 | 1.1033 |
| AmountSelector | 0.984607 | 0.0154946 | 0.000178608 | 4.89603 | 3.64801e-05 |
| Gate | 0.176065 | 0.0019681 | 0.00317448 | 6.63357 | 0.000478548 |

Candidate CriticはL2が2e-5でもAdam後に4e-4程度更新される。PlanとのL2差をそのまま学習速度差とは扱えない。問題は候補判別のActual教師が十分かという点。

source別寄与は、実optimizerのAdam stateとParameterをcloneし、観測したrouting/clip係数を固定したまま1 sourceだけ引いた更新と比較した。元optimizer/stateには書き込まない。通常更新cloneは全6 Stepで実更新とbitwise一致（最大誤差0）。これは条件付きLOOであり、寄与の一意な分配率ではない。負方向の競合・momentum・非線形性により100%を超えることがあり、source間の値を合計してはいけない。clipを再計算する別policyの学習実験でも、codec品質の改善検証でもない。

| Step 2 group | compression STE除去 | geometry pathwise除去 | Actual RD除去 | Proxy除去 | 別geometry方策除去 |
| --- | --- | --- | --- | --- | --- |
| head:den6_candidate_actor | 0 | 0 | 0 | 1.72748 | 0 |
| head:den6_candidate_critic | 0 | 0 | 1 | 0 | 0 |
| head:den6_plan_critic | 0 | 0 | 0.981195 | 0 | 0 |
| AddWhere | 0 | 0 | 0 | 1.13675 | 0 |
| PruneWhere | 0 | 0 | 0 | 1.00474 | 0 |
| AdjustWhere | 0 | 0 | 0 | 1.08821 | 0 |
| AddAmount | 0 | 0 | 0 | 0 | 0.51192 |
| PruneAmount | 0 | 0 | 0 | 0 | 0.279556 |
| AdjustAmount | 0 | 0 | 0 | 0 | 0.48311 |
| AmountSelector | 0 | 0 | 0 | 0 | 1.33823 |
| Gate | 0 | 0 | 0.000185531 | 0 | 0.446023 |

表の単位は||通常更新−source除去更新||/||実更新||。元のsource辞書はpolicy_loss全体を網羅しないため、CSVではその他policy（policy:unclassified）を残差として明示した。この6 Stepでは厳密比較が未成立で、局所operation/Amount/Fine creditや追加Entropy等が残る。厳密pair成立時には辞書に分類されないcritic pair項もあり得るため、残差を一律にProxyとは呼ばない。表のProxyはcandidate-local sourceのみ。共通ActorのProxy除去が1を超えるのは、少量Entropyや過去momentumとの向きが変わるためで、そのsourceが学習の172%を占めるという意味ではない。

| Step 5/6 group | S5 Actual条件付き更新変化率 | S6 Actual条件付き更新変化率 |
| --- | --- | --- |
| head:den6_candidate_actor | 0.00289779 | 0.00334996 |
| AddWhere | 0.0120096 | 0.0968974 |
| PruneWhere | 0.000490296 | 0.00127024 |
| AdjustWhere | 0.00818184 | 0.0131756 |

## 5. 方向競合とclip

| Step 2 group | 比較 | cosine |
| --- | --- | --- |
| head:den6_candidate_actor | policy:proxy vs policy:entropy | -0.793018 |
| PruneWhere | L_downstream vs weighted_geometry_primary | -1 |
| PruneWhere | policy:proxy vs policy:entropy | -0.56177 |

Prune Whereのpathwise圧縮/幾何は反対向きだが、その両方が後段で置き換えられる。このcosineだけを最終更新競合の原因とはしない。ActorではProxy/Entropyの競合を確認したが、Entropyはノルムが小さい。ゼロ勾配同士・一方ゼロのcosineは0と捏造せず未定義とした。

| clip対象 | 発生回数/6 | 備考 |
| --- | --- | --- |
| where | 0/6 | 上限1 |
| amount | 5/6 | 上限1 |
| action | 0/6 | 上限1 |
| global | 0/6 | 上限10 |

Amountはcoarse selectorと3つのFine Amountを同じgroupでclipする。Step 2では約6.84→1、Step 3では約21.81→1。coarse geometry方策が大きいStepに、Fine Amountの別sourceまで同じ縮小係数を受ける。Whereのgroup clipには共通Candidate Actorとsubtree_move_source_headは含まれていないため、Where clip発生0はこれらの勾配の健全性を保証しない。global clipは今回未発生。

## 6. 長時間ログでの成立頻度

既存実行中logの読み取り専用prefix snapshot。ログのavailable/weighted/pairsを集計したもので、live Parameter.gradの再計測ではない。current checkpointの勾配方向・更新率を証明する値ではない。

| 項目 | Step数 | 全audit Step比 |
| --- | --- | --- |
| steps | 12600 | 1 |
| anchor_reused | 3558 | 0.282381 |
| group_pair_available | 914 | 0.0725397 |
| global_credit_available | 12550 | 0.996032 |
| actual_set_credit_pairs_nonzero | 444 | 0.0352381 |
| actual_actor_replay_pairs_nonzero | 1110 | 0.0880952 |
| candidate_critic_direct_nonzero | 1110 | 0.0880952 |

既存GPU訓練のsource単独backwardでは、Step 12600のActual source→Actor=0.00192873、Candidate Critic=0.62780035、Plan Critic=6.04405833だった。これは更新直前の値ではない。初期の極小Candidate勾配が長期にも続いているわけではない。大きくなった間接Plan勾配と、候補集合の直接比較教師は区別する。

したがって初期3 Stepのゼロを全学習期間の切断とは判断できない。一方、専用Candidate教師や厳密pairは毎Step成立していない。anchor/replay capacity=512、pending=1024、max age=1600、min delta=0.05などが成立率に影響し得る。未成立を全てLRU eviction/差分marginのせいにしたとは主張しない。その内訳にはmatchedキー・frame再訪間隔・候補集合差・delta・ageを追加観測する必要がある。

## 7. 推奨修正案（今回は未実装・未検証）

| 問題箇所 | 原因・実測根拠 | 推奨修正案 | 期待効果 | 副作用/検証条件 |
| --- | --- | --- | --- | --- |
| Actual教師の成立 | 初期4 Stepは別frame/group。専用pairなし。S5/6で弱いsequence補正のみ成立 | 最優先: anchor→variantを同一frameの短い2-view単位で処理。Whereは3操作共同WhereAll、Amount/Gate/Fineは他因子固定。codec予算内でbaseline結果を再利用 | 因果的に比較できるActual ΔRDがActor/Candidate Critic/操作headへ届く | 同じframeの反復で系列多様性低下。per-Step encode数固定でも1系列当たりの計算増。anchor ageとpolicy versionを保存 |
| Candidate Criticの教師 | 専用Actual replayが初期6 Stepなし。観測勾配は全てPlan feature経由 | 同一frame/anchorの交換candidate集合に対するgroup-level Q差/不確実性を直接学習。plan scalarを個々の点へコピーしない | 候補集合間のActual順位を学ぶ。間接的なplan難易度回帰への依存を減らす | 相互作用を単点へ誤配分すると逆効果。独立pairで符号一致率・calibrationを検証 |
| 初期化による遅延 | Plan/Candidate最終層zero。S1はCandidateへPlan勾配0 | 直接教師を先に確保。必要ならPlan最終層zero vs std=1e-3の小初期化を独立ablation | 初回から入力側へ小さい勾配を返せる | zero解除だけでは教師不足は直らない。初期Qの任意biasをpolicy authorityへ入れない |
| WhereのActual寄与が小さい | S5/6で流れるがProxy優勢。global実効係数0.005 | 厳密pair教師成立後にmatched replay係数0.4→0.8等の1x/2xを比較。跨frame global weight 0.05はまず据置。無条件10x以上は避ける | 信頼できるActual比較の更新への寄与を増やす | 比較教師が誤ると増強も逆効果。圧縮/幾何の検証なしに最適値とは呼ばない |
| Amountのgroup clip | 6 Step中5 Stepでclip。coarse geometry方策とFineを同時縮小 | coarse/Fineを別観測・別clip候補にし、global安全clipは維持。別geometry方策0.55→0.275の0.5x ablationを候補。cp_lambda_geom=150は保持 | coarseの大勾配がFine Actual信号を巻き添え縮小するのを抑える | 幾何拘束を弱める危険。clip解除や倍率変更単独を解決としない。D1/Chamfer/Actualを同時評価 |
| 圧縮STEを強めても更新されない | policy-only復元で決定headのpathwise寄与0 | 復元を無条件撤去しない。整合するActual比較教師を増やす。必要なら整合性を検証済みの小さいSTE残差を別ablation | hard planと無関係なproxy勾配の混入を防ぎながら学習する | STE混合はbiased。圧縮重みだけ増やすと表示loss/他moduleのみ変化する可能性 |
| L2だけの判断 | Candidate Critic L2極小でも実Adam更新は非ゼロ | RMS・Δθ/θ・teacher成立率・source LOO・Actual符号一致率を継続記録。freeze/None/zeroを区別 | 不要な増幅を避け、教師不足と更新不足を分離 | 追加backward/LOOは重いので通常訓練へ毎Step追加しない。低頻度診断に限定 |

倍率は次の独立比較実験の候補であり、最適値/品質維持を保証したものではない。今回、重み変更後の訓練やRD改善実験は行っていない。同じ初期state・frame・候補・Actual backendでcontrolと比較し、Actual bits、同じGTに対する幾何品質、操作率、教師成立率を検証してから採用する。

## 検証と成果物

{
  "steps": 6,
  "main_optimizer_updates": 6,
  "emulator_updates": 6,
  "source_files_changed": [],
  "max_adam_clone_error": 0.0,
  "max_routing_decomposition_relative_error": 2.468914180749108e-07,
  "teacher_device": "cpu"
}

source/routing/contribution/cosine/clipのCSV、policy条件JSON、live log snapshotのprefix SHA256を併記した。raw routing.jsonlとstep_*_vectors.ptが元データ。初回runは監査wrapperのLoss.parameters呼出しで停止。v2はCPU/GPU cache混在のため予備扱い。最終数値は同一CPU teacherで完走したrunのみ。既存train.py/modelsのSHA256は開始/終了で一致。既存学習processは継続したまま。

コード根拠: models/network.py:954（方策）、:1197（global係数/比較条件）、:1360（Plan回帰）、:1622（strict pair）、:2010（Candidate replay）、:2941（source分類）。models/modules/structure_actuator.py:367（初期化）、:4377（Candidate入力detach）、:7422（選択Q集約）、:7458（Planへの入力）。models/utils/training/train_runtime.py:5180/:5275（保存/復元）、:5424（decision clip）。train.py:2622（保存）、:4350（復元）、:4411（global clip）、:4420（optimizer）。
