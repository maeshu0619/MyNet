"""Summarize actual gradient routing measurements and a bounded live-log snapshot."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re


def snapshot_log(path):
    size = path.stat().st_size
    digest = hashlib.sha256()
    counts, groups = Counter(), Counter()
    first, last, probe_last = [], {}, []
    by_group = {}
    used = 0
    with path.open('rb') as f:
        while used < size:
            raw = f.readline()
            if not raw or used + len(raw) > size:
                break
            used += len(raw); digest.update(raw)
            line = raw.decode('utf-8', errors='replace')
            if line.startswith('Den6SourceGradProbe:'):
                probe_last.append(line.strip())
                probe_last = probe_last[-6:]
            if not line.startswith('Den6OnlineAudit:'):
                continue
            group = re.search(r'factorized_group=([^, ]+)', line).group(1)
            groups[group] += 1
            row = by_group.setdefault(group, Counter())
            counts['steps'] += 1; row['steps'] += 1
            flags = {'anchor_reused': 'matched_anchor_reused=True' in line,
                     'group_pair_available': 'actual_group_pair=(available=True' in line,
                     'global_credit_available': 'policy_global_available=True' in line}
            for key, value in flags.items():
                counts[key] += value; row[key] += value
            for key in ['actual_set_credit', 'actual_actor_replay', 'actual_candidate_critic_replay']:
                part = line.split(key+'=(', 1)[1] if key+'=(' in line else ''
                if key == 'actual_actor_replay':
                    part = part.split('actual_candidate_critic_replay=', 1)[0]
                else:
                    part = part.split(')', 1)[0]
                last[key] = part.strip().rstrip(',')[:1200]
                match = re.search(r'\bpairs=(\d+)', part)
                if match:
                    counts[key+'_pairs_nonzero'] += int(match.group(1)) > 0
                    row[key+'_pairs_nonzero'] += int(match.group(1)) > 0
                if key == 'actual_candidate_critic_replay':
                    match = re.search(r'weighted=([^, )]+)', part)
                    if match:
                        counts['candidate_critic_direct_nonzero'] += float(match.group(1)) != 0
                        row['candidate_critic_direct_nonzero'] += float(match.group(1)) != 0
            if len(first) < 6:
                first.append(dict(group=group, **flags))
    return dict(path=str(path.resolve()), prefix_bytes=used,
                prefix_sha256=digest.hexdigest(), counts=counts, groups=groups,
                by_group=by_group, first_six=first, last_fields=last,
                last_source_grad_probes=probe_last,
                caveat='Counters/logged flags, not a fresh gradient audit of the live model.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--live-log', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    run = args.run.resolve()
    rows = [json.loads(x) for x in (run/'routing.jsonl').open()]
    assert rows[-2]['kind']=='completed' and rows[-1]==dict(kind='integrity',step=6,changed=[])
    updates = [r for r in rows if r['kind']=='optimizer_update']
    assert [r['step'] for r in updates]==list(range(1,7))

    def get(kind, step, source=None):
        return next(r for r in rows if r['kind']==kind and r['step']==step
                    and (source is None or r.get('source')==source))

    def csv_out(name, items):
        if not items: return
        with (out/name).open('x', newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0])); w.writeheader(); w.writerows(items)

    gradients=[]; stages=[]; contributions=[]; cosines=[]; clips=[]
    for r in rows:
        if r['kind']=='source_gradient':
            for g, v in r['groups'].items():
                gradients.append(dict(step=r['step'],source=r['source'],group=g,**v))
        if r['kind']=='optimizer_update':
            s=r['step']
            for g,v in r['gradient'].items():
                theta=r['theta'][g]['l2']; delta=r['update'][g]['l2']
                stages.append(dict(step=s,group=g,
                    compression_pathwise=get('source_gradient',s,'L_downstream')['groups'][g]['l2'],
                    geometry_pathwise=get('source_gradient',s,'weighted_geometry_primary')['groups'][g]['l2'],
                    joint_backward=get('routing',s)['before'][g]['l2'],
                    after_policy_routing=get('routing',s)['after'][g]['l2'],
                    after_decision_clip=get('balance',s)['after'][g]['l2'],
                    after_global_clip=get('clip',s)['after'][g]['l2'],
                    optimizer_gradient_l2=v['l2'],optimizer_gradient_rms=v['rms'],
                    actual_update_l2=delta,theta_before_l2=theta,
                    update_rate=delta/theta if theta else None,
                    small_denominator_warning=theta<1e-3))
            for x in r['conditional_adam_loo']:
                p=next(y for y in r['source_projection'] if y['group']==x['group'] and y['source']==x['source'])
                theta=r['theta'][x['group']]['l2']
                contributions.append(dict(step=s,**x,gradient_l2=p['l2'],gradient_rms=p['rms'],
                                          gradient_projection=p['gradient_projection'],
                                          theta_before_l2=theta,small_denominator_warning=theta<1e-3))
            cosines += [dict(step=s,**x) for x in r['gradient_cosines']]
        if r['kind']=='balance':
            d=r['debug']
            for group in ['where','amount','action']:
                pre=d.get('den6_online_'+group+'_grad_norm_before_balance',0.)
                post=d.get('den6_online_'+group+'_grad_norm_after_balance',0.)
                clips.append(dict(step=r['step'],group=group,pre=pre,post=post,
                                  scale=post/pre if pre else None,
                                  applied=d.get('den6_online_'+group+'_grad_balance_status')=='clipped'))
        if r['kind']=='clip':
            d=r['debug']; pre=d['train_grad_total_norm_before_clip']
            clips.append(dict(step=r['step'],group='global',pre=pre,
                              post=r['after']['ALL']['l2'],
                              scale=r['after']['ALL']['l2']/pre if pre else None,
                              applied=d['train_grad_clip_applied']))
    csv_out('source_gradients.csv',gradients); csv_out('routing_stages.csv',stages)
    csv_out('source_contributions_and_conditional_adam.csv',contributions)
    csv_out('gradient_cosines.csv',cosines); csv_out('clip_events.csv',clips)
    evidence=snapshot_log(args.live_log)
    (out/'live_log_snapshot.json').write_text(json.dumps(evidence,indent=2))
    conditions=[r for r in rows if r['kind']=='policy_conditions']
    mature=[]
    for line in evidence['last_source_grad_probes']:
        values=ast.literal_eval(line.split('norms=',1)[1])
        mature.append([re.search(r'step=(\d+)',line).group(1)]+[
            fmt_value for fmt_value in [values.get('actual_compression_policy_actor'),
                                        values.get('actual_compression_policy_candidate_critic'),
                                        values.get('actual_compression_policy_plan_critic')]])
    (out/'policy_conditions.json').write_text(json.dumps(conditions,indent=2))
    validation=dict(steps=6,main_optimizer_updates=6,
                    emulator_updates=sum(r['kind']=='emulator_update' for r in rows),
                    source_files_changed=rows[-1]['changed'],
                    max_adam_clone_error=max(r['max_abs_error'] for r in rows if r['kind']=='adam_clone_validation'),
                    max_routing_decomposition_relative_error=max(r['relative_error'] for r in rows if r['kind']=='decomposition_check'),
                    teacher_device=json.loads((run/'effective_args.json').read_text())['sparsepcgc_device'])
    assert validation['max_adam_clone_error']==0.
    assert validation['max_routing_decomposition_relative_error']<1e-5
    (out/'validation.json').write_text(json.dumps(validation,indent=2))

    def fmt(x): return '—' if x is None else '{:.6g}'.format(x)
    def table(header, data):
        return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+
                         ['| '+' | '.join(map(str,r))+' |' for r in data])
    def sg(step,source,group): return get('source_gradient',step,source)['groups'][group]['l2']
    focus=['head:den6_candidate_actor','head:den6_candidate_critic','head:den6_plan_critic',
           'AddWhere','PruneWhere','AdjustWhere','AddAmount','PruneAmount','AdjustAmount','AmountSelector','Gate']
    texts=['# SparsePCGC Actual creditとoptimizer更新の追加監査', '',
           '2026-10-20。既存train.py/models/args/codec/checkpoint/processは変更せず、'
           'experiments以下の独立wrapperでEpisode 1の6 Stepを実行した。'
           'seed=21、UVG/BlueBackpack、元の382 Episodeのscheduleを保持。'
           '本訓練の現在のcheckpointを測定したものではない。', '',
           '共有GPU高負荷で予備実行がauto→CPUへ退避し、GPUで保存済みのGT cacheとの混在に数bit差が出た。'
           '予備実行v2は最終計測に採用しない。最終runはteacherを明示CPUに固定し、GT Actual cacheを流用せず、'
           'GTと編集後の両方をCPU native codecで符号化した。Networkは通常のGPU実行。'
           '静的候補/構造/特徴cacheは既存生成物のコピーを独立出力先で使用。'
           'GPU teacherとbitwiseに同じ教師値とは主張しない。', '',
           '## 1. 勾配経路と初期ゼロの原因', '',
           '`Actual C,G → detached RD教師 → 方策/比較損失・Plan Critic → retained backwardで決定head勾配保存'
           ' → joint backward → 決定headをpolicy-onlyへ復元 → Where/Amount/Gate別clip → global clip → Adam`', '',
           'Actual codecを微分しないこと自体は正常。score-function/比較教師から更新する。'
           'detached Actual scalarを強引につなぎ直す修正は提案しない。', '',
           table(['Step','探索group','frame baseline既知','sequence/group baseline既知','global利用可能','厳密pair','Actor replay pair'],
                 [[r['step'],r['scalars'].get('factorized_group'),r['scalars'].get('baseline_seen'),
                   r['scalars'].get('sequence_baseline_seen'),r['debug']['global_actual_credit_available'],
                   r['debug']['actual_group_pair_available'],r['debug']['actual_actor_replay_pairs']] for r in conditions]), '',
           '初期4 StepはWhereAll/Fine/Amount/Gateの別groupで、新しいframeである。'
           'frame+groupとsequence+groupのbaselineが未初期化なのでpolicy_core_rawは0*log_probになる。'
           '厳密pairには同じframe/anchor/group、再利用可能anchor、比較plan、RD差>=0.05等が必要だが、'
           'この6 Stepは異なるframeなので直接比較教師も成立しない。'
           'Step 5/6では同じsequenceのWhereAll baselineが既知になり、弱いglobal補正だけが成立した。'
           'これは厳密な同一frame反実仮想ではない。', '',
           table(['Actual方策由来の勾配L2']+['S'+str(s) for s in range(1,7)],
                 [[g]+[fmt(sg(s,'policy:actual_compression_policy',g)) for s in range(1,7)] for g in focus]), '',
           '## 2. compression勾配はどこで失われるか', '',
           '以下はStep 2。joint列は全損失を合成した勾配であり、圧縮単独列と同じ量ではない。'
           '決定head復元後は圧縮pathwise寄与が厳密に0になる。'
           'そのためw_comだけを増やしても、そのSTE経路を通してWhere/Amountの最終更新は増えない。', '',
           table(['group','compression単独','joint backward','policy復元後','decision clip後','global clip後'],
                 [[g]+[fmt(next(x for x in stages if x['step']==2 and x['group']==g)[key]) for key in
                           ['compression_pathwise','joint_backward','after_policy_routing','after_decision_clip','after_global_clip']]
                  for g in focus]), '',
           '保存/復元はhardなargsort/int planと不整合なSTEの混入を防ぐ設計であり、'
           '無条件に復元を削除することも推奨しない。代わりに実codecの比較教師を成立させる。', '',
           '## 3. Candidate CriticとPlan Criticの違い', '',
           'Candidate Criticには初期6 Stepで専用Actual replay損失が成立していない。'
           'それでもStep 2以降に非ゼロになるのは、選択候補Qのmean/stdが18次元Plan featureに入り、'
           'Plan Critic損失から逆伝播するため。component:plan_critic_loss_weightedの'
           'Candidate Critic勾配とActual sourceの同勾配は今回全Stepで一致した。'
           '候補自身の独立したActual Δbits教師を学習できた証拠ではない。', '',
           '両Critic最終層はゼロ初期化。Step 1のPlan最終層weight L2=0なので、'
           'Plan headには更新があるが、入力側Candidate Criticへは勾配が届かない。'
           'Step 2直前のPlan最終層weight L2は約8e-4であり、さらに候補Qのmean集約が挟まる。'
           'Candidate Critic入力特徴もdetachされるので、その間接経路からWhereへは流れない。', '',
           'Plan Criticは毎StepのActual RDを直接回帰し、過去planもreplayする。'
           '係数は0.1、外側policy backward倍率10がこの回帰には残り、実効係数は1。'
           '一方global Actor補正の実効係数はpolicy_weight 0.1 × global_actual_credit_weight 0.05 = 0.005。'
           '専用Candidate replay係数0.4は比較pair成立時にだけ有効で、初期6 Stepでは不成立。'
           '選択頻度よりも教師の成立条件・回帰位置・初期化/集約の違いが支配的。', '',
           '## 4. RMS・実パラメータ更新・source寄与', '',
           'RMS=勾配L2/sqrt(group内全学習対象要素数)。更新率=||Δθ||/||θ_before||。'
           'ゼロ初期値では未定義、極小初期値では大きくなるため、操作Amountの更新率だけを発散とは解釈しない。', '',
           table(['Step 2 group','grad L2','grad RMS','Δθ L2','θ_before L2','更新率'],
                 [[g]+[fmt(next(x for x in stages if x['step']==2 and x['group']==g)[k]) for k in
                            ['optimizer_gradient_l2','optimizer_gradient_rms','actual_update_l2','theta_before_l2','update_rate']]
                  for g in focus]), '',
           'Candidate CriticはL2が2e-5でもAdam後に4e-4程度更新される。'
           'PlanとのL2差をそのまま学習速度差とは扱えない。問題は候補判別のActual教師が十分かという点。', '',
           'source別寄与は、実optimizerのAdam stateとParameterをcloneし、'
           '観測したrouting/clip係数を固定したまま1 sourceだけ引いた更新と比較した。'
           '元optimizer/stateには書き込まない。通常更新cloneは全6 Stepで実更新とbitwise一致（最大誤差0）。'
           'これは条件付きLOOであり、寄与の一意な分配率ではない。負方向の競合・momentum・非線形性により'
           '100%を超えることがあり、source間の値を合計してはいけない。'
           'clipを再計算する別policyの学習実験でも、codec品質の改善検証でもない。', '',
           table(['Step 2 group','compression STE除去','geometry pathwise除去','Actual RD除去','candidate-local Proxy除去','別geometry方策除去','その他policy除去'],
                 [[g]+[fmt(next(x for x in contributions if x['step']==2 and x['group']==g and x['source']==source)['loo_effect_over_actual'])
                           for source in ['compression_pathwise','geometry_pathwise','policy:actual_compression_policy','policy:proxy','policy:geometry','policy:unclassified']]
                  for g in focus]), '',
           '表の単位は||通常更新−source除去更新||/||実更新||。'
           '元のsource辞書はpolicy_loss全体を網羅しないため、その他policyを残差として明示した。'
           'この6 Stepでは厳密比較が未成立で、局所operation/Amount/Fine creditや追加Entropy等が残る。'
           '厳密pair成立時には辞書に分類されないcritic pair項もあり得るため、残差を一律にProxyとは呼ばない。'
           '共通ActorのProxy除去が1を超えるのは、少量Entropyや過去momentumとの向きが変わるためで、'
           'そのsourceが学習の172%を占めるという意味ではない。', '',
           table(['Step 5/6 group','S5 Actual条件付き更新変化率','S6 Actual条件付き更新変化率'],
                 [[g]+[fmt(next(x for x in contributions if x['step']==s and x['group']==g and x['source']=='policy:actual_compression_policy')['loo_effect_over_actual'])
                           for s in [5,6]] for g in ['head:den6_candidate_actor','AddWhere','PruneWhere','AdjustWhere']]), '',
           '## 5. 方向競合とclip', '',
           table(['Step 2 group','比較','cosine'],
                 [[x['group'],x['left']+' vs '+x['right'],fmt(x['cosine'])] for x in cosines
                  if x['step']==2 and x['group'] in ['head:den6_candidate_actor','PruneWhere','AmountSelector'] and x['cosine'] is not None]), '',
           'Prune Whereのpathwise圧縮/幾何は反対向きだが、その両方が後段で置き換えられる。'
           'このcosineだけを最終更新競合の原因とはしない。ActorではProxy/Entropyの競合を確認したが、'
           'Entropyはノルムが小さい。ゼロ勾配同士・一方ゼロのcosineは0と捏造せず未定義とした。', '',
           table(['clip対象','発生回数/6','備考'],
                 [[g,str(sum(x['applied'] for x in clips if x['group']==g))+'/6',
                   '上限1' if g!='global' else '上限10'] for g in ['where','amount','action','global']]), '',
           'Amountはcoarse selectorと3つのFine Amountを同じgroupでclipする。'
           'Step 2では約6.84→1、Step 3では約21.81→1。'
           'coarse geometry方策が大きいStepに、Fine Amountの別sourceまで同じ縮小係数を受ける。'
           'Whereのgroup clipには共通Candidate Actorとsubtree_move_source_headは含まれていないため、'
           'Where clip発生0はこれらの勾配の健全性を保証しない。global clipは今回未発生。', '',
           '## 6. 長時間ログでの成立頻度', '',
           '既存実行中logの読み取り専用prefix snapshot。ログのavailable/weighted/pairsを集計したもので、'
           'live Parameter.gradの再計測ではない。current checkpointの勾配方向・更新率を証明する値ではない。', '',
           table(['項目','Step数','全audit Step比'],
                 [[k,v,fmt(v/evidence['counts']['steps'])] for k,v in evidence['counts'].items()]), '',
           table(['既存GPU訓練Step','Actual source Actor','Actual source Candidate Critic','Actual source Plan Critic'],
                 [[r[0]]+[fmt(x) for x in r[1:]] for r in mature]), '',
           'これは既存ログのsource単独backwardであり更新直前ではない。'
           'Candidate Criticが初期の極小勾配のままではないことを確認した。'
           '大きくなった間接Plan勾配と、候補集合の直接比較教師は区別する。', '',
           'したがって初期3 Stepのゼロを全学習期間の切断とは判断できない。'
           '一方、専用Candidate教師や厳密pairは毎Step成立していない。'
           'anchor/replay capacity=512、pending=1024、max age=1600、min delta=0.05などが成立率に影響し得る。'
           '未成立を全てLRU eviction/差分marginのせいにしたとは主張しない。'
           'その内訳にはmatchedキー・frame再訪間隔・候補集合差・delta・ageを追加観測する必要がある。', '',
           '## 7. 推奨修正案（今回は未実装・未検証）', '',
           table(['問題箇所','原因・実測根拠','推奨修正案','期待効果','副作用/検証条件'],[
               ['Actual教師の成立','初期4 Stepは別frame/group。専用pairなし。S5/6で弱いsequence補正のみ成立',
                '最優先: anchor→variantを同一frameの短い2-view単位で処理。Whereは3操作共同WhereAll、Amount/Gate/Fineは他因子固定。codec予算内でbaseline結果を再利用',
                '因果的に比較できるActual ΔRDがActor/Candidate Critic/操作headへ届く',
                '同じframeの反復で系列多様性低下。per-Step encode数固定でも1系列当たりの計算増。anchor ageとpolicy versionを保存'],
               ['Candidate Criticの教師','専用Actual replayが初期6 Stepなし。観測勾配は全てPlan feature経由',
                '同一frame/anchorの交換candidate集合に対するgroup-level Q差/不確実性を直接学習。plan scalarを個々の点へコピーしない',
                '候補集合間のActual順位を学ぶ。間接的なplan難易度回帰への依存を減らす',
                '相互作用を単点へ誤配分すると逆効果。独立pairで符号一致率・calibrationを検証'],
               ['初期化による遅延','Plan/Candidate最終層zero。S1はCandidateへPlan勾配0',
                '直接教師を先に確保。必要ならPlan最終層zero vs std=1e-3の小初期化を独立ablation',
                '初回から入力側へ小さい勾配を返せる',
                'zero解除だけでは教師不足は直らない。初期Qの任意biasをpolicy authorityへ入れない'],
               ['WhereのActual寄与が小さい','S5/6で流れるがProxy優勢。global実効係数0.005',
                '厳密pair教師成立後にmatched replay係数0.4→0.8等の1x/2xを比較。跨frame global weight 0.05はまず据置。無条件10x以上は避ける',
                '信頼できるActual比較の更新への寄与を増やす',
                '跨frame global補正は高分散。圧縮/幾何の検証なしに最適値とは呼ばない'],
               ['Amountのgroup clip','6 Step中5 Stepでclip。coarse geometry方策とFineを同時縮小',
                'coarse/Fineを別観測・別clip候補にし、global安全clipは維持。別geometry方策0.55→0.275の0.5x ablationを候補。cp_lambda_geom=150は保持',
                'coarseの大勾配がFine Actual信号を巻き添え縮小するのを抑える',
                '幾何拘束を弱める危険。clip解除や倍率変更単独を解決としない。D1/Chamfer/Actualを同時評価'],
               ['圧縮STEを強めても更新されない','policy-only復元で決定headのpathwise寄与0',
                '復元を無条件撤去しない。整合するActual比較教師を増やす。必要なら整合性を検証済みの小さいSTE残差を別ablation',
                'hard planと無関係なproxy勾配の混入を防ぎながら学習する',
                'STE混合はbiased。圧縮重みだけ増やすと表示loss/他moduleのみ変化する可能性'],
               ['L2だけの判断','Candidate Critic L2極小でも実Adam更新は非ゼロ',
                'RMS・Δθ/θ・teacher成立率・source LOO・Actual符号一致率を継続記録。freeze/None/zeroを区別',
                '不要な増幅を避け、教師不足と更新不足を分離',
                '追加backward/LOOは重いので通常訓練へ毎Step追加しない。低頻度診断に限定'],
           ]), '',
           '倍率は次の独立比較実験の候補であり、最適値/品質維持を保証したものではない。'
           '今回、重み変更後の訓練やRD改善実験は行っていない。'
           '同じ初期state・frame・候補・Actual backendでcontrolと比較し、'
           'Actual bits、同じGTに対する幾何品質、操作率、教師成立率を検証してから採用する。', '',
           '## 検証と成果物', '',json.dumps(validation,ensure_ascii=False,indent=2), '',
           'source/routing/contribution/cosine/clipのCSV、policy条件JSON、live log snapshotのprefix SHA256を併記した。'
           'raw routing.jsonlとstep_*_vectors.ptが元データ。'
           '初回runは監査wrapperのLoss.parameters呼出しで停止。v2はCPU/GPU cache混在のため予備扱い。'
           '最終数値は同一CPU teacherで完走したrunのみ。'
           '既存train.py/modelsのSHA256は開始/終了で一致。既存学習processは継続したまま。', '',
           'コード根拠: models/network.py:954（方策）、:1197（global係数/比較条件）、:1360（Plan回帰）、'
           ':1622（strict pair）、:2010（Candidate replay）、:2941（source分類）。'
           'models/modules/structure_actuator.py:367（初期化）、:4377（Candidate入力detach）、'
           ':7422（選択Q集約）、:7458（Planへの入力）。'
           'models/utils/training/train_runtime.py:5180/:5275（保存/復元）、:5424（decision clip）。'
           'train.py:2622（保存）、:4350（復元）、:4411（global clip）、:4420（optimizer）。']
    (out/'report.md').write_text('\n'.join(texts)+'\n')
    print(json.dumps(validation,indent=2))
    print(json.dumps(evidence['counts'],indent=2))


if __name__=='__main__':
    main()
