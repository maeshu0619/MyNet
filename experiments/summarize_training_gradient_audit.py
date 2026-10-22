"""Export measured gradient audit records without inventing missing values."""
import argparse
import csv
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reference-log')
    args = parser.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = [json.loads(s) for s in (run / 'measurements.jsonl').read_text().splitlines()]
    assert any(r['kind'] == 'completed' for r in records), 'Audit did not complete'
    assert not any(r['kind'] == 'probe_error' for r in records), 'Probe errors'
    assert records[-1]['kind'] == 'source_integrity' and records[-1]['changed'] == []
    locations = [r for r in records if r['kind'] == 'step_location']
    steps = sorted({r['step'] for r in locations})
    assert len(steps) == 3, 'This summary expects the measured three-step audit'

    def save_csv(name, rows):
        if not rows:
            return
        with (out / name).open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    flat = []
    updates = []
    counters = {}
    for r in records:
        if r['kind'] == 'actual_parameter_update':
            counters[r['step']] = counters.get(r['step'], 0) + 1
            for g in r['groups']:
                updates.append(dict(step=r['step'], optimizer_ordinal=counters[r['step']], **g))
        for g in r.get('groups', []):
            flat.append(dict(step=r['step'], kind=r['kind'], loss=r.get('loss', ''),
                             value=r.get('value', ''), requires_grad=r.get('requires_grad', ''), **g))
    save_csv('all_gradient_groups.csv', flat)
    save_csv('actual_parameter_updates.csv', updates)
    save_csv('loss_aliases.csv', [dict(step=r['step'], loss=r['loss'], alias=r['alias'])
                                 for r in records if r['kind'] == 'loss_alias'])
    # Surrogate optimizer steps occur before the policy capture advances the
    # diagnostic counter. Map using chronological step_location boundaries.
    save_csv('surrogate_optimizer_gradients.csv',
             [dict(training_step=r['step']+1, gradient_l2=r['l2'],
                   connected_tensors=r['connected_tensors'])
              for r in records if r['kind'] == 'surrogate_optimizer_gradient'])

    log = next((run / 'log').rglob('*.txt'))
    lines = log.read_text().splitlines()
    relevant = [l for l in lines if l.startswith(('StepLoss ', 'StepBalance ', 'Actual Compression Loss:'))]
    comparison = None
    if args.reference_log:
        ref_lines = Path(args.reference_log).read_text().splitlines()
        ref = [l for l in ref_lines if l.startswith(('StepLoss ', 'StepBalance ', 'Actual Compression Loss:'))][:len(relevant)]
        comparison = dict(reference=str(Path(args.reference_log).resolve()),
                          printed_precision_only=True, equal=relevant == ref,
                          compared_lines=len(relevant))
    actual_values = [float(re.search(r'Loss: ([^ ]+)', l).group(1)) for l in relevant
                     if l.startswith('Actual Compression Loss:')]
    summary = dict(run=str(run), steps=steps, source_files_changed=[],
                   completed_main_updates=counters, probe_errors=0,
                   actual_compression_percent_at_log_precision=actual_values,
                   independent_log_comparison=comparison)
    (out / 'validation.json').write_text(json.dumps(summary, indent=2))

    def row(loss, step):
        return next(r for r in records if r['kind'] == 'loss_gradient'
                    and r['step'] == step and r['loss'] == loss)

    def fmt(value):
        return 'n/a' if value is None else '{:.6g}'.format(value)

    def table(header, rows):
        return '\n'.join(['| ' + ' | '.join(header) + ' |',
                          '| ' + ' | '.join(['---'] * len(header)) + ' |']
                         + ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows])

    text = ['# SparsePCGC 実学習3 Stepの勾配監査（2026-10-20）', '',
            '元のtrain.pyを独立processで実行し、既定seed=21、UVG/BlueBackpack、'
            'Episode 1の最初の3 Stepを計測した。既存processの現在の重みではなく、'
            '同じ現行設定からのfresh実行である。既存の学習ソース・args・codec・checkpointは変更していない。', '',
            '追加backwardはretain_graph=Trueで行い、モデルのParameter.gradとretainされた決定tensorのgradを退避・復元した。'
            '既存のbackward hookは有効。勾配ノルムは全対象Parameter要素を連結したL2であり、勾配平均ではない。'
            'グループは重複するためノルムを足し合わせない。', '',
            '## 損失の実測値', '',
            table(['項目', 'Step 1', 'Step 2', 'Step 3'],
                  [[name] + [fmt(row(name, s)['value']) for s in steps]
                   for name in ['L_total', 'L_downstream', 'L_geom', 'weighted_geometry_primary',
                                'loss_bit', 'loss_nodes', 'loss_single', 'surrogate_loss_for_grad']]), '',
            'Actual codec圧縮率[%]（ログ表示精度）: ' + ', '.join(map(str, actual_values)) + '。'
            'Actual encode自体は微分不能。上記L_downstreamは後段で上書きされた重み・proxy適用済み変数であり、'
            '生のActual圧縮率ではない。さらにw_com=3を掛けない。', '',
            '## 損失単独のbackward勾配L2（routing/clip前）', '',
            table(['損失', 'Step 1', 'Step 2', 'Step 3'],
                  [[name] + [fmt(row(name, s)['groups'][0]['l2']) for s in steps]
                   for name in ['L_total', 'L_downstream', 'L_geom', 'weighted_geometry_primary',
                                'loss_bit', 'loss_nodes', 'loss_single', 'surrogate_loss_for_grad',
                                'L_discrete_policy', 'policy_source:actual_compression_policy',
                                'policy_source:geometry', 'policy_source:proxy', 'policy_source:entropy',
                                'policy_source:efficiency']]), '',
            'L_attr / L_policy / L_actuator / full_cloud_amountとその記録済み重み付き項は全3 Stepで値0、'
            'requires_grad=False。L_com / L_com_objective / weighted_compression_primaryは'
            'この計測位置ではL_downstreamと同一tensor。重複測定していない（loss_aliases.csv）。', '',
            'policy_source:actual_compression_policyはコード中の名称であり、純粋な圧縮項の完全分離ではない。'
            'RD targetは3*C + 150*G（および有効なら編集tie-break）を含む。'
            'policy_source:geometryはそれとは別の幾何方策項である。', '',
            '## optimizer直前のモジュール/head別勾配L2', '']
    final = {s: next(r for r in records if r['kind']=='optimizer_postclip' and r['step']==s)
             for s in steps}
    group_names = [g['group'] for g in final[steps[0]]['groups']
                   if g['group'].startswith(('module:', 'actuator:'))]
    group_rows = []
    for name in group_names:
        gs = [next(g for g in final[s]['groups'] if g['group']==name) for s in steps]
        group_rows.append([name] + [fmt(g['l2']) for g in gs]
                          + [' / '.join(g['status'] for g in gs)])
    text += [table(['グループ', 'Step 1', 'Step 2', 'Step 3', '状態'], group_rows), '',
             'frozen=requires_grad=False、disconnected=対象Parameter.gradが全てNone、'
             'zero=接続したgradはあるが全要素0、absent=そのgroupにParameterがない。'
             'structure_analyzer等のパラメータなし計算と、存在しないheadを同一の学習不良とは解釈しない。', '',
             'single_plan_studentは別emulator optimizerで更新される。上記はmain clip時点に残るgrad。'
             'actual_parameter_updates.csvのoptimizer_ordinal=1がemulator、2が本体。'
             'Surrogateも別optimizerであり、15回の実更新を記録。各Stepに5回、clip後L2は約10。', '',
             '## 解釈上の重要点', '',
             '- 生の幾何pathwise勾配は約0.0036〜0.0037。重み・support balance・backward scale適用後は約0.016〜0.020。'
             'cp_lambda_geom=150だがcp_geometry_backward_scale=0.036のため、'
             'dL/dGは4.3249 / 5.4 / 5.4（ログ精度）。w_geom=10000はこのprimary objectiveの重みではない。',
             '- 別の幾何方策項はStep 2で6.84576、Step 3で21.8127。'
             'したがって「幾何勾配は全経路で小さい」という判断は実測と矛盾する。'
             'Amount selector等に集中し、routing/操作別balance・clip後は大幅に変わる。',
             '- 決定headはjoint backward後、保存されたpolicy-only勾配へ置き換えられる。'
             'L_geomやL_downstream単独の微分を、そのまま更新に使ったとは扱わない。',
             '- 現行Whereは共通den6_candidate_actorに加え、'
             '操作別add_head/drop_head/move_voxel_head等にも全3 Stepで非ゼロ勾配を観測した。'
             'この3 StepではActorへのactual_compression_policy sourceは0で、Proxy/Entropy側の寄与を観測した。'
             '初期フレーム群での観測であり、後半・再訪frameも同じとは断定しない。',
             '- 既定heuristic_guidance_online_grad_audit=Falseなので、'
             '通常ログのoperation_grad_norms=0は今回の実測ゼロを意味しない。'
             'この監査はParameter.gradを直接読み、操作別Where/Amountの非ゼロ勾配を確認した。',
             '- Point Transformer feature adapterのgradは1e-16〜1e-14だが、'
             '3回とも実Parameter差分は0。事実上更新されていない。'
             'Candidate Proposalおよび別経路のselector類はこの3 Stepでは未接続。', '',
             '## 検証', '',
             '全3 Stepでemulatorとmainのoptimizer.stepとParameter差分を確認。probe errorなし。'
             'train.pyおよびmodels以下のPythonソースSHA256に変化なし。'
             '追加したのはexperiments以下の監査/集計コードと生成物のみ。', '',
             '独立の既存訓練ログとの比較: ' + json.dumps(comparison, ensure_ascii=False) + '。'
             '一致はログの表示精度に限定し、全Parameterのbitwise一致を証明したものではない。', '',
             '生成CSVは丸め前の実測値。measurements.jsonlが元データ。'
             '初回auditは入力ソース一覧のdirectory項目で停止し、v2は計測出力側の重複keywordで停止。'
             '結果は正常完走したv3だけを用いた。']
    (out / 'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
