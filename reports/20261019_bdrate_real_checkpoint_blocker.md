# 仮パスではなく実ファイルを用いた再確認

2026-10-19。今回は構文テストだけでなく、指定日時の実checkpointを実際にロードした。

## 実行に使用したファイル

- checkpoint：`/data/maejima/log/20260801/MyNetwork_train/pretrained/072958_SparsePCGC/best_loss_joint.pth`
- 入力候補：`/data/maejima/data/ground/8i/loot/loot_vox10_1150.ply`（794,717点）
- 診断request／stdout／stderr／returncode：`experiments/results/20261019_bdrate_user_model_check_v1/`

このフレームはpath／model loaderの診断用であり、訓練splitとの独立性を確認した修論用評価データとは扱っていない。確認した元PLYには法線propertyがない。法線を勝手に生成したD2結果も作っていない。

## 確認結果

1. ユーザーのコマンドに残る`/path/to/...`は引き続き存在しない。現行preflightは正しく停止している。これはSyntaxErrorではない。
2. 実checkpointへ置き換えた独立worker試験も **1.87秒、returncode 1** で停止した。モデルの互換性検査が原因。
3. 現行model側の学習可能parameterが28件checkpointに存在しない。PointTransformer adapter、追加candidate actor／critic等。
4. Amount selectorの4 tensorでshape不一致。checkpointの`amount_bin_logits.weight`は`(8,64)`。
5. checkpoint内の`single_plan_distillation_updates=2500`、`single_plan_actual_training_updates=0`。蒸留学習済みであり「全く未学習」とは言わないが、現行testが要求するActual calibration条件を満たさない。
6. 対応学習ログ`/data/maejima/log/20260801/MyNetwork_train/SparsePCGC/072958.txt`の56行には、`actual_source=heuristic_teacher_plan, network_only_performance=0, deployment_checkpoint_eligible=0`がある。
7. 同runの`best.pth`、`best_actual_delta_joint.pth`、`best_actual_delta_improved.pth`も読取確認し、Student Actual更新0、feature adapterなし、同じAmount shapeだった。別のbestへ黙って切り替えても解消したとは扱わない。

現行checkpoint checkerだけを無効にしてランダムparameter・shape補完を許すこと、Student Actual更新0を隠すこと、別の学習済みモデルへ黙って変更することは行っていない。既存の訓練コード・args・Network・codec・checkpointは編集していない。

## 構文・既存評価器の回帰確認

評価用36テストは全合格。ただし、テスト合格をこのcheckpointの本評価成功と読み替えない。
現在のファイル不存在検出は不具合ではなく、誤った評価結果の生成を防ぐために必要である。

## 残る判断

このcheckpointを評価対象とするなら、保存当時の推論経路・設定と対応sourceを特定する必要がある。git内には近い日時のsourceもあるが、保存したcheckpointとの対応を確定したわけではない。
また、teacher-guided training時の実行planと、現行testのteacher-free Studentを同じ提案手法として無断で置き換えない。

選択肢は、(a) 072958モデルの当時の推論条件を独立コードで復元する、または(b) 現行の推論条件に対応する学習済みcheckpointを指定する、のいずれか。対象を確定せず「構文を通す」だけの変更では、研究上正しいBD-rateにならない。

**この指定モデルについて、推論・BD-rateまで成功したとは報告しない。**
