# BD-rateが一瞬で終了した件：実ログ監査と修正

2026-10-19（JST）。今回の監査対象：
`/data/maejima/log/20261019/MyNetwork_test_BDrate/SparsePCGC/020659_2acdf150/`。

## 結論：この実行では評価されていない

元の`run.log`には`FileNotFoundError: /path/to/compatible_model.pth`が記録されていた。
`rd_points.jsonl`は0 bytes、RD測定0件、model inference／codec／PSNRの生成物はない。
元の`summary.json`もfatal errorと不完全な曲線を記録しており、有効なBD-rateはない。

前に提示したコマンドの`/path/to/...`は置き換え用の例だったが、そのまま実行された。
併せて、従来の評価器が失敗理由をファイル内だけに残して最後に`Excel: ...`と表示したため、
評価完了に見えた。これは表示・事前検査の不備であり、モデル推論が実行された証拠ではない。
元の実行ログ・Excel・checkpointは変更していない。

## 修正内容

- `utils/bdrate/preflight.py`追加：指定したGT、法線、MINE checkpoint、選択codecのbinary／重み／Pythonの存在・実行権限を一括検査。代替モデルや別datasetへ勝手に切り替えない。
- `utils/bdrate/progress.py`追加：model inference、Actual encode/decode、MPEG PSNRの開始／終了／失敗と所要時間を表示。実測件数・実推論回数・BD-rateの有効性から終了statusを決める。
- `utils/bdrate/runner.py`変更：事前検査エラーと実行時エラーを端末にも表示。`run_status.json`を追加。0件の実行を成功扱いしない。空のRD図も生成しない。
- `utils/bdrate/io.py`変更：長いsubprocessでは30秒ごとに経過時間・ログ場所を表示。worker失敗時はstderr末尾を呼び出し元へ提示。timeout時に終了するのは当該評価器自身が起動した子だけ。
- `utils/bdrate/report.py`変更：Excel先頭に`Run status` sheetを追加。Excel保存と評価成功を区別する。
- `tests/test_bdrate_evaluation.py`：欠損パス、0件実行、stage counter、worker例外／timeout、CLI相当の失敗表示の回帰テスト追加。

数式、RD設定、Network、学習用args、既存train/test、codecソース、重みは変更していない。

## 修正後の実確認

| 実験 | MINE推論 | Actual圧縮・復号 | PSNR測定 | 経過時間 | 最終結果 |
|---|---:|---:|---:|---:|---|
| ユーザーと同じ仮パス | 0 | 0 | 0 | 0.25秒 | 欠損3ファイルを表示、exit 2 |
| SparsePCGC＋identity control、576点 | 0 | 12 | 12 | 28.39秒 | 測定完了、非単調曲線のためBD-rate未算出・exit 2 |
| 実Network＋SparsePCGC、576点 | 1 | 12 | 12 | 30.05秒 | 推論・測定完了、非単調曲線のためBD-rate未算出・exit 2 |
| Draco＋identity control、576点 | 0 | 12 | 12 | 1.93秒 | D1/D2 BD-rateとも0%、control成功・exit 0 |

「実Network」は動作確認用の互換G-PCC checkpoint
`/data/maejima/log/20261017/MyNetwork_train/pretrained/192050_G-PCC/best_loss_joint.pth`。
`--network-codec G-PCC --codec SparsePCGC`として実行した。これはユーザーが意図する
SparsePCGC向けモデルの実データ性能評価とは異なる。入力は解析法線付き合成576点であり、
学習効果・汎化・修論に用いる改善率の検証ではない。

model worker全体は約2.10秒、SparsePCGCの各圧縮・復号は約2.1～2.4秒、
各MPEG metricは約0.035～0.039秒だった。実データの所要時間をこの小規模検証から推定しない。

元のSparsePCGC自己比較実験と今回の6設定を比較し、実bitstream／side streamのhash、
bits、D1、D2が全設定で一致した。進捗表示追加による評価値の変化は確認されなかった。
36評価テスト＋既存codec routeの9テストに合格。

成果物：

- `experiments/results/20261019_bdrate_failure_visibility_v1/`
- `experiments/results/20261019_bdrate_progress_sparse_v1/`
- `experiments/results/20261019_bdrate_progress_model_v1/`
- `experiments/results/20261019_bdrate_progress_success_v1/`

## 次に本評価を行うために必要な指定

実在する**互換性のあるMINE checkpoint、評価GT、対応GT法線PLY**が必要。
`20260801/072958`の旧checkpointには前回確認した現在のNetworkとの互換性問題もあるため、
パスが実在するだけで正しいモデル評価になるとは扱わない。
法線がなければD2を勝手に推定せず、意図的なD1-only評価を指定するか、正式な法線を用意する。

モデルは各点群で1回推論し、固定したMINE出力を6つのcodec rate設定で評価する。
そのため6点評価は**推論6回**ではなく、通常**推論1回＋圧縮・復号12回＋metric12回**。
処理時間だけで成功を判断せず、`run_status.json`の件数と有効なBD-rateを確認する。
