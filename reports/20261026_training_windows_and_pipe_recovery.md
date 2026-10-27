# 訓練入力固定・BrokenPipe調査と修正

2026-10-26。既存の未コミット変更を保持して作業した。

## 確認できたログと損失推移

`/data/maejima/log` 以下のEpisode CSVを更新日時で調べたところ、最新は
`20261023/MyNetwork_train/SparsePCGC/162249_*` だった。
ユーザーの指す「これより新しい実行」はこの環境では特定できておらず、別実行との比較は未完了。
`030549_*` はそれより古い実行であり、最新実行として扱わない。

以下は `162249_epi_metrics.csv` と `162249_checkpoint_metrics_epi.csv` の実測値。
負のActual圧縮率は改善を示す。

| Episode | 訓練Actual平均 [%] | 同じ5フレームの固定Validation Actual [%] |
|---:|---:|---:|
| 1 | −3.481886262 | −3.611088278 |
| 2 | −3.368408861 | −3.614787187 |
| 20 | −3.221552226 | −3.603909740 |
| 40 | −3.343516950 | −3.610595723 |
| 80 | −3.616229012 | −3.639949343 |
| 100 | −3.863132292 | −3.618241028 |
| 294 | −3.792040995 | 未測定：全5件が通信エラー |

比較対象の実行にも初期の悪化がある。Episode 1→20で訓練平均は
`+0.260334036` percentage points、固定Validationは `+0.007178539`
percentage pointsだった。
**訓練平均の変化全体を、モデル性能の悪化とみなす根拠はない。**

この実行の `TrainingWindow` は系列ごとに10個の元フレームを選び、
matched 2-viewにより20 Stepずつ処理していた。4系列合計40元フレーム・80 Step。
Episodeごとに別の窓を使い、10 Episodeごとに周回する。周回ごとに局所seedで再配置する。
さらに訓練には探索が入り、固定Validationでは探索が無効である。
入力構成、探索、学習更新をこのログだけで定量的に因果分離することはできない。
固定入力への変更は入力構成の交絡を取り除くもので、単調な損失低下を強制する処理ではない。

## BrokenPipeの原因と修正範囲

`162249.txt` ではEpisode 294の学習80/80更新が完了した後、固定Validationで
10回の `BrokenPipeError` が記録され、Episode 295の最初に訓練が停止した。
直前にホスト空きメモリ `4665.6 MiB < 8192 MiB` の待機がある。
worker stderrは初期化完了までの記録しかなく、終了時のPython tracebackはない。
カーネルログの参照権限もないため、**OOM kill、外部停止、native crashのどれかは断定できない。**

確認できた実装上の問題は、`_loaded=True` を維持した終了workerへ送信を続けること、
Pipe切断から復旧せず、Validationが同じ壊れた通信先へ要求を繰り返すことだった。

`models/utils/loss/actual_encoder.py` のSparsePCGCクラスだけに以下を追加した。

- 送信前にworkerの終了状態を検査する。
- Pipe切断・接続切断・EOF・worker終了に限り、同じ設定でworkerを再起動する。
- 同じ入力要求を再実行し、部分出力は破棄する。既定の復旧上限は1回。
- Actualが取得できなければエラーを送出する。Proxyや架空の教師で更新しない。
- 計算エラー、設定エラー、Timeoutを通信切断として一律に再試行しない。
- 終了コード、stderrパス、復旧回数を記録する。終了したpipeのfile descriptorも閉じる。

`train.py` は停止時の完全なtracebackもログへ保存するようにした。
この修正は外部のメモリ圧迫を解消するものではない。
メモリ不足が続く場合は既存の待機・停止条件を守り、安全に失敗を報告する。

## 毎Episode同じ100フレーム

新規 `models/utils/training/episode_frames.py` で一度だけmanifestを構成する。
`--train_fixed_episode_frames` の既定値を100とした。

- 訓練用領域の先頭から系列間で均等に選ぶ。4系列なら25ずつ、3系列なら34/33/33。
- ファイル番号順で固定し、Episodeやglobal RNGで並べ替えない。
- 元の `all_files` とValidation分割境界は維持する。
- 十分な訓練フレームがなければ、重複やValidation流用で補わずエラーとする。
- `--train_all_datasets true` の場合も選択対象の全系列を通した合計100フレーム。
- 特別な短時間診断のみ `--train_fixed_episode_frames 0` で従来の窓選択を利用できる。

実データで確認した8iの選択結果：

| 系列 | 個数 | 最初 | 最後 |
|---|---:|---|---|
| longdress | 25 | 1051 | 1075 |
| loot | 25 | 1000 | 1024 |
| redandblack | 25 | 1450 | 1474 |
| soldier | 25 | 0536 | 0560 |

382 Episodeすべてでパス一覧が一致することを確認した。ここでの確認はmanifestの検証であり、
382 Episodeを再訓練したという意味ではない。

**100フレームは元データの個数。既定のmatched 2-viewでは200 Step/Episodeとなる。**
Actual比較教師の経路を壊さないため2-viewを削除していない。
探索の全訓練長・前半25%というEpisode比率は、既存処理が新しいStep数から計算する。
同じEpisode数でも総更新回数と処理時間は旧80 Step/Episodeから増える。

新しい実行には `{time}_training_frames.json` と、系列ごとの個数・署名・view数・Step数を記録する。
Cacheのworking-set見積りも実際に選んだ固定フレームを使う。

## 図・検証

損失・Voxel操作割合は `*_epi_*` だけを生成する。
100 Step履歴の集計と100 Step図の生成を止めた。既存の図やログは削除していない。
固定Validationの検証図・診断CSVは保持する。

実施した検証：

1. 修正前の関連69テスト合格。
2. 修正後は入力固定、plot、OOM/通信復旧、勾配関連、既存Actual semantics、
   G-PCC/OctAttention分岐を含む184テスト合格。
3. 実8iの100フレームが全382 Episodeの選択で同一、Validation領域との重複なし。
4. CPUの実SparsePCGCで `longdress_vox10_1051.ply`（765,821点）を符号化。
   修正前、修正後、実験用workerの強制終了・復旧後で教師値は全て38,288 bit。
   `input_R0.bin` のSHA256は全て
   `900992ddbbed0a09e6ba74192e350a326de61f2669fa7a8316ae45bb948882e4`。
   これは既存教師APIのbit値の一致確認であり、物理streamのbit数定義を変更した実験ではない。
5. `train.py --help` 成功。確認中に見つかった既存のhelp内の未escape `%` 3箇所も修正。
6. 編集したPythonの構文検査とsession差分確認。

実験中に終了させたのは独立実験が起動したworkerだけ。
既存の実行中process、checkpoint、訓練ログ、Network、損失関数、Codecのソースは変更していない。
訓練入力・通信wrapper・ログ/plot・argsの対応設定は今回の変更対象。

## 残る確認と再現資料

新設定での長期訓練や、修正後の実Networkの数Step訓練は実施していない。
今回の実測は実Codecの同値性と故障復旧、入力・plotの回帰確認であり、
勾配改善や最終圧縮率−3.9%への到達を証明するものではない。
固定入力でも探索・学習更新による非単調な損失変化は残り得る。
入力集合と1 Episodeあたりの更新回数が変わるため、旧実行と最終性能が一致する保証もない。

成果物：`/data/maejima/log/independent_experiments/20261026_training_reliability_blh1sdft/`

- `git_status_before.txt`, `git_diff_before.patch`, `before/`, `session_changes.patch`
- `tests_before.txt`, `tests_after.txt`, `tests_final.txt`
- `real_8i_manifest.json`
- `native_probe_v3/measurements.json`, `success.json`, `native_probe_v3_stdout.txt`
- `train_help_fixed.txt`

独立再現コード：`experiments/training_reliability_probe.py`。
初回 `native_probe/` と `_v2/` は実験用snapshotのworkerパス解決で失敗したため、
成功結果に数えない。参照先を正しく揃えた `_v3/` のみを同値性・復旧の根拠とした。
