# データセットごとに固定100 Step

2026-10-27。前日の「100フレームだが200 Step」「全データセット合計100フレーム」を修正した。

- 単一データセットの1 Episodeは100フレーム・100 Step。
- 8iの4系列は各25フレーム・25 Step、3系列なら34/33/33。
- 各Episodeで同じファイル一覧・同じ順番を使う。
- 固定入力モードでは `sparsepcgc_matched_frame_repeats=2` が残っていても実効repeatは1。
- 全データセットモードでは各データセットに100 Stepを割り当てる。
  例：8iとMVUBを同じ外側Episodeで巡回する既存構造では100＋100＝200 Step。
  他データセットのフレームで不足を補わず、不足したデータセット名を付けてエラーにする。
- 元の訓練/Validation分割、Codec、損失、Networkは変更しない。
- Actual比較のframe/anchorメモリは維持する。隣接2-viewの代わりに、同じframeの次回訪問時に
  有効期限・pool整合性を満たす既存anchorを比較できる経路が残る。
  比較教師の成立頻度や最終圧縮性能を実訓練で検証したという意味ではない。

変更：`episode_frames.py`（データセット別配分）、`sparsepcgc_actual_credit.py`
（固定モードではrepeat=1）、`train.py`（manifestログ）、`args.py`（設定説明）、回帰テスト。

関連157テスト合格。100/300 Step後に同一frameの未完了Actual比較groupへ戻れることも
既存のgroup選択関数で確認した。これは教師値・勾配の実測試験ではない。

実8iのPLYを既存Datasetとbatch_size=1・shuffle=FalseのDataLoaderで読み出し、
longdress/loot/redandblack/soldierが各25回、合計100回で完了した。
全382 Episode分の選択でも合計100 Stepを確認した。
実8i/MVUB/UVGのファイル一覧をまとめた場合も、それぞれ100 Step（合計300 Step）と確認した。
後者はmanifest件数の確認であり、MVUB/UVG全フレームの読出しや訓練はしていない。

実測資料と変更前snapshot：
`/data/maejima/log/independent_experiments/20261027_episode_100steps_bjsp6e0e/`

実点群の読出し検証は `real_8i_dataloader.json`、テスト結果は `tests_final.txt`。
Networkを用いた再訓練ではなく、訓練と同じDataset/DataLoader経路の件数確認である。
