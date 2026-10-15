# 2026-10-16: Episode変動とAmount学習の監査

対象の完了ログ: `/data/maejima/log/20261014/MyNetwork_train/SparsePCGC/104216.txt`（382 Episode）。現在のコードとCSVから再集計した。既存の `train_runtime.py` の最適化とそのテストは保持した。

## Episodeごとの上下

Episode 250–382でのActual圧縮値の標準偏差:

| 系列 | 標準偏差 |
|---|---:|
| 訓練（毎Episode異なる40フレーム） | 0.0443728 |
| 固定検証（同じ5フレーム、集合signature不変） | 0.00137460 |

固定検証は約32倍安定している。収束後も異なるフレームの損失平均は同じ値にはならない。Episode 40は探索終了でもなく、既存設定の探索終了はEpisode 95.5。探索終了後にも残る主な訓練曲線の変動は入力集合の違いで説明される。生の訓練値を平滑化・差し替えする変更は行っていない。既存の `104216_fixed_validation_rd.png` と `104216_learning_evidence.png` で固定集合を確認できる。

## Amountが減らない原因

1. E41–95の2200 StepでAmount探索は9 Step、WhereAllは1957 Step。UCBのcoverage項は累積countが増えると縮小するため、有限のuncertainty/evidenceボーナスでも長期に他群を占有できる。連続回数制限だけではAmountの再訪を保証しない。
2. E41以降は全Stepのcoarse Amountが0.0025。既存の最小positive binは0.0015であり、40/40/20配分なら0.06/0.06/0.03%に対応する。候補は存在するが選択できていない。
3. 中心logitを±0.35へ制限したうえで低confidenceでpriorと混合。E250以降のconfidenceは0.05–0.122043、平均0.069498。この範囲ではNetworkの最大順位差でもpriorの最も近い別binとの差を逆転できない。勾配が存在してもhard argmaxが変わらない。
4. E250以降のcoarse selectorパラメータ勾配L2中央値は2.489645e-8、Fine head合算は2.075345e-8。tanhが飽和してlogit勾配が失われる問題もある。古い操作別内訳欄には未計測を0表示するものがあるため、古い内訳の0を操作別勾配消失の証拠にはしていない。
5. Fineの新planはNormal log_probとsigma倍のbackward、anchorは定数項なしの二次式とsigma補正なし。比較する2つの確率と勾配スケールが異なっていた。
6. WhereAllのanchor identityに変更対象の候補IDまで含まれていた。候補を変えるほど別条件として扱われ、joint WhereのActual evidenceが得られなかった。

## 最終変更

- 各探索groupを最低10%再訪するcoverage floorを追加（初回の全群観測後、新規群選択時に適用）。係数は `--heuristic_guidance_online_factorized_min_share`。
- coarse Amountのforward authority下限を0.75に設定。既存の有界logitでも全6 binに到達可能。`--heuristic_guidance_online_amount_authority_floor 0` で旧confidence混合へ戻せる。Fine/Gateのforward confidenceと既存の量・安全上限は維持。
- coarse logitの有界化をtanhからsoftsignへ変更し、大きな入力でも勾配を保持。
- 新planとFine anchorで同じNormal density、同じsigma、同じsigma倍のbackwardを使用。
- WhereAllは3操作の要求数・pool・順序を固定条件とし、候補IDは変更対象としてhashから除外。要求数をanchorに保存し再利用する。
- 新設定はargs.pyに定義し起動ログへ表示。

損失式・重み・既存Amount bin集合・探索減衰期間・候補pool・3操作同時の配分・1 Step 1 Actual encodeは維持。0.06/0.06/0.03%を固定目標にしたり、直接その比率へ減衰させる処理は追加していない。

## 検証

122テスト成功。追加テストは、低confidenceでの旧bin到達不能、全bin到達可能性、大logitの非ゼロ勾配、小Amountを好む比較信号からの学習移動、Fine確率・勾配の一致、WhereAll比較hashの正しい不変条件、2200回のadversarial scheduler再訪を検査する。構文とdiff whitespaceチェックも成功。

同じseed・入力・初期Surrogateで、編集前後それぞれ2 Episode、8 Actual Step実行。入力は4系列それぞれの先頭1フレームを2回反復。探索減衰基準は382 Episodeのまま。各ランのログ・checkpoint・Surrogate保存先は `/tmp/mynet-amount-z2m5o1/` 内で分離。

| 項目 | 編集前 | 最終版 |
|---|---:|---:|
| Actual edited encode / Step | 1 | 1 |
| Candidate Actual encode / Step | 0 | 0 |
| 初回を除く平均Step時間 | 4.7536 s | 4.7114 s |
| 学習プロセスGPU peak allocated | 11486.438 MiB | 11487.743 MiB |
| Fine比較StepのAdd Amount decision勾配 | 26.5103 | 1.04847 |
| 同StepのAmount合算・正規化前パラメータ勾配 | 2539.57 | 114.827 |

Fineの比較計算修正で過大勾配は減ったが、既存の勾配正規化は引き続き適用される。すべてのStepでAmountが非ゼロであることを要求してはいない（初見フレームや他groupではゼロになり得る）。

探索中は行動とgroup順序が変わるため、損失値の完全一致は期待しない。例えば最終版の訓練Episode 2の幾何損失0.0407848は旧版0.0335313より高い。この結果を隠して非劣化とは判定せず、追加の探索なし固定検証を実施した。途中で試したFine forward authority拡大は最終版から除外した。

## 探索なし固定検証

前後8 Step学習後のモデルを、それぞれ旧/新の実行経路で評価。訓練とは重ならないlongdressの1052–1055の4フレームを使用。両方のsignatureは `493782b0d3b441e19ed031611ba09020678b49f524879125d236d446c5fadb66`。

| 指標 | 編集前 | 最終版 |
|---|---:|---:|
| Actual圧縮 | -3.614006917 | -3.619085121 |
| 幾何損失 | 0.0324336132 | 0.0324467551 |

圧縮は0.0050782 percentage point改善、幾何は約0.04052%増。完全一致・幾何改善とは報告しないが、この短期の固定検証ではほぼ同水準。これは1系列の4検証フレームに限る。全系列や382 Episode後の品質維持、操作率0.06/0.06/0.03%への到達は未実証。

## 再確認用ファイル

- 編集前訓練: `/tmp/mynet-amount-z2m5o1/before/log/20261016/MyNetwork_train/SparsePCGC/001602.txt`
- 最終版訓練: `/tmp/mynet-amount-z2m5o1/final/log/20261016/MyNetwork_train/SparsePCGC/002335.txt`
- 固定検証スクリプト: `/tmp/mynet-amount-z2m5o1/evaluate_fixed.py`
- 前後checkpoint: `/tmp/mynet-amount-z2m5o1/{before,final}/checkpoints/1.pth`

検証時に他用途GPUプロセスが存在したため、Step時間の差から恒常的な高速化率は主張しない。訓練の追加Actual encodeや大きなメモリ増加は観測されなかった。
