# Trueルートのden6非依存性と初回Step時間の再監査

2026-10-31。開始時HEAD `9f70923`、Git差分は空。既存の訓練PID 118732、そのteacher PID 119268、checkpoint、通常ログ、codecソースは変更していない。追加最適化は次回Python起動から有効であり、実行中processへのhot patchではない。

## 1. den6のAmount・Whereへ固定されているか

**現在の有効なTrueルートは、den6の事前探索profile・候補cache・完成planを使用しない。** ただし独自の合法候補poolは存在する。任意の全座標を無制限に直接操作する方式とは異なる。

| 段階 | 現在の実処理 |
|---|---|
| 設定 | `configure_autonomous_search`がdataset別Amount初期値を共通の最大比率由来へ置換。anchor期間=0、rank入力=False、旧Where prior係数=0、den6 teacher必須/自動構築=False |
| GT観測 | 最初のGT Actual encodeからNativeの整数CDFを取得。現在の点群に対するcodec観測であり、den6の編集結果ではない |
| 候補生成 | GTから作る空間候補＋符号化占有を変え得る候補。Native gainの正/0/負を残し、候補のrank/heuristic scoreは0 |
| Where | 学習済みpoint heads、Actor、Proposal、学習するNative-bit選好からlogitを作る。Native選好の初期値は0。探索時には共有係数のGaussianと候補別Gumbelを使う |
| Amount | 3操作それぞれが15個の正値bin＋No-opと連続Fineから選ぶ。操作別固定quotaやden6の総量・比率へ戻す処理は最終選択に使わない |
| 共同実行 | Add/Prune/Adjustを同時に選択し、衝突・重複・bbox・合法方向を検査して1つの最終planへまとめる |
| 教師 | 最終planのActual bitsとGeometryから学習。自己模倣も、同一GT/codecで実測した有利な完全共同planだけを使う |

既定の正値Amountは0.01 / 0.03 / 0.06 / 0.1 / 0.2 / 0.3 / 0.5 / 0.7 / 0.9 / 1 / 1.2 / 1.5 / 2 / 2.5 / 3 [%]。Fineによりbinそのものに固定されず、No-opもある。安全上限は各操作3%、移動は各軸1 voxel以内、候補数には上限がある。これらの制限は今回変更していない。

名称に `den6` が残る関数・cache directoryもあるが、Trueで共有するのは主に座標写像・入出力契約である。誤解を避けるため `heuristic_guidance.py` のmodule説明を修正した。

### 実行して確認した内容

`experiments/autonomous_independence_probe.py`で以下を例外化し、アクセスすれば停止する状態で実際の8 Step訓練を完走した。

- `_FAST_PROFILES.get` / 添字参照
- `_load_exact_single_plan_teacher`
- `_load_or_build_payload`
- `heuristic_guidance.resolve_profile`

呼出し回数はすべて0。事前解析のden6 poolを読み込まない状態でもforward/backward/Adam/Actual encodeが成立した。Native codec観測は許可しているため、「codecの知識も使わない白紙の探索」とは主張しない。

## 2. 現在実行中の訓練で分かったこと

監査対象ログは `/data/maejima/log/20261031/MyNetwork_train/SparsePCGC/080320.txt`。

初回100 Stepでは、Actual Cが負のplanを36回観測し、最小は−5.578718%。そのplanのAdd/Prune/Adjust比率は約0.0433 / 0.9833 / 0.1256 [%]であり、旧den6の固定比率ではない。全3操作No-opは100件中0件。Where探索試行は53件中14件、clean WhereでAmountを試す試行も47件中22件が負のCだった。

ただし、この最小値は探索を含む試行である。**これを学習済み方策の−4%安定性能とは扱わない。** 自動的に出力された探索なしValidationはEpisode 1がC=0/G=0、Episode 2がC=−1.062109%/G=0.000395263、Episode 3がC=−1.124318%、Episode 4がC=−1.128220%、Episode 5がC=−1.080594%だった。No-opから改善し始めているが、現在の100フレーム訓練で最終−4%収束を確認した段階ではない。5/382 Episodeの観測だけで長期停滞とも断定しない。したがって「den6非依存の行動空間と−4%を超える実行例」は確認できても、「現在の学習済み方策でden6水準を安定選択できた」という要件は未達である。

初回100フレームは各frameの初訪問なので、保存済み同一frameの自己模倣は0回。2巡目100 Stepでは32回使用された。これは過去に測定した同じframeの教師を再利用する設計による遅延であり、初回の勾配がすべて0という意味ではない。以前の4フレーム反復実験は同一frameの再訪が速く、通常の100フレーム訓練の学習速度や初回時間を十分には代表していなかった。

前回の独立128 Step×2 seedのValidation C=−4.032 / −3.771%は実測だが、4フレーム反復条件の証拠である。前回のValidationは4系列各1枚、現在の通常Validationはlongdress 1151–1155の5枚で、評価集合も異なる。今回のログの最終性能保証へ読み替えない。今回、学習則・損失重み・Amount探索scheduleを変更して学習曲線を補正することはしていない。

## 3. 10秒以上の主因と実装した最適化

実行中ログの初回100 Stepは平均14.7389秒、Actual encodeはおおむね0.4～0.6秒だった。独立したcold訓練profileでは次が大きかった。

- 候補のPython dict生成、重複排除、座標ごとの確率lookup。
- Add候補の26近傍をPython set/dictへ展開してから、GPU座標へ対応付ける処理。
- 初回GT encode、入力・特徴・写像cache構築。特に最初のStepはworker初期化も含む。

`build_heuristic_guidance`はforward内で2回呼ばれるが、同一写像のcache再利用もある。呼出し回数だけをもとに「すべて二重計算」とは判断していない。

| ファイル | 最適化内容 |
|---|---|
| `autonomous_context_arrays.py` | 座標を整数配列で保持し、重複排除・coarse-cell lookup・gain計算・候補混合を配列処理へ変更。選ばれた候補だけdict化 |
| `autonomous_context_proposals.py` | 上記へ接続。旧loop版をreferenceとして保持。候補ID・順序・RNG・cache形式を維持 |
| `autonomous_proposals.py` | Nativeルートの空間候補を配列のまま次段へ渡し、後で捨てる候補のdict・ID文字列の二重生成を省略。target単位の最初の候補を保持する規則も維持 |
| `autonomous_guidance_mapping.py` | Add全26近傍、Prune、Adjustの座標をまとめてGPU join。方向順序・合法性・無効候補・overflow等の契約を維持 |
| `heuristic_guidance.py` | Trueの単点候補だけを新mapperへ分岐。False、G-PCC/OctAttention、bundle等は旧mapperを使う |

候補数削減、Actual評価の間引き、精度変更、proxyへの置換、den6へのfallbackは行っていない。Network/loss/args/train.py/codec/checkpointの変更もない。

## 4. 修正前後の実測と一致確認

### 実サイズの独立比較

longdress 1051、765,821 voxels、各操作45,950候補、合計137,850候補。

| 処理 | 旧実装 | 新実装 | 検証 |
|---|---:|---:|---|
| Native候補生成、core cacheなし | 5.2170秒 | 2.3732秒 | 全候補の全field・順序・JSON SHA256一致 |
| Native候補生成、core cacheあり | 2.9284秒 | 2.3590秒 | 全候補の全field・順序・JSON SHA256一致 |
| 全点群への候補写像 | 4.0159秒 | 0.5374秒 | 47 Tensorとmetadataが完全一致、dtypeも一致 |

Native coreのcache hitでも、pool全体のRAM hitとは異なり、空間候補との混合・dict化が残る。両者の時間を混同しない。

### 同条件で8 Stepずつ実訓練

同じ初期重み・seed・4入力・データ順序・Amount bins・Native設定を使い、両方とも空の独立cacheから開始した。CPUはOMP/MKL各2 thread、RTX A6000。通常設定と同じNative CPU trim間隔16。source-gradient診断は速度計測では無効。通常訓練PIDは並行して動いており、共有機の負荷差がある。

| Step / 入力 | 修正前 | 修正後 |
|---|---:|---:|
| 1 / longdress、profile・初期化あり | 23.0016秒 | 15.4548秒 |
| 2 / loot、初回 | 14.3956秒 | 5.8315秒 |
| 3 / redandblack、初回 | 12.5799秒 | 7.1174秒 |
| 4 / soldier、初回 | 20.5076秒 | 9.8472秒 |
| 5–8 / 同じ4入力の再訪平均 | 3.1054秒 | 3.1597秒 |

主な改善は初回処理にある。再訪平均の差からwarm処理が高速化したとは主張しない。実行中の旧コード自体も、2巡目100 Stepは平均3.9152秒だった。これは元からあるcacheの効果であり、今回のコードをhot patchした結果ではない。

8 Stepすべてで最終編集集合hash・操作数・Actual C・Geometry G・nativeへの入力SHA・bits・点数・node統計が一致した。全137,850候補と全47写像Tensorも別途照合し、空間配列入力とlist入力の候補同値性も追加テストしたため、最終選択された候補だけが偶然同じだったという検証にはしていない。

### 浮動小数点・bitstreamの再現性の限界

勾配と全パラメータのbit完全一致は得られていない。最初の配列版の比較では最大勾配差は約1.55e−4、最大パラメータ差は約3.23e−4。変更していない修正前sourceを再実行した対照でも最大勾配差約2.19e−3、最大パラメータ差約2.04e−4があり、一部Stepの編集集合やCも変動した。最終版の比較値も `parity_final.json` に保存した。

Native bitstreamのSHAも実行間で異なった。同じ入力SHAでも修正前同士で生じる差があり、今回の最適化でcodecを変更したものではない。ただしその原因の完全分解、復号後PSNRの一致は今回未確認。今回一致したActual値・編集集合・整数写像と、すべての浮動小数点計算・stream bytesの完全再現性は区別する。

**1秒/Stepは未達。** warmでも約3秒、初回は約7～10秒以上の例が残る。Actual encodeに加え、全点群Network、backward、点群読込、pool復元が残る。1秒を実現した、全フレームの精度・長期収束を保証した、とは報告しない。

## 5. 保護と再現

成果物root: `/data/maejima/log/independent_experiments/20261031_true_speed_reaudit_pyymcgqh`。

- `before/manifest.json`: 開始時sourceとSHA。
- `before_train/` / `after_train/` / `after_train_v2/` / `control_train/`: 実訓練、パラメータ・勾配、Actual stream情報、profile。`after_train_v2`が最終版。
- `after_train_v2/den6_independence.json`: 禁止したden6参照の呼出し回数0。
- `real_array_benchmark/comparison.json`: 全候補一致。
- `real_guidance_benchmark/comparison.json`: 全写像一致。
- `parity_round1.json` / `parity_final.json` / `control_parity.json`: 学習値・更新の比較。
- `release_tests.txt`: 関連266テスト合格。Falseの既存args・9 pool・RNG一致も確認。

独立experimentは出力先が既存なら停止する。通常訓練の100 Step/EpisodeやDatasetのフレーム選択を変更していない。実行中processは旧コードのままであり、checkpointを勝手に切替・再起動していない。
