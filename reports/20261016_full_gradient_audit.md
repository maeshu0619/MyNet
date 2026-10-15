# 2026-10-16: 実訓練による全パラメータ勾配・更新監査

## 実行条件と範囲

- 変更前16 Step、変更後16 Step（それぞれ4 Episode × 4系列 × 先頭1フレーム）。同一seed・初期Surrogate・入力を使用。探索の減衰基準は382 Episodeのまま。
- 元環境 `/home/maejima/miniconda3/envs/myOA/bin/python` を使用。実際のSparsePCGC圧縮とoptimizer.stepを実行した。既存訓練プロセスは停止・再開していない。
- 学習対象の全184パラメータテンソルとSurrogateの10テンソルを、各optimizerの実行直前・直後で検査。勾配None/ノルム/有限性、更新差分、学習率、optimizer登録漏れを記録。
- Networkには別途274個の凍結テンソルがある。意図的に固定しているEncoder等を、非ゼロ勾配にする変更は行っていない。
- 実験用ログ・checkpoint・Surrogate保存先は `/tmp/mynet-grad-audit-sCOyoX/` に分離した。測定用wrapperは `audit_train.py`、全テンソルの生データは各runの `gradients.jsonl`。

## 発見・修正した遮断

`models/network.py` のActual候補比較経路が、操作名を `factorized_group.endswith("Where")` で抽出していた。`WhereAll` はこの条件に該当せず、空の操作名になる。

このため、3操作同時探索時には候補集合contrastとActual Actor/Proposal/Candidate-Critic replayへ候補が渡らなかった。合成planのPolicy Gradientやproxy教師は別経路で動くため、Actor等の勾配が存在するだけでは発見できない。変更前16 StepではProposalの4テンソルすべてが勾配・更新ゼロだった。

修正後はWhereAllを明示的に扱い、3操作の変更候補を1つのjoint replay観測にまとめる。Actualの合成planスカラーを、各操作が独立に改善したという教師へコピーしない。候補集合contrastもjoint score差へ1回だけ適用する。従来の単一操作探索は1操作のまま。別フレームのActualから候補教師を作ることは引き続き禁止。

## 勾配診断の修正

`model.zero_grad()` はretain_gradした非leaf決定テンソルのgradを消さない。そのため、source別診断・policy capture・joint backwardの勾配が加算され、操作別decision勾配ログが実更新と一致していなかった。

policy capture前に診断用勾配をクリアし、captureしたdecision勾配をjoint backward後に復元する。学習パラメータの勾配隔離方針は変えない。診断無効時には決定テンソルのretain自体を停止し、診断用保存の追加コストを通常訓練へ持ち込まない。

## 実測結果

以下は変更後16 Stepにおける、optimizerへ渡したパラメータ勾配のL2ノルム。範囲は非ゼロStepだけで集計。全体clip後の値であり、非leaf decisionの勾配とは区別している。

| 対象 | 非ゼロStep / 16 | 勾配L2の範囲 | 実更新Step / 16 |
|---|---:|---:|---:|
| Add Where | 16 | 1.41666e-6–0.250451 | 16 |
| Prune Where | 16 | 5.97487e-9–0.000363186 | 16 |
| Adjust Where | 16 | 4.03387e-6–0.311461 | 16 |
| Add Fine Amount | 6 | 7.85799e-5–0.978491 | 14 |
| Prune Fine Amount | 6 | 0.000330954–0.850389 | 14 |
| Adjust Fine Amount | 6 | 0.000103046–0.317430 | 14 |
| 共通Coarse Amount | 7 | 0.00568390–1.000000 | 14 |
| Gate | 16 | 0.0372406–1.000000 | 16 |
| Candidate Actor | 16 | 1.39094e-5–0.167824 | 16 |
| Candidate Proposal | 3 | 0.0330416–0.0622614 | 3 |
| Candidate Critic | 15 | 2.47110e-5–0.132651 | 15 |
| Proxy Critic | 16 | 0.105263–0.148254 | 16 |
| Plan Critic | 16 | 0.627373–2.218470 | 16 |

Proposalの具体的な変化:

| Global Step | 修正前 grad / update L2 | 修正後 grad / update L2 |
|---|---:|---:|
| 13 | 0 / 0 | 0.0622614 / 0.00149482 |
| 14 | 0 / 0 | 0.0422349 / 0.00106865 |
| 16 | 0 / 0 | 0.0330416 / 0.000652771 |

194テンソル中、累積で非ゼロ勾配・実更新があったのは変更前107、変更後111。増えた4個はProposal。全対象でNaN/Infは0、学習対象のoptimizer登録漏れも0。

残り83テンソルは、現行modeで使わない別modeのNetwork-only/旧Proposal/旧Full-cloud Amount selector、shadow emulatorで使わないAmount/share/gate/interaction等のhead、現行の独立Fineに置換済みの共有Amount residual、非学習confidenceへ置換済みのutility gate。これらを強制接続すると別設計になるため変更していない。

Fine/Coarseの勾配が毎Step非ゼロでないこと自体は遮断ではない。探索group、初見baseline、圧縮guard、Actual比較marginに依存する。Adamのmomentumによって、現在勾配がゼロのStepにも更新が起き得るため、勾配Step数と更新Step数を別々に集計した。初期化時に出力重みをゼロにするheadの前段が初回だけゼロになることも確認対象に含めた。

## 動作維持と限界

- 前後32 Stepすべてでedited Actual encodeは1/Step、候補別Actual encodeは0/Step。探索pool・操作上限・Amount bin・損失重み・減衰期間は変更していない。
- 前後の最初の2 Episodeは、平均損失・Actual・幾何がCSV桁精度で一致。その後は正しく復活したActual候補学習によって経路が変わる。
- Episode 4: Actualは旧-3.595980287、新-3.589015543。幾何は旧0.03296778537、新0.03296754882。この短期結果を圧縮精度改善や長期収束改善の証明とはしない。
- 124テスト成功。joint WhereAll replayのActor/Proposal/Critic勾配、別フレーム教師の禁止、診断の重複加算防止、既存Amount/Fine/安全性の回帰を含む。
- 最終コードで診断無効の通常設定も追加4 Step実行して完走。初回EpisodeのLoss=-6.023672462、Actual=-3.728889942、幾何=0.03441948071は診断有効runとCSV桁精度で一致した。ただしcheckpointの503テンソル中82個に差があり、最大絶対差は0.00141196（shadow studentのlocal trunk重み）。決定論を強制していないGPU実行であり、診断有無だけがこの差の原因と断定しない。パラメータ更新のbitwise一致まで確認済みとはしない。
- 今回の16 Stepで通った現在の有効経路について、意図しない継続遮断を修正し、実更新まで確認した。全データ・全探索状態・訓練後半での勾配や幾何改善まで確認済みとは主張しない。

## 生ログ

- 変更前: `/tmp/mynet-grad-audit-sCOyoX/before/log/20261016/MyNetwork_train/SparsePCGC/004814.txt`
- 変更後: `/tmp/mynet-grad-audit-sCOyoX/after/log/20261016/MyNetwork_train/SparsePCGC/005047.txt`
- 診断無効の通常設定: `/tmp/mynet-grad-audit-sCOyoX/normal/log/20261016/MyNetwork_train/SparsePCGC/005456.txt`
- 全勾配・更新量: `/tmp/mynet-grad-audit-sCOyoX/{before,after}/gradients.jsonl`
- テスト結果: `/tmp/mynet-grad-audit-sCOyoX/final-tests.txt`
