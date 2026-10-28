# True専用Amount候補の拡張とStep時間の検証

2026-10-28。`--test_route false` の既存計算を維持した実験ルートの変更。
実験結果は短時間の検証であり、長期収束精度の保証ではない。

## Amount候補

変更前のTrueは各操作について **0 / 0.1 / 0.9 / 1.0 / 1.5 / 3.0%** の6クラスだった。
変更後は各操作について次の16クラスを使う。

```text
0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5,
0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0 [%]
```

0はNo-op。Add / Prune / Adjustは同じStepで独立に選ぶため、coarseだけで16³=4,096通りの組合せを持つ。さらに既存Fineの連続倍率で調整するので、最終Amountが上記16個に限定されるわけではない。各操作の上限は3%。競合・bbox保護・合法候補数の制約は残る。

定義：`models/utils/config/args.py` の `--test_route_amount_bins`（既定15正値、No-opを自動追加）、`--test_route_max_operation_ratio`。値の単位は0–1の比率であり、例えば1%=0.01。

```bash
# 新規True学習。その他の既存起動オプションは必要に応じて指定する。
python train.py --test_route true --more_training false

# 旧6候補を明示する場合
python train.py --test_route true \
  --test_route_amount_bins 0.001,0.009,0.01,0.015,0.03
```

候補数を変えるとAmount selectorの出力層の形状も変わる。旧6候補checkpointを新16候補として黙って読み込まない。再開時は保存時と同じ候補数・値・上限を指定する。新checkpointには候補定義をruntime metadataとして保存し、同じ候補数でも値が違う場合を検出する。既存checkpointを補間・切捨て・上書きしていない。

## なぜ1つのplanでも重いか

**現在のTrueも合法候補poolを生成している。** den6の事前順位・固定Amountへの依存を外したことは、pool自体をなくしたこととは異なる。

実測例の最初のframeでは、入力765,821 voxelに対し、Add 44,603 / Prune 45,950 / Adjust 44,604候補を生成した。合計135,157候補。ネットワークは入力全体の特徴と候補scoreからWhereを選び、Amountと合わせた1つの共同planを作る。

```text
入力点群 → voxel/構造/Point Transformer特徴
→ 合法26近傍pool → Actor/ProposalによるWhere分布
→ 3操作の独立Amount・Whereをサンプル → 競合を解決 → 共同plan
→ 全点群の幾何損失 + Native SparsePCGC Actual encode
→ Actual/幾何方策勾配 → head別clip → optimizer.step()
```

4,096組合せをすべてencodeしていない。候補ごとのActual encodeも行っていない。それでも全点群のネットワーク計算、全点間の最近傍探索、選んだ点群全体の実符号化が必要。最初に触れるframeではGT側の符号化も必要で、その後は私有GT cacheを利用する。

開始時に確認した本訓練ログ `20261028/.../171853.txt` には約19.7秒/Stepと、編集後Actual encode約2.07秒が記録されていた。独立計測でも幾何損失が主要な負荷だった。

## 採用した最適化

1. Trueの全点Chamferを厳密KD-treeで候補抽出し、**元のCUDA Chamfer kernel**で最終距離とindexを決める方式へ変更した。近似NN、点数削減、loss倍率変更を行っていない。
2. 同じ位置に残った一意最近傍の点は、距離0・座標微分0を使う。非ゼロ距離・同距離候補は元CUDA kernelで処理し、元reference index順のtie規則を維持する。候補数が安全上限を超えるケース等は元計算へ戻す。
3. 幾何監査ログ用のhard / weighted / weight-only surrogateが同じ距離を繰り返し計算する重複を解消。既存の重みclamp、平均、STE式、fitの有効/無効条件を維持した。
4. Trueで損失に使われないlegacy proxy教師・proxy Critic forwardを省略。選択点ごとのTensorスカラ参照を操作単位のgatherにまとめた。Whereの合法条件とjoint方策likelihoodは維持した。

現行データの幾何経路は `ste_hard` / `disabled_without_addition_provenance`。高速化のために別のplane-fit付き経路へ切り替えたわけではない。

重複policy backwardを再利用する案も実装・計測したが、ピークGPUメモリが約1.1 GiB増えた版になったため採用しなかった。最終版は既存のbackward・勾配置換の回数/順序を保つ。

## 同条件での速度比較

同じ初期重み、seed22、4フレーム、Native CUDA teacher、6候補を明示し、変更前source snapshotと最終版をそれぞれ**8更新**した。100フレーム/Episodeという本訓練設定は変えず、独立診断runだけ4フレームに限定した。

各runの1–4更新はGT初回符号化・worker起動を含む。次表は5–8更新の平均。同期計測とcheckpoint記録を行う診断環境の値で、本訓練全frameの保証値ではない。source別勾配probeは両runとも無効化し、更新自体は実行した。

| 計測対象 | 変更前 | 最終版 |
|---|---:|---:|
| 表示Step時間 | 16.9203 s | 8.1728 s |
| Geometry | 8.5637 s | 1.7972 s |
| 共同plan選択 | 1.1804 s | 0.5779 s |
| Network forward全体 | 3.8855 s | 3.1522 s |
| Actual encode | 3.3097 s | 2.4186 s |
| TorchピークGPU allocated、全8更新の最大 | 11,776.35 MiB | 11,684.23 MiB |

表示Stepは約51.7%短縮。plan選択はNetwork forwardに含まれるので加算してはいけない。Actual encoderソースは変更しておらず、この列の差をcodec高速化の効果と断定しない。GPU共有状態や実行時間変動も含む。

1–4更新の表示時間は変更前16.65–28.44秒、最終版12.06–19.84秒。初回codec起動・GT encodeを含むcold時間と、再利用時を区別する。

Torch memoryは当該モデルprocessのallocator計測。別processのActual encoderや他アプリを含む`nvidia-smi`全体使用量とは異なる。

## 数値と更新の確認

初期state_dictは、両方に存在する同じNaN sentinel bufferを同一と扱う比較で一致した。8更新すべてでActual圧縮損失Cと幾何損失Gが丸め前の値で一致した。

| 更新 | 両runのActual C [%] | 両runのG |
|---|---:|---:|
| 1 | 4.9519433975 | 0.01929643005 |
| 2 | 16.5706977844 | 0.04085787013 |
| 6 | 18.1062068939 | 0.07295144349 |
| 8 | 6.1559886932 | 0.03509333357 |

7/8更新で編集後voxel hashも一致した。6更新目は操作数・C/Gが同じでもhashが異なった。勾配とパラメータはbitwise一致ではなく、全勾配の相対L2差は最大約1.32%、state要素の最大絶対差は約0.000410。演算順序・CUDAの非決定性等の寄与は完全分離していない。**完全に同じ学習軌跡や、長期精度不変とは断定しない。**

小規模CUDA単体試験では、ランダム座標、整数格子・重複・tie、全点保持について、元Chamferとの距離・index一致と、座標/weight勾配の許容誤差内一致を確認した。過大tie集合は元実装へfallbackする。既存回帰218件と高速化4件が合格した。

## 16候補版の追加確認

最終版で16候補を使って8更新を追加実行した。5–8更新の平均表示時間は **9.7257秒**（7.9501–12.7688秒）、Geometry平均 **1.7537秒**、Actual encode平均 **3.5249秒**、Torchピークallocated **11,682.76 MiB**。

実行された操作率の範囲はAdd **0.00836–2.99986%**、Prune **0–1.99630%**、Adjust **0.13837–2.64109%**。全8更新で勾配は有限。Actual Cは **+1.83408～+14.71525%**、Gは **0.00471917～0.05678663**。候補拡張は意図して行動分布を変える変更なので、6候補版と同じ損失値になることは要求しない。この初期探索結果を圧縮改善の達成とは扱わない。

以下は8更新中の最大値。勾配は実際のclip後、更新量はoptimizer.step前後の差。各列の最大値は同じStepとは限らない。

| head | clip後L2最大 | clip後RMS最大 | Δθ L2最大 |
|---|---:|---:|---:|
| Add Where | 0.862602 | 0.00654483 | 0.00610878 |
| Prune Where | 0.966652 | 0.01089432 | 0.00566081 |
| Adjust Where | 0.973183 | 0.00738384 | 0.00924202 |
| Add Amount | 0.860271 | 0.11394567 | 0.00006560 |
| Prune Amount | 0.892006 | 0.11814914 | 0.00006565 |
| Adjust Amount | 0.705571 | 0.09345510 | 0.00006557 |
| coarse Amount | 1.000000 | 0.01460516 | 0.00027915 |

丸め前の値、各更新、Actor/Proposal、更新率は `release_summary.json` に保存。Fineの初期parameter normは0または非常に小さく、相対更新率だけで更新の大きさを判断しない。

独立同時探索自体の以前の8更新では、固定ValidationでNo-opに留まり、den6を超える改善を確認できていない。16候補化・高速化を、den6同等以上の最終精度の達成と取り違えてはいけない。

## Falseの保全と実験上の制約

自律探索変更前snapshotのFalseと、8i / MVUB / UVGの全既存1,378 args、9種類のpool・RNGが一致した。高速化分岐もTrue / SparsePCGCに限定。以前に実行したFalseの8更新・4frame Validationでは16回の編集点群hashとC/Gが一致している。

追加のFalse確認を同期timing wrapperで試したrunは、最初の更新前に待ち状態になったため打ち切った。原因は未確定。このrunを成功したFalse学習検証には数えない。当該wrapperはTrue専用へ制限し、Falseには従来の診断runnerを使う。停止したのは今回新規に起動した当該診断とその子workerだけで、既存の訓練processを停止していない。

**1秒/Step未満は未達。** 最終版のActual encoderだけでも再利用時平均約2.42秒。現在の環境で毎Stepの全点Actual教師を維持したまま1秒未満を達成した証拠はない。次の候補はpoolのTensor化、座標対応計算のさらなる厳密高速化、native教師の詳細profile。ただしcodecや評価頻度を変更する場合は別途同条件検証が必要。Actual省略や代理loss置換で達成したことにはしない。

## 成果物と再現

`/data/maejima/log/independent_experiments/20261028_autonomous_speed_mo5pxeo9/`

- `before/`：最適化前source hash・内容、作業前Git差分。
- `before_k6_replay8/` / `after_k6_v4/`：同条件8更新、勾配・重み、Actual/Geometry、タイミング。
- `expanded_k16_release8/`：最終16候補版。
- `comparison8_final.json`：6候補版の数値比較。`release_summary.json` は16候補版の全8更新・勾配・更新量・時刻も含む。
- `final_source_v2/manifest.json`：最終117 sourceのSHA256。先の `final_source/` は既存の`.py`名ディレクトリをfileとして読もうとして停止した未完了snapshotなので使用しない。
- `tests_regression_final.txt` / `tests_v4.txt`：回帰・CUDA検証。
- `false_isolation_final.json`：False設定/pool/RNG比較。

新規の `experiments/autonomous_speed_probe.py` で私有cache / registry / 出力先を指定し、`experiments/summarize_autonomous_speed.py` で集計する。最適化前は `--source-snapshot` で既存sourceを変更せず読み取りreplayする。

新しい主要計算helperは `models/utils/loss/autonomous_exact_geometry.py`。既存のtrain/model/lossにはTrue分岐として接続した。実行中runはsource変更がないことをhashで監査。既存checkpoint・既存ログ・G-PCC / OctAttention / Native SparsePCGCソースは上書きしていない。
