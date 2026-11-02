# Network側高速化調査の数値再検証

2026-11-03。前回のNetwork側調査開始時snapshotを基準に再確認した。**入力・CPU勾配試作の一致は再現したが、実行中訓練のGPU→CPU自動移行に関連するActual符号量の差を確認した。「すべての数値が変わっていない」とは結論できない。**

## 本番ソース

`/data/maejima/log/independent_experiments/20261103_network_only_rf8i6b4w/before/manifest.json`の135ファイル、および`codec_sha.json`のSparsePCGC本体53ファイルを再hashし、変更0件だった。Network側の読込cache・限定backward試作は本番へ導入していない。比較基準はこのsnapshotであり、過去のden6設計全般との同一性を意味しない。

本確認でも本番train/Network/loss/args/worker/codecを編集せず、既存process・checkpoint・ログを手動変更していない。追加は独立CPU再検証scriptと本レポートのみ。

## 入力の再検証

元parserで4系列を新しく読み直し、試作のloadtxt / fromstring / 解析済みNPYと比較した。入力PLY全体のSHA256、NPY配列全byteのSHA256も前回記録と一致した。

| 入力 | 比較対象 | 最大絶対差 | 異なるfloat32 bit pattern |
|---|---|---:|---:|
| longdress 1051 | 3方式すべて | 0 | 0 |
| loot 1000 | 3方式すべて | 0 | 0 |
| redandblack 1450 | 3方式すべて | 0 | 0 |
| soldier 0536 | 3方式すべて | 0 | 0 |

点数・dtype・shape・座標・色・行順が一致した。今回の新しい照合は12組で、前回のparser45件/cache20件と区別する。この確認は4フレームの入力同値性であり、100フレームのend-to-end学習比較ではない。

## 勾配・更新の再検証

`network_capture_speed_audit.py --self-test`を再実行。CPUの検証用モデルでreentrant/non-reentrant checkpointの両条件について、元captureと対象限定captureのparameter勾配・retained decision勾配が一致した。各条件3回のAdam更新後parameterと乱数状態も一致した。使ったlossは検証用モデルのlossであり、Actual codec教師を使った実Networkの確認ではない。

既存の`test_gradient_diagnostic_batching`と`test_autonomous_joint_learning`もCPUで再実行し、16テスト合格。**実Networkの全head・AMP・成熟checkpoint・長期収束・固定Validationの圧縮率/Geometry維持は未確認**。限定backwardは本番未導入のままとした。

## CPUへ切り替わったActualの差

実行中の`train.py --test_route true`（PID 766550）のログに、GPU空き2,605.9 MiBが必要4,096 MiBを下回ったための`SparsePCGCAutoCPUFallback`があり、workerはCPUのPID 1703418で継続している。前回の独立検証もGPUを共有していたため、競合へ寄与した可能性がある。元のGPU workerを手動復帰・再起動していない。

GPUを追加使用せず、`CUDA_VISIBLE_DEVICES=''`で別processを起動した。過去のGPU実測と同じ入力byte hashを持つ2点群を、同じnativeモデルpath、dense_lossy、m=8 / AE=0 / SR=2でCPU encode/decodeした。SparsePCGCソースは編集せず既存workerを読み込んだ。CPU main streamをnative decodeし、nativeのcount/AE side情報でupscaleした座標を比較した。

| 独立fixture | GPU physical main bits | CPU physical main bits | GPU logical bits | CPU logical bits | 復号後座標集合 |
|---|---:|---:|---:|---:|---|
| GT | 38,272 | 38,272 | 38,280 | 38,280 | 一致 |
| Add/Prune/Adjustを各128適用 | 38,304 | **38,296** | 38,312 | **38,304** | 一致 |

logicalは既存native計算に従うcount side情報8 bits込み。CPU/GPUともpoint/node/single-child数は一致し、両fixtureの復号点数は765,821点だった。復号座標集合のSHA256も一致した。PSNRを新たに測ったわけではないが、この2件では最終復号座標に変化はなかった。

編集fixtureはCPU encodeをさらに3回反復し、全てphysical 38,296 / logical 38,304 bitsだった。最初のdecode付き1件を含めCPU4件で同値。GPUの保存済みnative反復3件は全てphysical 38,304 / logical 38,312 bitsで、差8 bitsが再現した。追加CPU3件はencodeのみで、decodeまで再実行した結果として数えない。

このfixtureは学習済みNetworkが選んだplanではない。単純なrate比`100*(edited_logical_bits/GT_logical_bits-1)`に換算すると、GPU +0.083594566%、CPU +0.062695925%、差 **−0.020898642 percentage point**。これは全体lossの変化量ではない。

**この8 bits差は、今回のNetwork側試作を本番へ導入したための差ではない。** 同じ点群でもCPUと保存済みGPU結果で実測符号量が異なった。一方、NN出力・整数CDFをsymbolごとにtraceしていないため、特定の演算・丸めだけを原因と断定しない。一般の全フレーム・全checkpointで差が8 bits以内という保証でもない。

学習のActual教師へ厳密に同じ数値を要求するなら、比較の実行デバイスも固定する必要がある。今後のActual教師の違いは学習軌跡へ影響し得るため、現状で長期収束後の精度不変を保証できない。CPUへの移行を隠したまま「ソースが同じなので訓練数値も完全一致」とは報告しない。

## 成果物

`/data/maejima/log/independent_experiments/20261103_network_only_rf8i6b4w/recheck_kdt_hxzg/`:

- `results.json`: 再hash、入力12組、CPU勾配/更新fixtureの結果。
- `gradient_tests.txt`: 16テスト合格。
- `cpu_codec/results.jsonl`: CPU encode/decode2件とGPU参照の直接比較。
- `cpu_repeat/results.jsonl`: 編集fixtureの追加CPU encode3件。
- GPU参照: `20261103_native_encode_4n19ybd9/timing/records.jsonl`。

追加scriptは`experiments/network_numeric_cpu_recheck.py`。GPUを隠したCPU実行をassertし、新規出力先だけを使用する。後続反復ではJIT build cacheもprivate出力先に分離した。最初の2件の実行では環境既定のtorch extension cacheが使われたが、codecソースやモデルcheckpointを書き換える処理は行っていない。
