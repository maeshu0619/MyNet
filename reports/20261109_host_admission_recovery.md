2026-11-09 SparsePCGC True訓練のホストRAM待機timeout修正

最新ログは`/data/maejima/log/20261107/MyNetwork_train/SparsePCGC/142233.txt`。MVUB・5系列×20フレーム、毎Episode100 Step。最後の表示はE98/Epo5/Step20だが、98 Episodeと固定Validationは完了し、E99のandrew先頭、global_step=9800の圧縮前admissionで止まっていた。

直接原因はNetwork側wrapperの`_wait_for_host_capacity`。空きRAMが設定閾値8192 MiBを満たさないまま600秒経つとRuntimeErrorを投げる実装だった。待機開始時7857.0 MiB、最後5854.9 MiB。これはこのTracebackの原因であり、圧縮計算のエラーや構文エラーとして扱うものではない。

過去のメモリ診断CSVを確認した。

| Episode終了 | 学習本体RSS[MiB] | teacher RSS[MiB] | host MemAvailable[MiB] |
|---|---:|---:|---:|
| 1 | 6086.664 | 5811.566 | 18333.434 |
| 20 | 5831.508 | 5869.578 | 17477.918 |
| 60 | 5396.582 | 6266.621 | 15481.301 |
| 80 | 5417.930 | 6746.316 | 10180.840 |
| 90 | 5430.961 | 6977.996 | 13678.887 |
| 98 | 5440.742 | 7180.855 | 8406.938 |

本体RSSは際限なく増えていない。一方teacher RSSは約1369 MiB増加。host全体の空き減少はこの増分より大きく、共有hostの負荷変動も含む。停止時の全process別履歴がないため、他jobだけのせいとも、worker増加だけのせいとも断定しない。worker増加が生存Tensor・allocator保持・driver等のどれに何MiBあるかは未分離である。

実装した対処

対象はSparsePCGCの`test_route True`だけ。閾値8192 MiB、worker初期化時の12288 MiBは維持した。空きが十分な通常Stepは解放RPC・GC・sleepを追加しない。

RAM不足時は以下を行う。

1. 親processで不要オブジェクトをGCし、glibcの未使用領域を`malloc_trim`でOSへ返す。生きたmodel/optimizer/gradient graphや実測教師memoryは削除しない。
2. 既存のidle teacherへ`release_runtime_cache`制御RPCを送る。Network側worker wrapperが既存のcleanup（座標manager/GC/CUDA allocator cache）とmalloc_trimを呼ぶ。モデルの再ロード・checkpoint変更・device変更・encodeは行わない。
3. 閾値を満たすまで同じStepを保持して待つ。600秒を超えても強制終了せず、状況を通知して再解放・待機を継続する。回復後に元の要求を実行する。

制御RPCでは次frameのpool prefetchを開始しない。RAM解放のためにRAMを追加消費することを避ける。workerのencode完了counterは増やさず、Actual結果やbit値を返すcommandにもしていない。低RAMのままActualを見切り実行する、proxyで代用する、Stepを飛ばす、optimizerだけ進める変更ではない。

新しい設定は`--sparsepcgc_host_pressure_recovery true`（既定True）。実際に有効になるのはSparsePCGC / test_route Trueのみ。既存の`--sparsepcgc_host_wait_timeout 600`はこの経路では通知・再解放間隔となる。明示的にrecovery=falseなら元のtimeout例外を再現する。test_route Falseは従来の待機上限・例外動作を維持する。

ログは`SparsePCGCHostAdmissionWait`、`SparsePCGCHostMemoryReclaim`、`SparsePCGCHostAdmissionDeferred`、`SparsePCGCHostAdmissionReady`で追跡できる。KeyboardInterruptは握りつぶさない。

検証

- 関連126テスト合格。Reported値5854.9 MiBを使い、600秒・1200秒を超えた待機後、1202秒相当で回復するケースを時計だけ模擬して確認。実際にRAMを枯渇させたり、20分sleepした試験ではない。
- 安全閾値に達する前にencodeされず、回復後は同じ要求を1回だけ実行。proxyラベルの生成なし。
- 通常RAM時は解放RPC/GC/sleepなし。worker初期化12288 MiB維持、strict/Falseのtimeout維持、Ctrl-C伝播を確認。
- 待機中のmodel parameter・Adam state・RNGが変わらず、保持したautograd graphを待機後にbackward/stepできることを確認。
- worker cleanupでモデルparameterを変えず、制御RPCがpool prefetchを開始しないことを確認。
- False経路の既存args、9 pool/RNGケース、12 worker command/fingerprintが開始時snapshotと一致。

実圧縮の確認はCPUで、MVUB andrew/frame0000.plyの全279664点を使った。物理RAMに16 GiB以上空きがあることを確認し、同じworkerで通常encode→admission readerが1回だけ5854.9 MiBを返す模擬イベント→解放後encode→次のencodeを行った。実際の共有hostメモリを占有して低RAM状態を作る試験ではない。

| 項目 | 通常 | 解放後 | 次要求 |
|---|---:|---:|---:|
| bit（既存teacherのrate定義） | 17400 | 17400 | 17400 |
| bpp | 0.06221751816465473 | 同値 | 同値 |
| 圧縮入力点数 | 279664 | 279664 | 279664 |
| node統計 | 90198 | 90198 | 90198 |
| single統計 | 11335 | 11335 | 11335 |

入力Tensorのhashも不変、worker起動は1回。解放RPC時のCPU worker RSSは697.85→617.02 MiBだった。これは独立したCPU検証の値であり、過去GPU workerの約1.37 GiB増加を全部回収できたという意味ではない。復号D1/D2やGPU上の長期98 Episodeを再実行した検証ではない。

最初のpilotは検証scriptがencoderの単一cloud APIへbatch次元付きTensorを渡し、誤って3点になっていたため精度証拠に使わない。scriptを[3,N]へ修正し、全点数・入力hashのassertを追加して`actual_cpu_full`で再実行した。`actual_cpu/`を成功の根拠としない。

変更・保全

- 新規`models/utils/loss/host_memory_admission.py`：True専用の解放・継続待機。
- `actual_encoder.py`：上記admissionの分岐、制御RPC時のprefetch抑止。
- `sparsepcgc_teacher_worker.py`：既存modelを保つメモリ解放command。これはNetwork側wrapperであり、compress/octree/SparsePCGC本体ではない。
- `args.py`：recovery flagとhelp。開始前からあったdatasetのMVUB選択編集は保持した。
- 新規tests/test_sparsepcgc_host_admission.pyとexperiments/verify_sparsepcgc_host_recovery.py。

SparsePCGC本体Python53ファイルはSHA256変更0。train.py、Network、Surrogate損失式、行動選択・探索schedule、圧縮パラメータは変更していない。既存checkpoint・ログへの書込みや、他processの終了・再設定は行っていない。

物理RAMが回復しない間は訓練も待機する。安全条件を緩めて必ず進ませる修正ではなく、回復可能な共有資源不足で600秒後に訓練状態を失う問題への対処である。未使用メモリの全量回収や、共有hostでの将来のOOM回避を保証するものではない。

同じ`python train.py --test_route True`で次回起動から有効。既存runの自動再開やcheckpoint巻き戻しは行わない。

成果物：`/data/maejima/log/independent_experiments/20261109_host_admission_7j5bfky_/`。before/、user_diff_before.patch、memory_evidence.json、recovery_tests.log、all_tests_final.log、false_regression.json、actual_cpu_full/results.json・worker_events.txt、integrity.json、patch_with_preserved_user_edit.diff。
