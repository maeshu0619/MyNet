# Actual BD-rate evaluation utilities

Entry point: `../../test_BDrate.py` (from the repository root, `test_BDrate.py`).
No training/codec source is changed. Modules are intentionally independent:

- `io.py`: existing binary PLY IO, exact coordinates, durable exclusive artifacts/hashes.
- `codecs.py`: existing G-PCC/Draco command builders plus independent native decode.
- `learned_worker.py`: isolated OA / SparsePCGC physical-stream encode/decode wrappers.
- `metrics.py`: MPEG `pc_error`, fixed GT normals/resolution and duplicate policy.
- `curves.py`: strict Akima log10-rate integration and dense/coarse comparison.
- `inference_worker.py`: original `test.py` inference, with strict checkpoint validation.
- `runner.py`: streaming runs, per-point JSONL failures, immutable output directories.
- `report.py`: Excel, full-precision JSON, aggregate completeness and RD plots.

Python environments need numpy, scipy, openpyxl, matplotlib and plyfile. The
existing network and learned codecs retain their original runtime requirements.
The codec-only SparsePCGC environment can use existing binary PLY IO without
plyfile. No automatic environment installation or CUDA-process management occurs.

## Dense point configuration

`--points-json FILE` accepts a JSON list, not invented codec quality switches:

```json
[{"positionQuantizationScale":0.125}, {"positionQuantizationScale":0.25},
 {"positionQuantizationScale":0.5}, {"positionQuantizationScale":0.75},
 {"positionQuantizationScale":0.875}, {"positionQuantizationScale":0.9375}]
```

Use the corresponding keys `qp` (Draco), `qs` (OA), or `scale_ae`, `scale_sr`
(SparsePCGC; optional `voxel_size`, which explicitly changes voxelization).
No arbitrary continuous quality parameter exists for the official six-preset
SparsePCGC dense codec. Extra settings must use these actual controls and are
not automatically advertised as paper CTC configurations.

With 10–15 points, select six `--coarse-indices 0 3 6 8 11 14` (example for
15 points). The report compares ordinary BD-rates and, separately, both estimates
on their identical intersection. Difference is in percentage points, not percent
relative error. An invalid/nonmonotonic curve is never silently repaired.

## Scientific protocol

- Baseline and MINE share native codec settings; both use **Original** point count.
- One deterministic MINE point cloud per GT is reused across RD points. Network
  args are fixed, independent of the downstream RD sweep. Per-rate retraining or
  conditioning would be a different experiment and must be specified separately.
- D1 and D2 come from the actual decoded cloud against the same unchanged GT.
- D2 uses supplied GT unit normals; no PCA fallback. Float32 normal properties
  are staged separately, because the installed pc_error 0.13.4 misreads binary
  float64 normals. XYZ coordinates are not changed by this format conversion.
- `resolution` is explicit; PSNR convention is `10log10(3*resolution^2/MSE)`.
- `pc_error` defaults are pinned: symmetric, dropdups=2, neighborsProc=1,
  averageNormals=1, one thread. Input vertex count, before metric deduplication,
  remains the bpp denominator.
- Infinite PSNR is recorded as `"inf"`, excluded with a warning from integration.
  Conflicting duplicate PSNR, nonmonotonic rates or fewer than four finite knots
  produce no number. Six finite knots are recommended, not assumed.
- Native tool output has limited textual precision (typically six significant
  digits). We retain its output without further rounding; no false claim of
  inaccessible internal double-precision measurements is made.
- Original `test.py` can adapt old checkpoint tensor shapes. For thesis runs the
  new wrapper rejects missing trainable keys, reshaped weights, unexpected keys
  or changes to saved weights instead of evaluating an implicitly modified model.
- `--identity-control` is an explicitly labelled codec/metric control, NOT a
  trained MINE result. Expected BD-rate is zero only when valid RD curves exist.
- Failures are preserved in per-point JSON and fsynced JSONL. Complete-dataset
  averages are suppressed when a sequence fails; valid-only means are separate.

## Learned codec side information

OA's `.mat` contains source occupancy and is NOT counted as a decoder stream.
The autonomous decoder uses prior decoded symbols only. A **56-byte MYOA0001**
record (`<8sQd3dII`: magic, symbol count, quantizer, XYZ offset, context length,
tree depth) is serialized and included in bits along with actual arithmetic bytes.

SparsePCGC dense lossy `LossyCoderDense.test` has two issues for this task: its
upsampling call receives encoder `x` instead of freshly decoded `x_dec`, and its
stage point-count overhead is only a logical estimate. The wrapper reuses the
same downscale/encode/unpack/decode/upscale functions but upscales the **decoded**
low-resolution tensor. Lossless low-resolution coordinate equality is checked.
**MYSP0001** serializes `<8sBId3dIQ` (magic, AE/SR, voxel size, XYZ offset,
stage count, AE byte count), uint64 stage point counts and actual AE bytes.
All these physical bytes are charged. Checkpoints are fixed shared decoder models,
as usual; their files are hashed but not charged per frame. This wrapper overhead
differs from paper conventions and is explicitly identified in every result.

Neither wrapper changes the underlying codec/checkpoint. A standards-native
container overhead comparison would be a separate protocol, not a free omission
of decoder-required information.
