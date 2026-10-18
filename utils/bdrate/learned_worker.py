"""Isolated codec process. Wrappers add serialized side data, never codec edits.

OctAttention probabilities during decoding come only from previously decoded
symbols. Encoder probabilities / source occupancy / .mat files are not provided
to the decoder. SparsePCGC lossy stage counts and AE bytes are physically saved.
"""
import json
import os
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils.bdrate.io import read_vertices, save_json, sha256, write_geometry


def _oa_decode(teacher, stream, count, bptt):
    # Causal queue/window traversal follows official decoder.py:decodeOct.
    from collections import deque
    import numpy as np
    import torch
    import numpyAc
    from Octree import dec2bin
    model, device = teacher._model, teacher._oa_device
    mask = teacher._generate_square_subsequent_mask(bptt)
    data = torch.zeros((bptt, 1, 4, 3), dtype=torch.long, device=device)
    padding = torch.zeros_like(data)
    decoder = numpyAc.arithmeticDeCoding(None, count, 255, str(stream))

    def symbol(output):
        probs = torch.softmax(output, 1).squeeze().cpu().numpy()
        return int(np.asarray(decoder.decode(probs[None])).item()) + 1

    with torch.inference_mode():
        root = symbol(model(data, mask, [])[-1])
        codes = [root]
        queue = deque([[[255, 0, 0], [root, 1, 1], [root, 1, 1]]])
        node_id = 0
        while len(codes) < count:
            if not queue:
                raise RuntimeError("OctAttention queue exhausted before declared symbol count")
            father = queue.popleft()
            occupancy = dec2bin(father[-1][0])[::-1]
            for i in range(8):
                if not occupancy[i]:
                    continue
                feat = torch.tensor([[father + [[root, father[-1][1] + 1, i + 1]]]], device=device).long()
                feat[:, :, :, 0] -= 1
                offset = node_id % bptt
                if offset == 0:
                    data = torch.cat((data[bptt:], feat, padding[:bptt - 1]))
                else:
                    data[offset] = feat
                root = symbol(model(data, mask, [])[offset])
                node_id += 1
                codes.append(root)
                queue.append(father[1:] + [[root, father[-1][1] + 1, i + 1]])
                if len(codes) == count:
                    break
    return np.asarray(codes)


def octattention(request):
    import numpy as np
    from types import SimpleNamespace
    import torch
    from models.utils.loss.actual_encoder import _OctAttentionActualEncoder
    s, folder = request["settings"], Path(request["folder"])
    teacher = _OctAttentionActualEncoder(SimpleNamespace(
        octattention_actualcode=True, octattention_ckpt=s["oa_checkpoint"],
        octattention_teacher_device=s["codec_device"], bptt=1024))
    teacher._lazy_init()
    from Octree import GenOctree, GenKparentSeq, DeOctree
    xyz, _ = read_vertices(request["source"])
    offset = xyz.min(axis=0)
    qs = float(request["point"]["qs"])
    q = np.unique(np.rint((xyz - offset) / qs).astype(np.int64), axis=0)
    _, tree, depth = GenOctree(q)
    if depth > 12:
        raise ValueError("OctAttention pretrained model maximum supported octree level is 12")
    seq = GenKparentSeq(tree, 4)
    seq = np.concatenate((seq["Seq"][:, :, None], seq["Level"], seq["Pos"]), axis=2)
    stream = folder / "occupancy.bin"
    _, count = teacher._compress_oct_seq(seq, str(stream))
    bptt = min(1024, count - 1)
    # Stable 56-byte wrapper side record. Not a benchmark .mat containing GT.
    side = folder / "geometry.side.bin"
    with side.open("xb") as handle:
        handle.write(struct.pack("<8sQd3dII", b"MYOA0001", count, qs, *offset, bptt, int(depth)))
    # Decoder-side settings are re-read from the actual serialized record.
    magic, n, quant, ox, oy, oz, window, level = struct.unpack("<8sQd3dII", side.read_bytes())
    if magic != b"MYOA0001":
        raise RuntimeError("Unknown OA wrapper format")
    codes = _oa_decode(teacher, stream, n, window)
    reconstructed = np.asarray(DeOctree(codes)) * quant + np.array([ox, oy, oz])
    # Integrity assertions only AFTER autonomous decoding.
    if not np.array_equal(codes, seq[:, -1, 0]) or not np.array_equal(np.unique(reconstructed, axis=0), np.unique(q * qs + offset, axis=0)):
        raise RuntimeError("Autonomous OctAttention symbol/geometry roundtrip mismatch")
    write_geometry(folder / "decoded.ply", reconstructed)
    return dict(streams=[str(stream), str(side)], checkpoint_sha256=sha256(s["oa_checkpoint"]),
                decoder="causal autonomous probabilities; no encoder probability/symbol side channel",
                rate_definition="actual arithmetic bytes + 56-byte serialized geometry side record",
                qs=qs, octree_depth=int(level), symbols=int(n))


def sparsepcgc(request):
    import importlib.util
    import numpy as np
    from types import SimpleNamespace
    s, folder, point = request["settings"], Path(request["folder"]), request["point"]
    root = REPO.parent / "compress/octree/SparsePCGC"
    spec = importlib.util.spec_from_file_location("bd_sparse_codec", root / "encoder_multiple.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(device=s["codec_device"], mode="dense_lossy",
        ckptdir=Path(s["sp_checkpoint"]), ckptdir_sr=Path(s["sp_sr_checkpoint"]),
        ckptdir_ae=Path(s["sp_ae_checkpoint"]))
    encoder = module.SparsePCGCEncoder(args)
    torch = encoder.torch
    # All loaded codec components must be deterministic evaluation models.
    for model in (encoder.coder.basic_coder.model, encoder.coder.model_AE, encoder.coder.model_SR):
        model.eval()
    ae, sr, voxel = int(point["scale_ae"]), int(point["scale_sr"]), float(point.get("voxel_size", 1))
    original, _ = read_vertices(request["source"])
    if not np.array_equal(original / voxel, np.rint(original / voxel)):
        # Existing load_sparse_tensor uses its own rounding: retain it explicitly.
        quantization_note = "existing load_sparse_tensor voxelization"
    else:
        quantization_note = "integer input on requested voxel grid"
    with torch.inference_mode():
        basic = encoder.coder.basic_coder
        x = basic.load_data(request["source"], voxel_size=voxel, posQuantscale=1)
        ref = getattr(basic, "ref_point", None)
        ref_xyz = np.zeros(3) if ref is None else ref[1:].cpu().numpy().astype(float)
        x_low, ae_bytes, counts = encoder.coder.downscale(x, scale_AE=ae, scale_SR=sr)
        stream = folder / "occupancy.bin"
        basic.pack_bitstream(basic.encode(x_low), str(stream))
        # Counts are decoder inputs, not guessed 4-bit estimates. Serialize uint64.
        side = folder / "geometry.side.bin"
        ae_bytes = b"" if ae_bytes is None else bytes(ae_bytes)
        with side.open("xb") as handle:
            handle.write(struct.pack("<8sBId3dIQ", b"MYSP0001", ae, sr, voxel, *ref_xyz, len(counts), len(ae_bytes)))
            handle.write(np.asarray(counts, dtype="<u8").tobytes())
            handle.write(ae_bytes)
        raw = side.read_bytes()
        fmt = "<8sBId3dIQ"
        head_size = struct.calcsize(fmt)
        magic, ae_dec, sr_dec, voxel_dec, rx, ry, rz, ncounts, nbytes = struct.unpack(fmt, raw[:head_size])
        counts_dec = np.frombuffer(raw[head_size:head_size + 8*ncounts], dtype="<u8").tolist()
        ae_dec_bytes = raw[head_size + 8*ncounts:]
        if magic != b"MYSP0001" or len(ae_dec_bytes) != nbytes or ncounts != sr_dec + ae_dec:
            raise RuntimeError("Invalid SparsePCGC wrapper side information")
        decoded_low = basic.decode(basic.unpack_bitstream(str(stream)))
        if not np.array_equal(np.unique(decoded_low.C.cpu().numpy(), axis=0), np.unique(x_low.C.cpu().numpy(), axis=0)):
            raise RuntimeError("SparsePCGC lossless low-resolution roundtrip mismatch")
        # Crucial: upscale DECODED low resolution, never encoder x_low.
        reconstructed = encoder.coder.upscale(decoded_low, ae_dec_bytes if ae_dec else None, counts_dec)
        xyz = (reconstructed.C[:, 1:].cpu().numpy() + np.array([rx, ry, rz])) * voxel_dec
    write_geometry(folder / "decoded.ply", xyz)
    return dict(streams=[str(stream), str(side)],
                checkpoint_sha256=sha256(s["sp_checkpoint"]), sr_checkpoint_sha256=sha256(s["sp_sr_checkpoint"]),
                ae_checkpoint_sha256=sha256(s["sp_ae_checkpoint"]),
                rate_definition="physical basic stream + serialized stage counts/AE bytes/coordinate side record",
                quantization_note=quantization_note, scale_ae=ae, scale_sr=sr,
                compatibility_note="MYSP0001 evaluation wrapper; overhead differs from paper logical-count convention")


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    import torch
    import numpy as np
    import random
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.set_num_threads(int(request["settings"]["threads"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {"OctAttention": octattention, "SparsePCGC": sparsepcgc}[request["codec"]](request)
    save_json(Path(request["folder"]) / "codec_result.json", result)


if __name__ == "__main__":
    main()
