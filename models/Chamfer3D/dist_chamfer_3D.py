from torch import nn
from torch.autograd import Function
import torch
import importlib
import importlib.util
import os
chamfer_found = importlib.util.find_spec("chamfer_3D") is not None
chamfer_3D = None
if not chamfer_found and os.environ.get("MYNET_USE_CHAMFER_CUDA", "0").strip().lower() in {"1", "true", "yes"}:
    ## Cool trick from https://github.com/chrdiller
    print("Jitting Chamfer 3D")

    from torch.utils.cpp_extension import load
    chamfer_3D = load(name="chamfer_3D",
          sources=[
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_cuda.cpp"]),
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer3D.cu"]),
              ])
    print("Loaded JIT 3D CUDA chamfer distance")

else:
    if chamfer_found:
        import chamfer_3D
    # print("Loaded compiled 3D CUDA chamfer distance")


def _fallback_chunk_size(batch_size, other_points):
    max_chunk = int(os.environ.get("MYNET_CHAMFER_CHUNK_SIZE", "4096"))
    max_elements = int(os.environ.get("MYNET_CHAMFER_MAX_CDIST_ELEMENTS", str(64 * 1024 * 1024)))
    denom = max(int(batch_size) * int(other_points), 1)
    by_elements = max(max_elements // denom, 1)
    return max(1, min(max_chunk, by_elements))


def _chunked_min_cdist(src, dst):
    batch_size, src_points, _ = src.shape
    other_points = int(dst.shape[1])
    if src_points == 0:
        return src.new_zeros(batch_size, 0), torch.zeros(batch_size, 0, device=src.device, dtype=torch.long)
    if other_points == 0:
        return src.new_zeros(batch_size, src_points), torch.zeros(batch_size, src_points, device=src.device, dtype=torch.long)
    chunk_size = _fallback_chunk_size(batch_size, other_points)
    mins = []
    indices = []
    for start in range(0, src_points, chunk_size):
        stop = min(start + chunk_size, src_points)
        dist = torch.cdist(src[:, start:stop, :], dst).pow(2)
        vals, idx = dist.min(dim=2)
        mins.append(vals)
        indices.append(idx)
    return torch.cat(mins, dim=1), torch.cat(indices, dim=1)


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamfer_3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        if chamfer_3D is None:
            dist1, idx1 = _chunked_min_cdist(xyz1, xyz2)
            dist2, idx2 = _chunked_min_cdist(xyz2, xyz1)
            ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
            return dist1, dist2, idx1.int(), idx2.int()
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)
        torch.cuda.set_device(device)

        chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        if chamfer_3D is None:
            xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
            grad1 = torch.zeros_like(xyz1)
            grad2 = torch.zeros_like(xyz2)

            idx1 = idx1.long()
            idx2 = idx2.long()
            idx1_exp = idx1.unsqueeze(-1).expand(-1, -1, xyz1.shape[-1])
            idx2_exp = idx2.unsqueeze(-1).expand(-1, -1, xyz2.shape[-1])

            nearest2 = torch.gather(xyz2, 1, idx1_exp)
            diff1 = xyz1 - nearest2
            contrib1 = 2.0 * diff1 * graddist1.unsqueeze(-1)
            grad1 = grad1 + contrib1
            grad2.scatter_add_(1, idx1_exp, -contrib1)

            nearest1 = torch.gather(xyz1, 1, idx2_exp)
            diff2 = xyz2 - nearest1
            contrib2 = 2.0 * diff2 * graddist2.unsqueeze(-1)
            grad2 = grad2 + contrib2
            grad1.scatter_add_(1, idx2_exp, -contrib2)
            return grad1, grad2
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.to(device)
        gradxyz2 = gradxyz2.to(device)
        chamfer_3D.backward(
            xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
        )
        return gradxyz1, gradxyz2


class chamfer_3DDist(nn.Module):
    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_3DFunction.apply(input1, input2)
