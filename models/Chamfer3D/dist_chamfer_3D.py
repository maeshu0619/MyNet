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


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamfer_3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        if chamfer_3D is None:
            dist = torch.cdist(xyz1, xyz2).pow(2)
            dist1, idx1 = dist.min(dim=2)
            dist2, idx2 = dist.min(dim=1)
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
            with torch.enable_grad():
                x1 = xyz1.detach().requires_grad_(True)
                x2 = xyz2.detach().requires_grad_(True)
                dist = torch.cdist(x1, x2).pow(2)
                d1 = dist.min(dim=2).values
                d2 = dist.min(dim=1).values
                loss = (d1 * graddist1).sum() + (d2 * graddist2).sum()
                grad1, grad2 = torch.autograd.grad(loss, (x1, x2), allow_unused=True)
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
