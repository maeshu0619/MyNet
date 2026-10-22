import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.shape).encode()+array.dtype.str.encode()+array.tobytes()).hexdigest()


def canonical_points(value):
    array = np.asarray(value)
    return np.unique(array.reshape(-1, 3), axis=0)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False))


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, default=str, allow_nan=False)+'\n')


def correlations(predicted, target):
    from scipy.stats import spearmanr
    x, y = np.asarray(predicted, dtype=float), np.asarray(target, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    result = dict(n=len(x), pearson=None, spearman=None, pair_accuracy=None, clear_pairs=0)
    if len(x) > 2 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
        result.update(pearson=float(np.corrcoef(x,y)[0,1]),spearman=float(spearmanr(x,y).correlation))
    correct=[];strict=[];ties=[]
    for i in range(len(x)):
        for j in range(i):
            if abs(y[i]-y[j]) > 1e-9:
                tie=abs(x[i]-x[j])<=1e-12
                right=(x[i]-x[j])*(y[i]-y[j])>0
                correct.append(.5 if tie else float(right));strict.append(float(right));ties.append(tie)
    if correct:
        result.update(pair_accuracy=float(np.mean(correct)),strict_pair_accuracy=float(np.mean(strict)),
                      predicted_tie_fraction=float(np.mean(ties)),clear_pairs=len(correct))
    return result


class FrozenSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, source):
        self.source=Path(source)

    def find_spec(self, fullname, path=None, target=None):
        relative=Path(*fullname.split('.')).with_suffix('.py')
        if (self.source/relative).is_file():
            return importlib.util.spec_from_loader(fullname,self,origin=str(ROOT/relative))

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        relative=Path(*module.__name__.split('.')).with_suffix('.py')
        module.__file__=str(ROOT/relative)
        exec(compile((self.source/relative).read_text(),str(ROOT/relative),'exec'),module.__dict__)


def source_hashes():
    paths=[ROOT/'train.py',*sorted((ROOT/'models').rglob('*.py'))]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def module_digest(model):
    import torch
    h=hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def runtime_digest(value):
    """Content hash, retaining OrderedDict eviction order, never object IDs."""
    import torch
    from collections import OrderedDict
    def encode(item):
        if torch.is_tensor(item):return ('tensor',str(item.dtype),digest(item.detach().cpu().contiguous().numpy()))
        if isinstance(item,np.ndarray):return ('numpy',digest(item))
        if isinstance(item,dict):
            entries=list(item.items()) if isinstance(item,OrderedDict) else sorted(item.items(),key=lambda kv:repr(kv[0]))
            return (type(item).__name__,tuple((repr(k),encode(v)) for k,v in entries))
        if isinstance(item,(list,tuple)):return (type(item).__name__,tuple(encode(v) for v in item))
        return repr(item)
    return hashlib.sha256(repr(encode(value)).encode()).hexdigest()


def materialize(source, pools, order, counts, ordered_indices):
    occupied=set(map(tuple,np.asarray(source,dtype=np.int64)))
    removes, adds, selected=set(),set(),{op:[] for op in order}
    for op in order:
        for index in ordered_indices[op]:
            if len(selected[op]) >= int(counts[op]):
                break
            candidate=pools[op][index]
            r=set(map(tuple,candidate.get('remove_coords') or ()))
            a=set(map(tuple,candidate.get('add_coords') or ()))
            if not r.issubset(occupied) or a & occupied:
                continue
            if (r|a) & (removes|adds):
                continue
            removes.update(r); adds.update(a);selected[op].append(index)
    final=np.asarray(sorted((occupied-removes)|adds),dtype=np.int64).reshape(-1,3)
    return final, selected


def geometry(source, edited):
    from scipy.spatial import cKDTree
    a,b=np.asarray(source,dtype=np.float64),np.asarray(edited,dtype=np.float64)
    x=cKDTree(b).query(a,k=1,workers=2)[0]
    y=cKDTree(a).query(b,k=1,workers=2)[0]
    return dict(chamfer_sum=float(np.mean(x*x)+np.mean(y*y)),
                d1_mse=float(max(np.mean(x*x),np.mean(y*y))),max_distance=float(max(x.max(),y.max())))


def physical_points(plan, coords):
    step=np.asarray(plan.get('voxel_step',1.),dtype=np.float32).reshape(-1)[0]
    offset=np.asarray(plan.get('voxel_offset',np.zeros(3)),dtype=np.float32).reshape(3)
    return np.asarray(coords,dtype=np.float32)*step+offset


def geometry_with_fit(plan, coords, selected, args):
    """Reuse the real added-point fit implementation, not a substitute loss.

    NN ties choose the first original GT row, as the CUDA Chamfer kernel does.
    Validate the aggregate against saved native-forward Geometry before using
    this reconstruction to claim preservation of the training objective.
    """
    import torch
    from scipy.spatial import cKDTree
    from models.utils.loss.geometry import GeometryLossMixin
    source=np.asarray(plan['source'],dtype=np.int64)
    gt=physical_points(plan,source);edited=physical_points(plan,coords)
    result=geometry(gt,edited)
    occupied=set(map(tuple,source));final=set(map(tuple,coords))
    add=set()
    for index in selected.get('Add',[]):
        add.update(map(tuple,plan['pools']['Add'][index].get('add_coords') or ()))
    add=np.asarray(sorted((add-occupied)&final),dtype=np.int64).reshape(-1,3)
    if len(add):
        locations=physical_points(plan,add);tree=cKDTree(gt)
        distances,indices=tree.query(locations,k=1,workers=2)
        for i,near in enumerate(tree.query_ball_point(locations,distances+1e-5,workers=2)):
            near=np.asarray(near,dtype=int);squared=np.sum((gt[near]-locations[i])**2,axis=1)
            indices[i]=int(near[squared==squared.min()].min())
        fit,count,valid=GeometryLossMixin._added_point_to_plane_fit(
            torch.from_numpy(locations.T.copy()).unsqueeze(0),torch.from_numpy(gt.T.copy()).unsqueeze(0),
            torch.from_numpy(source),torch.from_numpy(indices).long(),
            radius=int(getattr(args,'geometry_fit_normal_radius',2)),
            min_neighbors=int(getattr(args,'geometry_fit_min_neighbors',3)))
        fit=float(fit.detach());count=int(count);valid=int(valid)
    else:fit=0.;count=0;valid=0
    weighted=float(getattr(args,'geometry_fit_weight',.05))*fit
    result.update(fit=fit,fit_weighted=weighted,fit_count=count,fit_valid=valid,
                  training_G_reconstructed=result['chamfer_sum']+weighted)
    return result
