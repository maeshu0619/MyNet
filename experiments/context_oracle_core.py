"""Isolated context-oracle helpers. Existing training and codecs are read-only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from octree_codec_research import OAActual, ActualCodec, unique, quality, load_ply, difference
from octree_joint_mask_research import apply_plan, parts, quality_ok, POPS


def costs(probs, symbols=None):
    from numpyAc.numpyAc import pdf_convert_to_cdf_and_normalize, _convert_to_int_and_normalize
    cdf = _convert_to_int_and_normalize(pdf_convert_to_cdf_and_normalize(probs), True)
    u = cdf.astype(np.uint16).astype(np.int64)
    freq = (u[:, 1:]-u[:, :-1]) % 65536
    if not np.all(freq.sum(1)==65536) or not (freq>0).all():
        raise AssertionError('Invalid integer CDF')
    c = 16.-np.log2(freq)
    return c if symbols is None else c[np.arange(len(c)), symbols]


def stream(probs, symbols):
    import numpyAc
    payload, bits = numpyAc.arithmeticCoding().encode(probs, symbols.astype(np.int16))
    if bits != len(payload)*8:
        raise AssertionError('Arithmetic stream length convention changed')
    return dict(bits=int(bits), stream_sha256=hashlib.sha256(payload).hexdigest())


class WindowCodec:
    """Exact fixed-tree leaf edits, all affected inference chunks recomputed.

    Non-boundary leaf symbols only. Topology-changing edits must use full OA.
    Full arithmetic encoder is used for every reported Actual stream length.
    """
    def __init__(self, oa, q):
        self.oa=oa
        self.baseline,self.seq,self.probs=oa.encode(q,detail=True,roundtrip=True)
        self.symbols=(self.seq[:,-1,0]-1).astype(np.int16)
        self.all_cost=costs(self.probs).astype(np.float32)
        self.base_cost=costs(self.probs,self.symbols)
        self.bptt=min(oa.teacher._oa_bptt,len(self.seq)-1)
        self.ids,self.padded=oa.teacher._batchify(self.seq,self.bptt,len(self.seq))
        self.mask=oa.teacher._generate_square_subsequent_mask(self.bptt)
        self.calls=0

    def group(self,row):
        if row % self.bptt==0:
            raise ValueError('Boundary leaf not in fixed-window search')
        return row//self.bptt

    def predict(self, edits):
        groups={}
        for row,after in edits:
            if self.seq[row,-1,1]!=self.seq[:,-1,1].max() or not 1<=after<=255:
                raise ValueError('Only nonempty deepest leaf masks are supported')
            groups.setdefault(self.group(row),[]).append((row,after))
        chunks=[]
        for group,items in sorted(groups.items()):
            start=(group+1)*self.bptt
            inp=self.padded[start:start+self.bptt].clone()
            target_ids=self.ids[start+1:start+self.bptt+1].reshape(-1).numpy()
            valid=target_ids>=0
            target_ids=target_ids[valid]
            symbols=self.symbols[target_ids].copy()
            for row,after in items:
                inp[row%self.bptt,0,-1,0]=after-1
                symbols[row-(group*self.bptt+1)]=after-1
            with torch.inference_mode():
                output=self.oa.teacher._model(inp.to(self.oa.teacher._oa_device),self.mask,[])
                p=torch.softmax(output.reshape(-1,255),1)[torch.from_numpy(valid).to(output.device)].cpu().numpy()
            self.calls+=1
            chunks.append((target_ids,p,symbols))
        return chunks

    def evaluate(self, edits, actual=False, return_probs=False):
        edit_map=dict(edits)
        if len(edit_map)!=len(edits):
            raise AssertionError('Duplicate parent mask')
        chunks=self.predict(edits)
        own=sum(float(self.all_cost[r,self.symbols[r]]-self.all_cost[r,a-1]) for r,a in edits)
        total,edited_gain,affected_cost,affected_prob=0.,0.,0,0
        node_rows=[]
        for ids,p,symbols in chunks:
            new_cost=costs(p,symbols)
            gain=self.base_cost[ids]-new_cost
            changed=np.array([int(r) in edit_map for r in ids])
            total+=float(gain.sum())
            edited_gain+=float(gain[changed].sum())
            affected_cost+=int((np.abs(gain[~changed])>1e-9).sum())
            affected_prob+=int((np.max(np.abs(p[~changed]-self.probs[ids[~changed]]),axis=1)>1e-7).sum())
            node_rows.append((ids,gain,changed))
        result=dict(frozen_self_cdf_gain=own,edited_node_cdf_gain=edited_gain,
            unedited_context_cdf_gain=total-edited_gain,total_cdf_gain=total,
            affected_unedited_cost_nodes=affected_cost,affected_unedited_probability_nodes=affected_prob,
            changed_nodes=len(edits))
        if actual:
            full=self.probs.copy()
            symbols=self.symbols.copy()
            for ids,p,target in chunks:
                full[ids]=p
                symbols[ids]=target
            result['actual_oa']=stream(full,symbols)
            result['actual_oa_gain']=self.baseline['bits']-result['actual_oa']['bits']
            if abs(result['actual_oa_gain']-total)>16:
                raise AssertionError('CDF / Actual stream gain discrepancy exceeds byte termination')
        if return_probs:
            result['_chunks']=chunks
            result['_node_rows']=node_rows
        return result


def prior_features(q, seq, edits, model):
    """Current cold-start prior logit, generalized additively to joint masks.

    Not an evaluation of the running Actor. No Network checkpoint is loaded.
    Native default empirical model has no neighbor dependency for fixed leaves.
    """
    prior_gain,geometric,logit=0.,0.,0.
    for row,after in edits:
        before=int(seq[row,-1,0])
        parent=seq[row,-1,3:6].astype(np.int64)
        split=parts(before,int(after),parent)
        rem=[x[0] for x in split['Adjust']]+split['Prune']
        add=[x[1] for x in split['Adjust']]+split['Add']
        key=(1,)+tuple((parent//2).tolist())
        gain=model.cost(key,before)-model.cost(key,int(after))
        occupied=[parent+np.array([(s>>2)&1,(s>>1)&1,s&1]) for s in range(8) if before&(1<<s)]
        geom=0.
        for point in rem:
            others=[p for p in occupied if not np.array_equal(p,point)]
            geom+=4.*min((float(np.sum((np.asarray(point)-p)**2)) for p in others),default=4.)
        for point in add:
            geom+=4.*min(float(np.sum((np.asarray(point)-p)**2)) for p in occupied)
        count=sum(map(len,split.values()))
        prior_gain+=gain
        geometric+=geom
        logit+=float(np.tanh((gain-.05*geom)/(4.*count)))
    return dict(current_prior_score=logit,empirical_gain=prior_gain,proposal_geometry_proxy=geometric)


def protect_snapshot(root):
    root=Path(root)
    files=[root/'train.py']+list((root/'models').rglob('*.py'))
    codec=root.parent/'compress/octree'
    for name in ('G-PCC','OctAttention'):
        for suffix in ('*.py','*.cpp','*.h','*.cfg'):
            files+=list((codec/name).rglob(suffix))
    files += [codec/'G-PCC/build/tmc3/tmc3', codec/'OctAttention/modelsave/obj/encoder_epoch_00800093.pth']
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files)) if p.is_file()}
