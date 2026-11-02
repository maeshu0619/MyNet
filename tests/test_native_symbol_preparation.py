from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.native_symbol_preparation import ExactCPUMembership, ExactTensorMembership, sorted_symbol_view


def vector(rows, step):
    rows, step = rows.long(), step.long()
    return sum(rows[:,i]*(step**i) for i in range(rows.shape[-1]))


def reference(data, gt):
    if not len(gt):
        return torch.zeros(len(data),dtype=torch.bool,device=data.device)
    step=torch.max(data.max(),gt.max())+1
    return torch.isin(vector(data,step),vector(gt,step))


class NativeSymbolPreparationTest(unittest.TestCase):
    def test_tensor_keys_preserve_native_negative_overflow_and_cache_semantics(self):
        torch.manual_seed(81)
        for dtype in (torch.int32,torch.int64):
            for maximum in (10,1000,2**30):
                gt=torch.randint(-maximum,maximum,(100,4),dtype=dtype)
                query=torch.cat([gt[:20],torch.randint(-maximum,maximum,(100,4),dtype=dtype)])
                fast=ExactTensorMembership(reference,vector,verify=True)
                for _ in range(2):self.assertTrue(torch.equal(fast(query,gt),reference(query,gt)))
                self.assertEqual(fast.counts['hits'],1)
                gt[0,0]+=1
                self.assertTrue(torch.equal(fast(query,gt),reference(query,gt)))

    def test_matches_native_keys_with_duplicates_and_outside_queries(self):
        rng=np.random.RandomState(89)
        for dimension in (1,2,3,4):
            for dtype in (torch.int32,torch.int64):
                for maximum in (1,5,1023,32767):
                    gt=torch.as_tensor(rng.randint(0,maximum+1,(300,dimension)),dtype=dtype)
                    queries=torch.cat([gt[:80],torch.as_tensor(rng.randint(0,maximum+2,(250,dimension)),dtype=dtype)])
                    fast=ExactCPUMembership(reference,verify=True)
                    self.assertTrue(torch.equal(fast(queries,gt),reference(queries,gt)))
                    self.assertEqual(fast.counts['fast'],1)

    def test_cache_tracks_tensor_identity_and_mutation(self):
        gt=torch.tensor([[0,1,2,3],[0,4,5,6]],dtype=torch.int32)
        query=gt.clone();fast=ExactCPUMembership(reference)
        self.assertTrue(fast(query,gt).all())
        fast(query,gt);self.assertEqual(fast.counts['hits'],1)
        gt[0,1]=2
        self.assertTrue(torch.equal(fast(query,gt),reference(query,gt)))
        self.assertFalse(bool(fast(query,gt)[0]))
        other=gt.clone();other[1,1]=0
        self.assertTrue(torch.equal(fast(query,other),reference(query,other)))
        for _ in range(10):fast(query,gt.clone())
        self.assertLessEqual(len(fast.cache),4)
        fast.clear();self.assertEqual(len(fast.cache),0)

    def test_empty_negative_noninteger_and_large_inputs_use_native(self):
        for gt in (torch.empty((0,3),dtype=torch.int64),
                   torch.tensor([[-1,2,3]],dtype=torch.int64),
                   torch.tensor([[32768,2,3]],dtype=torch.int64),
                   torch.tensor([[1.,2.,3.]])):
            q=torch.tensor([[0,2,3]],dtype=gt.dtype)
            fast=ExactCPUMembership(reference)
            self.assertTrue(torch.equal(fast(q,gt),reference(q,gt)))
            self.assertEqual(fast.counts['fallback'],1)

    def test_untracked_inference_tensor_does_not_reuse_stale_keys(self):
        # Inference tensors do not expose the mutation version counter.
        with torch.inference_mode():
            gt=torch.tensor([[0,0,0],[1,1,1]])
            q=gt.clone();fast=ExactCPUMembership(reference)
            fast(q,gt);gt[0,0]=1
            self.assertTrue(torch.equal(fast(q,gt),reference(q,gt)))

    def test_zero_maximum_and_noncontiguous_rows(self):
        gt=torch.zeros((4,4),dtype=torch.int32);q=torch.eye(4,dtype=torch.int32)
        fast=ExactCPUMembership(reference)
        self.assertTrue(torch.equal(fast(q,gt),reference(q,gt)))
        gt=torch.arange(120,dtype=torch.int32).reshape(10,12)[:,::3]
        q=gt.flip(0)
        self.assertTrue(torch.equal(fast(q,gt),reference(q,gt)))

    def test_symbol_view_preserves_native_sort_rows_features_and_sigmoid(self):
        torch.manual_seed(16)
        coordinates=torch.cartesian_prod(torch.arange(2),torch.arange(5),torch.arange(3))
        permutation=torch.randperm(len(coordinates))
        tensor=SimpleNamespace(C=coordinates[permutation],F=torch.randn(len(coordinates),1))
        before=tensor.F.clone();order=torch.argsort(vector(tensor.C,tensor.C.max()+1))
        result=sorted_symbol_view(tensor,vector)
        self.assertTrue(torch.equal(result.C,tensor.C[order]))
        self.assertTrue(torch.equal(result.F,tensor.F[order]))
        self.assertTrue(torch.equal(torch.sigmoid(result.F),torch.sigmoid(tensor.F[order])))
        self.assertTrue(torch.equal(tensor.F,before))


if __name__=='__main__':unittest.main()
