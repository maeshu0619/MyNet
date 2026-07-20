import ast
import gzip
import inspect
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import models.network as network_module
from models.modules.network_k_proposal_policy import NetworkKProposalPolicy
from models.utils.loss.k_proposal_distillation import (
    KProposalSetLoss,
    OfflineKProposalTeacherStore,
)


def _args(**overrides):
    values = dict(
        sparsepcgc_scale_ae=0,
        sparsepcgc_scale_sr=2,
        sparsepcgc_scale_m=8,
        sparsepcgc_voxel_size=1.0,
        sparsepcgc_pos_quantscale=1.0,
        sparsepcgc_psnr_resolution=1023,
        sparsepcgc_native_bit_depth=10,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class NetworkKProposalPolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.policy = NetworkKProposalPolicy(
            25, hidden_dim=16, proposal_count=8, shortlist_size=256
        ).eval()
        self.features = torch.randn(1, 25, 4096)
        self.fixed = torch.rand(1, 6, 4096)
        self.coords = torch.randint(0, 512, (1, 3, 4096))

    def _forward(self):
        return self.policy(
            self.features, _args(), training=False,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )

    def test_inference_module_has_no_forbidden_dependency(self):
        source = inspect.getsource(NetworkKProposalPolicy)
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        forbidden = ("den4", "den5", "den6", "teacher", "cache", "actual_encoder")
        self.assertFalse(any(any(term in name for term in forbidden) for name in imports))

    def test_deterministic_specialized_slots_and_counter_contract(self):
        first = self._forward()
        second = self._forward()
        for key in ("slot_logits", "total_ratio", "shares", "priorities", "critic_score"):
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertTrue(torch.equal(first["selected_slot"], second["selected_slot"]))
        masks = first["compact_plans"]["selected_shortlist_mask"][0]
        hashes = {bytes(mask.detach().cpu().numpy()) for mask in masks}
        self.assertGreater(len(hashes), 1)
        expected = {
            "shared_encoder_forward_count": 1,
            "shared_basis_forward_count": 1,
            "proposal_count": 8,
            "critic_batch_count": 1,
            "selected_plan_count": 1,
            "den6_call_count": 0,
            "cache_reference_count": 0,
            "teacher_reference_count": 0,
            "sparsepcgc_probe_count": 0,
            "candidate_actual_encode_count": 0,
        }
        for key, value in expected.items():
            self.assertEqual(first[key], value, key)

    def test_one_token_changes_its_mode_not_other_slots(self):
        before = self._forward()
        with torch.no_grad():
            self.policy.plan_tokens[3].add_(2.0)
        after = self._forward()
        unchanged = [
            torch.equal(before["slot_logits"][:, slot], after["slot_logits"][:, slot])
            for slot in range(8) if slot != 3
        ]
        self.assertTrue(all(unchanged))
        self.assertFalse(torch.equal(before["slot_logits"][:, 3], after["slot_logits"][:, 3]))

    def test_codec_conditioning_changes_proposal_set(self):
        first = self._forward()["slot_logits"]
        second = self.policy(
            self.features,
            _args(sparsepcgc_scale_ae=1, sparsepcgc_scale_sr=0),
            training=False,
            fixed_features=self.fixed,
            voxel_coords=self.coords,
        )["slot_logits"]
        self.assertFalse(torch.equal(first, second))

    def test_selected_amount_matches_collision_resolved_compact_counts(self):
        output = self._forward()
        selected = int(output["selected_slot"][0])
        expected = output["compact_plans"]["accepted_count"][0, selected].long()
        terms = output["selected_policy_terms"]
        actual = torch.round(
            terms["total_ratio"][0, 0, 0]
            * terms["shares"][0, :, 0]
            * float(self.features.shape[-1])
        ).long()
        self.assertTrue(torch.equal(actual, expected), (actual, expected))

    def test_forbidden_runtime_functions_are_not_reached(self):
        error = RuntimeError("forbidden path reached")
        with mock.patch.object(network_module, "attach_ana_den6_online_guidance", side_effect=error), \
             mock.patch.object(network_module, "attach_ana_den6_reference_anchor", side_effect=error), \
             mock.patch.object(network_module, "build_heuristic_guidance", side_effect=error):
            result = self._forward()
        self.assertEqual(result["proposal_count"], 8)

    def test_actual_reward_reverses_k_critic_selection_gradient(self):
        self.policy.train()
        output = self.policy(
            self.features, _args(), training=True,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        log_probability = output["critic_selection_log_prob"]
        parameter = self.policy.critic_gain_head.weight
        improving = torch.autograd.grad(
            -(+1.0) * log_probability, parameter, retain_graph=True
        )[0]
        worsening = torch.autograd.grad(-(-1.0) * log_probability, parameter)[0]
        self.assertGreater(float(improving.abs().sum()), 0.0)
        self.assertTrue(torch.allclose(improving, -worsening, atol=1e-7, rtol=1e-5))

    def test_set_loss_updates_proposal_and_critic_heads(self):
        self.policy.train()
        output = self.policy(
            self.features, _args(), training=True,
            fixed_features=self.fixed, voxel_coords=self.coords,
        )
        descriptor = output["compact_plans"]["descriptor"].detach()
        teacher = {
            "mode_descriptor": descriptor.roll(1, dims=1),
            "actual_gain": torch.linspace(-0.5, 4.0, 8).view(1, 8),
            "geometry": torch.linspace(0.1, 0.8, 8).view(1, 8),
            "mode_mask": torch.ones(1, 8, dtype=torch.bool),
            "voxel_relative_value": torch.rand(1, 3, output["slot_logits"].shape[-1]),
        }
        losses = KProposalSetLoss()(output, teacher)
        self.assertTrue(losses["total"].requires_grad)
        losses["total"].backward()
        groups = {
            "token": self.policy.plan_tokens.grad,
            "amount": self.policy.amount_head.weight.grad,
            "share": self.policy.share_head.weight.grad,
            "where": self.policy.shared_basis_head.weight.grad,
            "critic": self.policy.critic_gain_head.weight.grad,
        }
        for name, gradient in groups.items():
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.detach().abs().sum()), 0.0, name)
        self.assertEqual(set(losses["raw"]), set(KProposalSetLoss.DEFAULT_WEIGHTS))

    def test_offline_teacher_join_is_explicit_and_add_target_is_not_fabricated(self):
        output = self._forward()
        descriptor = output["compact_plans"]["descriptor"][0, 0].tolist()
        shortlist_index = int(output["shortlist_indices"][0, 0])
        coord = self.coords[0, :, shortlist_index].tolist()
        mode = {
            "plan_key": "diagnostic",
            "descriptor": descriptor,
            "actual_gain_percent": 2.0,
            "geometry": {"D1_loss_db": 0.2, "D2_loss_db": 0.3},
            "member_count": 1,
        }
        payload = {
            "schema_version": "mynet_kproposal_mode_dataset_v1",
            "offline_only": True,
            "voxel_target_semantics": "rank_weighted_relative_value_not_causal_gain",
            "split": {"train": "sha|setting"},
            "states": {
                "sha|setting": {
                    "mode_medoids": [mode],
                    "voxel_relative_values": [
                        {"operation": "Prune", "coord": coord, "rank_weighted_relative_value": 0.9},
                        {"operation": "Add", "coord": coord, "rank_weighted_relative_value": 0.9},
                    ],
                }
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".json.gz") as stream:
            with gzip.open(stream.name, "wt", encoding="utf-8") as compressed:
                json.dump(payload, compressed)
            store = OfflineKProposalTeacherStore(stream.name)
            teacher = store.teacher_for_output(
                "sha|setting", output, self.coords, split="train"
            )
        self.assertTrue(bool(teacher["voxel_value_mask"][0, 0].any()))
        self.assertFalse(bool(teacher["voxel_value_mask"][0, 1].any()))
        self.assertFalse(teacher["add_where_teacher_available"])
        self.assertEqual(teacher["voxel_target_semantics"], "rank_weighted_relative_value_not_causal_gain")


if __name__ == "__main__":
    unittest.main()
