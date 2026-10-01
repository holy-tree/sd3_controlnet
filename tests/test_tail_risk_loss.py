import unittest

import torch
import torch.nn.functional as F

from dpo.losses import diffusion_dpo_loss, flow_matching_gt_losses


class TailRiskLossTest(unittest.TestCase):
    def inputs(self, batch_size=3):
        return (
            torch.tensor([0.4, 1.2, 0.7][:batch_size], requires_grad=True),
            torch.tensor([1.4, 0.6, 0.9][:batch_size], requires_grad=True),
            torch.tensor([0.8, 0.9, 0.5][:batch_size], requires_grad=True),
            torch.tensor([1.0, 0.7, 1.2][:batch_size], requires_grad=True),
        )

    def test_default_matches_literal_legacy_value_and_gradients(self):
        for beta in (0.1, 0.7, 2.0):
            with self.subTest(beta=beta):
                inputs = self.inputs()
                chosen, rejected, ref_chosen, ref_rejected = inputs
                expected = -F.logsigmoid(
                    float(beta) * ((rejected - chosen) - (ref_rejected - ref_chosen))
                ).mean() + 0.3 * chosen.mean()
                actual, stats = diffusion_dpo_loss(*inputs, beta=beta, sft_weight=0.3)
                self.assertEqual(actual.shape, torch.Size([]))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual_grads = torch.autograd.grad(actual, inputs, retain_graph=True)
                expected_grads = torch.autograd.grad(expected, inputs)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
                self.assertFalse(stats["loss_dpo"].requires_grad)

    def test_external_all_one_weights_preserve_legacy_value_and_gradients(self):
        inputs = self.inputs()
        chosen, rejected, ref_chosen, ref_rejected = inputs
        beta, sft_weight = 0.4, 0.2
        expected = -F.logsigmoid(
            float(beta) * ((rejected - chosen) - (ref_rejected - ref_chosen))
        ).mean() + float(sft_weight) * chosen.mean()
        per_pair, _ = diffusion_dpo_loss(*inputs, beta=beta, reduction="none")
        actual = (per_pair * torch.ones(3)).mean() + sft_weight * chosen.mean()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_grads = torch.autograd.grad(actual, inputs, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_legacy_sample_weights_use_batch_mean_and_unweighted_sft(self):
        inputs = self.inputs()
        chosen, rejected, ref_chosen, ref_rejected = inputs
        weights = torch.tensor([0.5, 2.0, 3.0], dtype=torch.float64)
        losses = -F.logsigmoid(0.6 * ((rejected - chosen) - (ref_rejected - ref_chosen)))
        dpo = (losses * weights.to(losses)).mean()
        expected = dpo + 0.25 * chosen.mean()
        actual, stats = diffusion_dpo_loss(
            *inputs, beta=0.6, sample_weights=weights, sft_weight=0.25
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(stats["loss_dpo"], dpo.detach(), rtol=0, atol=0)
        actual_grads = torch.autograd.grad(actual, inputs, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_none_returns_pure_per_pair_losses_and_existing_mean_stats(self):
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                inputs = self.inputs(batch_size)
                chosen, rejected, ref_chosen, ref_rejected = inputs
                margin = rejected - chosen
                ref_margin = ref_rejected - ref_chosen
                logits = margin - ref_margin
                expected = -F.logsigmoid(0.7 * logits)
                actual, stats = diffusion_dpo_loss(*inputs, beta=0.7, reduction="none")
                self.assertEqual(actual.shape, (batch_size,))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected_stats = {
                    "loss_dpo": expected.mean(),
                    "policy_margin": margin.mean(),
                    "reference_margin": ref_margin.mean(),
                    "implicit_accuracy": (logits > 0).float().mean(),
                    "chosen_mse": chosen.mean(),
                    "rejected_mse": rejected.mean(),
                }
                self.assertEqual(stats.keys(), expected_stats.keys())
                for key, value in expected_stats.items():
                    self.assertEqual(stats[key].shape, torch.Size([]))
                    self.assertFalse(stats[key].requires_grad)
                    torch.testing.assert_close(stats[key], value.detach(), rtol=0, atol=0)

    def test_batch_one_weight_changes_loss_and_gradients_fourfold(self):
        inputs = self.inputs(1)
        per_pair, _ = diffusion_dpo_loss(*inputs, reduction="none")
        low = (per_pair * torch.tensor([0.5])).mean()
        high = (per_pair * torch.tensor([2.0])).mean()
        self.assertGreater(float(low.detach()), 0.0)
        torch.testing.assert_close(high, 4.0 * low, rtol=0, atol=0)
        low_grads = torch.autograd.grad(low, inputs, retain_graph=True)
        high_grads = torch.autograd.grad(high, inputs)
        for low_grad, high_grad in zip(low_grads, high_grads):
            self.assertGreater(float(low_grad.abs().sum()), 0.0)
            torch.testing.assert_close(high_grad, 4.0 * low_grad, rtol=0, atol=0)

    def test_external_sft_gradient_is_independent_of_pair_weights(self):
        inputs = self.inputs()
        chosen = inputs[0]
        per_pair, _ = diffusion_dpo_loss(*inputs, reduction="none")
        sft = 0.3 * chosen.mean()
        expected_grad = torch.full_like(chosen, 0.3 / chosen.numel())
        for weights in (torch.tensor([0.5, 0.5, 0.5]), torch.tensor([2.0, 2.0, 2.0])):
            with self.subTest(weights=weights.tolist()):
                dpo = (per_pair * weights).mean()
                total_grad = torch.autograd.grad(dpo + sft, chosen, retain_graph=True)[0]
                dpo_grad = torch.autograd.grad(dpo, chosen, retain_graph=True)[0]
                torch.testing.assert_close(total_grad - dpo_grad, expected_grad)

    def test_gt_flow_and_x0_gradients_are_independent_of_pair_weights(self):
        inputs = self.inputs()
        per_pair, _ = diffusion_dpo_loss(*inputs, reduction="none")
        prediction = torch.full((3, 1, 2, 2), 1.5, requires_grad=True)
        target = torch.ones_like(prediction)
        clean = torch.zeros_like(prediction)
        sigma = torch.tensor([0.2, 0.5, 0.8]).view(3, 1, 1, 1)
        noisy = sigma * target
        flow, x0 = flow_matching_gt_losses(prediction, target, noisy, clean, sigma)
        flow_grad = torch.autograd.grad(flow, prediction, retain_graph=True)[0]
        x0_grad = torch.autograd.grad(x0, prediction, retain_graph=True)[0]
        self.assertGreater(float(flow_grad.abs().sum()), 0.0)
        self.assertGreater(float(x0_grad.abs().sum()), 0.0)
        for auxiliary, expected_grad in ((0.1 * flow, 0.1 * flow_grad),
                                         (0.05 * x0, 0.05 * x0_grad)):
            for weight in (0.5, 2.0):
                with self.subTest(weight=weight, auxiliary=float(auxiliary.detach())):
                    total = (per_pair * weight).mean() + auxiliary
                    actual_grad = torch.autograd.grad(total, prediction, retain_graph=True)[0]
                    torch.testing.assert_close(actual_grad, expected_grad)

    def test_none_rejects_sample_weights_even_when_all_one(self):
        with self.assertRaisesRegex(ValueError, "sample_weights.*reduction='none'"):
            diffusion_dpo_loss(
                *self.inputs(), sample_weights=torch.ones(3), reduction="none"
            )

    def test_none_rejects_nonzero_sft_weight(self):
        for sft_weight in (0.1, -0.1):
            with self.subTest(sft_weight=sft_weight):
                with self.assertRaisesRegex(ValueError, "sft_weight.*reduction='none'"):
                    diffusion_dpo_loss(
                        *self.inputs(), sft_weight=sft_weight, reduction="none"
                    )

    def test_invalid_reduction_is_rejected(self):
        for reduction in ("sum", "invalid", "", None):
            with self.subTest(reduction=reduction):
                with self.assertRaisesRegex(ValueError, "reduction must be"):
                    diffusion_dpo_loss(*self.inputs(), reduction=reduction)

    def test_nonpositive_beta_is_still_rejected(self):
        for reduction in ("mean", "none"):
            for beta in (0.0, -0.1):
                with self.subTest(reduction=reduction, beta=beta):
                    with self.assertRaisesRegex(ValueError, "beta must be positive"):
                        diffusion_dpo_loss(*self.inputs(), beta=beta, reduction=reduction)


if __name__ == "__main__":
    unittest.main()
