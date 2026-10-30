import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gradient_accumulation import accumulated_step


class GradientAccumulationTests(unittest.TestCase):
    def compare_update(self, rows, microbatch):
        torch.manual_seed(42)
        full = torch.nn.Sequential(torch.nn.Linear(7, 9), torch.nn.LayerNorm(9),
                                   torch.nn.GELU(), torch.nn.Linear(9, 5)).double()
        full[0].weight.requires_grad_(False)
        full[0].bias.requires_grad_(False)
        micro = copy.deepcopy(full)
        samples = torch.randn(rows, 7, dtype=torch.float64)
        targets = torch.randint(0, 5, (rows,))
        criterion = torch.nn.CrossEntropyLoss()
        full_optimizer = torch.optim.AdamW([p for p in full.parameters() if p.requires_grad], lr=0.001)
        micro_optimizer = torch.optim.AdamW([p for p in micro.parameters() if p.requires_grad], lr=0.001)
        full_outputs = full(samples)
        full_loss = criterion(full_outputs, targets)
        full_loss.backward()
        full_optimizer.step()
        guard = Mock()
        validate = Mock()
        with patch.object(micro_optimizer, "step", wraps=micro_optimizer.step) as step:
            loss, logits = accumulated_step(micro, criterion, micro_optimizer, samples, targets,
                "cpu", microbatch, before_microbatch=guard, before_optimizer_step=validate)
            self.assertEqual(step.call_count, 1)
        self.assertEqual(guard.call_count, (rows + microbatch - 1) // microbatch)
        self.assertEqual(validate.call_count, 1)
        self.assertAlmostEqual(loss.item(), full_loss.item(), places=12)
        torch.testing.assert_close(logits, full_outputs.detach(), rtol=1e-11, atol=1e-12)
        for p_full, p_micro in zip(full.parameters(), micro.parameters()):
            torch.testing.assert_close(p_full, p_micro, rtol=1e-10, atol=1e-12)
            if p_full.requires_grad:
                torch.testing.assert_close(p_full.grad, p_micro.grad, rtol=1e-10, atol=1e-12)
            else:
                self.assertIsNone(p_micro.grad)

    def test_effective_batch64_microbatch8(self):
        self.compare_update(64, 8)

    def test_uneven_final_microbatch_weighting(self):
        self.compare_update(19, 8)

    def test_invalid_batch_does_not_update(self):
        model = torch.nn.Linear(7, 5)
        optimizer = torch.optim.AdamW(model.parameters())
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaises(ValueError):
                accumulated_step(model, torch.nn.CrossEntropyLoss(), optimizer,
                                 torch.empty(0, 7), torch.empty(0, dtype=torch.long), "cpu", 8)
            self.assertEqual(step.call_count, 0)

    def test_nonfinite_loss_does_not_update(self):
        model = torch.nn.Linear(7, 5)
        optimizer = torch.optim.AdamW(model.parameters())
        criterion = lambda logits, targets: logits.sum() * float("nan")
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaises(RuntimeError):
                accumulated_step(model, criterion, optimizer, torch.randn(8, 7),
                                 torch.zeros(8, dtype=torch.long), "cpu", 8)
            self.assertEqual(step.call_count, 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
