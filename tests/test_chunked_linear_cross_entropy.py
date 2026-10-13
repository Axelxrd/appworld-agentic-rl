"""CPU correctness checks for verl's chunked linear cross-entropy path."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_chunked_log_probs_match_full_logits_and_gradients() -> None:
    pytest.importorskip("verl")
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    torch.manual_seed(7)
    hidden = torch.randn(2, 11, 13, requires_grad=True)
    weight = torch.randn(29, 13, requires_grad=True)
    labels = torch.randint(0, weight.shape[0], (2, 11))

    fused = FusedLinearForPPO(chunk_size=3)
    fused_log_probs, _ = fused(hidden, weight, labels)
    fused_loss = -fused_log_probs.mean()
    fused_loss.backward()
    fused_hidden_grad = hidden.grad.detach().clone()
    fused_weight_grad = weight.grad.detach().clone()

    full_hidden = hidden.detach().clone().requires_grad_(True)
    full_weight = weight.detach().clone().requires_grad_(True)
    full_log_probs = torch.nn.functional.log_softmax(full_hidden @ full_weight.t(), dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    full_loss = -full_log_probs.mean()
    full_loss.backward()

    torch.testing.assert_close(fused_log_probs, full_log_probs, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(fused_hidden_grad, full_hidden.grad, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(fused_weight_grad, full_weight.grad, rtol=5e-4, atol=5e-4)
