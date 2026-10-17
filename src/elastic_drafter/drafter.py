"""A PARD parallel drafter: it proposes k tokens in one forward pass.

PARD drafters are ordinary causal language models fine-tuned with a reserved placeholder token
(`pard_token` in the model's config). To draft k tokens, the input is the text so far followed by
k - 1 placeholders; the logits at the last real token give the 1st draft token and those at the
placeholders give the 2nd to k-th (PARD, https://arxiv.org/abs/2504.18583; reference code in
https://github.com/AMD-AIG-AIMA/PARD, pard/pard2_infer.py).

The PARD-2 checkpoints also ship `warp_model.bin`, a projection that adds the target's hidden
states to the drafter's input (their target-dependent mode). Loading the model without it, as here,
is their target-independent mode: the drafter sees only tokens.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM


class Drafter:
    def __init__(self, path, device=None, dtype=torch.bfloat16):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        self.mask_id = AutoConfig.from_pretrained(path).pard_token
        self.device = device

    def logits(self, context_ids, k):
        """Logits for the next k positions after `context_ids` (a list of token IDs), shape [k, vocab].
        Gradients flow when the caller enables them."""
        ids = torch.tensor([list(context_ids) + [self.mask_id] * (k - 1)], device=self.device)
        return self.model(input_ids=ids).logits[0, -k:].float()

    @torch.no_grad()
    def draft(self, context_ids, k):
        """The k tokens the drafter proposes after `context_ids`."""
        return self.logits(context_ids, k).argmax(-1).tolist()
