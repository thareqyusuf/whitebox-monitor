"""A model wrapper that makes the tiny random model emit a scripted reply.

It adds a logit bonus to scripted tokens at scripted steps of the unforced
generation, so the greedy path through the real backend code (cache, hooks,
parsing, score-step lookup) can be tested with a known answer. The bonus is
finite, so distributions stay real softmax outputs.
"""

from __future__ import annotations

import torch


class ScriptedLM(torch.nn.Module):
    def __init__(self, inner, script_ids, bonus: float = 40.0):
        super().__init__()
        self.inner = inner
        self.model = inner.model          # embed_tokens / layers live here
        self.config = inner.config
        self.script_ids = list(script_ids)
        self.bonus = bonus
        self.n_prompt = None

    def forward(self, input_ids, past_key_values=None, use_cache=True, logits_to_keep=None, **kw):
        out = self.inner(input_ids, past_key_values=past_key_values, use_cache=use_cache, **kw)
        seen = input_ids.shape[1]
        if past_key_values is None:
            self.n_prompt = seen
            step = 0
        else:
            step = out.past_key_values.get_seq_length() - self.n_prompt
        if 0 <= step < len(self.script_ids):
            out.logits[0, -1, self.script_ids[step]] += self.bonus
        return out
