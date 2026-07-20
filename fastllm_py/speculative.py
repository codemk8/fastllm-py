"""Greedy speculative decoding: a small draft model proposes, the target model
verifies γ tokens in one forward pass.

Output is IDENTICAL to greedy decoding with the target alone (the target's
argmax is always what gets emitted), so it's a pure latency optimization —
whenever the draft agrees, the target advances several tokens per forward pass.

State per round: both KV caches hold the same P committed tokens, and `cur` is
the last emitted token (position P, not yet in either KV).
  1. Draft rolls out γ candidates t1..tγ (γ cheap draft forwards).
  2. Target verifies with ONE forward over [cur, t1, ..., t_{γ-1}] (γ tokens),
     giving its own argmax at each position.
  3. Accept the longest prefix where target's argmax == the draft token; emit
     those, plus the target's own token at the first disagreement (the
     correction). Roll both KV caches back to the accepted length.

Both models must share a tokenizer / vocabulary.
"""
from __future__ import annotations

import numpy as np

from .model import KVCache, Model


def _argmax_last(logits, xp_asnumpy):
    row = logits[-1]
    return int(np.argmax(xp_asnumpy(row)))


class SpeculativeDecoder:
    def __init__(self, target: Model, draft: Model, gamma: int = 4):
        if target.cfg.vocab_size != draft.cfg.vocab_size:
            raise ValueError("target and draft must share a vocabulary")
        self.target = target
        self.draft = draft
        self.gamma = gamma
        import cupy as cp

        self._np = lambda a: cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)

    def generate(self, prompt_ids, max_new_tokens: int = 128):
        np_ = self._np
        ids = np.asarray(prompt_ids, dtype=np.int64)

        t_log, t_kv = self.target.forward(ids)
        d_log, d_kv = self.draft.forward(ids)
        P = len(ids)                       # committed length in both KV caches
        cur = _argmax_last(t_log, np_)     # first target token (position P)
        out = [cur]
        self.stats = {"rounds": 0, "target_forwards": 1, "accepted": 0,
                      "proposed": 0}

        while len(out) < max_new_tokens:
            self.stats["rounds"] += 1
            g = min(self.gamma, max_new_tokens - len(out))

            # 1. draft rollout: forward cur, t1, ..., t_{g-1}; propose t1..tg
            proposals, x = [], cur
            for _ in range(g):
                d_log, d_kv = self.draft.forward(np.array([x]), d_kv)
                x = _argmax_last(d_log, np_)
                proposals.append(x)
            self.stats["proposed"] += g

            # 2. target verifies [cur, t1, ..., t_{g-1}] in one forward
            verify = np.array([cur] + proposals[:-1], dtype=np.int64)
            t_log, t_kv = self.target.forward(verify, t_kv)
            self.stats["target_forwards"] += 1
            tlog = np_(t_log)                         # (g, vocab)
            tpred = tlog.argmax(-1).tolist()          # target argmax at each pos

            # 3. accept the longest matching prefix
            n = 0
            while n < g and proposals[n] == tpred[n]:
                n += 1
            self.stats["accepted"] += n

            # Both KV caches hold P + g tokens (cur, t1..t_{g-1}) after the
            # verify/rollout. Emit + roll back:
            if n < g:  # first disagreement at n -> emit t1..tn + target's token
                emit = proposals[:n] + [tpred[n]]   # correction not yet in KV
                keep = P + 1 + n                    # cur, t1..tn are in KV
                cur = tpred[n]
            else:      # all g matched -> emit t1..tg (no bonus without another fwd)
                emit = list(proposals)
                keep = P + g                        # cur, t1..t_{g-1} in KV
                cur = proposals[-1]                 # tg, forwarded next round
            out.extend(emit)
            for c in t_kv:
                c.truncate(keep)
            for c in d_kv:
                c.truncate(keep)
            P = keep

        return out[:max_new_tokens]
