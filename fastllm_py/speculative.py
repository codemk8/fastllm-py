"""Speculative decoding: a small draft model proposes, the target verifies γ
tokens in one forward pass.

Greedy mode (temperature<=0): output is IDENTICAL to greedy decoding with the
target alone (the target's argmax is always what gets emitted) — a pure latency
optimization. Sampling mode (temperature>0): rejection sampling (Leviathan et
al. / Chen et al.) makes the output distribution EXACTLY the target's own
truncated sampling distribution (same temperature/top_p/top_k), again a pure
latency optimization — quality is unchanged, only speed improves.

State per round: both KV caches hold the same P committed tokens, and `cur` is
the last emitted token (position P, not yet in either KV).
  1. Draft rolls out γ candidates t1..tγ (γ cheap draft forwards); sampling mode
     also records each draft distribution q_i.
  2. Target verifies with ONE forward over [cur, t1, ..., t_{γ-1}] (γ tokens),
     giving its distribution p_i at each position.
  3. Accept: greedy compares argmax; sampling accepts t_i w.p. min(1,p_i/q_i)
     and draws the correction from (p_i - q_i)+ on the first reject. Emit the
     accepted prefix + correction; roll both KV caches back to that length.

Both models must share a tokenizer / vocabulary.
"""
from __future__ import annotations

import numpy as np

from .model import KVCache, Model


def _argmax_last(logits, xp_asnumpy):
    row = logits[-1]
    return int(np.argmax(xp_asnumpy(row)))


def rejection_step(p, q, x, rng):
    """One speculative-sampling accept/reject decision (Leviathan et al. /
    Chen et al.). x was drawn from draft distribution q; p is the target
    distribution at the same position. Returns (accepted, correction):
      * accepted=True  -> commit x (correction is None)
      * accepted=False -> commit `correction`, drawn from the residual
        (p - q)+ normalized, and stop the round here.
    Guarantees the committed token is distributed exactly as p, for any q with
    q(x) > 0 (which holds since x ~ q)."""
    if rng.random() < min(1.0, p[x] / q[x]):
        return True, None
    resid = np.clip(p - q, 0.0, None)
    s = resid.sum()
    corr = int(rng.choice(p.size, p=(resid / s) if s > 0 else p))
    return False, corr


class SpeculativeDecoder:
    def __init__(self, target: Model, draft: Model, gamma: int = 4,
                 draft_graph: bool = True, draft_max_len: int = 4096):
        if target.cfg.vocab_size != draft.cfg.vocab_size:
            raise ValueError("target and draft must share a vocabulary")
        self.target = target
        self.draft = draft
        self.gamma = gamma
        import cupy as cp

        self._np = lambda a: cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)

        # graph-accelerate the draft rollout (the hot loop): gamma cheap
        # single-token steps per round is exactly GraphDecoder's best case.
        self.draft_gd = None
        if draft_graph:
            try:
                from .graph_decode import GraphDecoder, graph_capable

                if graph_capable(draft):
                    gd = GraphDecoder(draft, max_len=draft_max_len)
                    gd.capture()
                    self.draft_gd = gd
            except Exception:
                self.draft_gd = None

    def generate(self, prompt_ids, max_new_tokens: int = 128, stop_ids=None,
                 on_token=None, temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = 0, seed=None):
        """Speculative decode. temperature<=0 is greedy (output identical to
        greedy target decoding); temperature>0 uses rejection sampling (Leviathan
        et al. / Chen et al.) so the output distribution is exactly the target's
        own truncated sampling distribution at the same temperature/top_p/top_k.
        stop_ids: halt once one is emitted. on_token(tok): streaming callback."""
        from .graph_decode import logits_to_probs

        np_ = self._np
        ids = np.asarray(prompt_ids, dtype=np.int64)
        stop = set(stop_ids) if stop_ids else None
        sampling = temperature > 0.0
        rng = np.random.default_rng(seed) if sampling else None

        def _tprobs(logits_row):
            return logits_to_probs(logits_row, temperature, top_p, top_k)

        def _pick(logits_row):  # target/draft token draw (last-row logits)
            if not sampling:
                return int(np.argmax(logits_row))
            p = _tprobs(logits_row)
            return int(rng.choice(p.size, p=p)), p

        def _emit(tok):
            if on_token is not None:
                on_token(tok)
            return stop is not None and tok in stop

        t_log, t_kv = self.target.forward(ids)
        if self.draft_gd is not None:
            self.draft_gd.prime(ids)       # fills the draft graph's KV buffers
            d_kv = None
        else:
            d_log, d_kv = self.draft.forward(ids)
        P = len(ids)                       # committed length in both KV caches
        if sampling:
            cur = int(rng.choice(t_log.shape[-1], p=_tprobs(np_(t_log)[-1])))
        else:
            cur = _argmax_last(t_log, np_)  # first target token (position P)
        out = [cur]
        self.stats = {"rounds": 0, "target_forwards": 1, "accepted": 0,
                      "proposed": 0}
        if _emit(cur):
            return out

        while len(out) < max_new_tokens:
            self.stats["rounds"] += 1
            g = min(self.gamma, max_new_tokens - len(out))

            # 1. draft rollout: from cur, propose g tokens (writes cur..t_{g-1}).
            #    Sampling mode also records each draft distribution q_i.
            proposals, qprobs, x, dpos = [], [], cur, P
            for _ in range(g):
                if self.draft_gd is not None:
                    dl = self.draft_gd.step(x, dpos)
                else:
                    d_log, d_kv = self.draft.forward(np.array([x]), d_kv)
                    dl = np_(d_log)[-1]
                if sampling:
                    x, q = _pick(dl)
                    qprobs.append(q)
                else:
                    x = int(np.argmax(dl))
                dpos += 1
                proposals.append(x)
            self.stats["proposed"] += g

            # 2. target verifies [cur, t1, ..., t_{g-1}] in one forward
            verify = np.array([cur] + proposals[:-1], dtype=np.int64)
            t_log, t_kv = self.target.forward(verify, t_kv)
            self.stats["target_forwards"] += 1
            tlog = np_(t_log)                         # (g, vocab)

            # 3. acceptance. Greedy: longest exact-match prefix. Sampling:
            #    accept x_i w.p. min(1, p_i(x_i)/q_i(x_i)); on first reject draw
            #    the correction from the residual (p_i - q_i)+ (normalized).
            if sampling:
                n, correction = g, None
                for i in range(g):
                    accepted, corr = rejection_step(
                        _tprobs(tlog[i]), qprobs[i], proposals[i], rng)
                    if accepted:
                        continue
                    n, correction = i, corr
                    break
            else:
                tpred = tlog.argmax(-1).tolist()
                n = 0
                while n < g and proposals[n] == tpred[n]:
                    n += 1
                correction = tpred[n] if n < g else None
            self.stats["accepted"] += n

            # Both KV caches hold P + g tokens after the verify/rollout.
            # Emit + roll back (identical bookkeeping for both modes):
            if n < g:  # first reject/mismatch at n -> emit t1..tn + correction
                emit = proposals[:n] + [correction]  # correction not yet in KV
                keep = P + 1 + n                     # cur, t1..tn are in KV
                cur = correction
            else:      # all g accepted -> emit t1..tg (no bonus without a fwd)
                emit = list(proposals)
                keep = P + g                         # cur, t1..t_{g-1} in KV
                cur = proposals[-1]                  # tg, forwarded next round
            for c in t_kv:
                c.truncate(keep)
            if self.draft_gd is not None:
                self.draft_gd.truncate(keep)
            else:
                for c in d_kv:
                    c.truncate(keep)
            P = keep

            # commit + stream the accepted tokens (respect the budget + stop)
            for tok in emit:
                if len(out) >= max_new_tokens:
                    break
                out.append(tok)
                if _emit(tok):
                    return out[:max_new_tokens]

        return out[:max_new_tokens]
