"""OrbitNet: an AlphaStar-inspired policy network for Orbit Wars.

Pipeline (see implmentation.md):
  1. Embed every planet/comet feature vector.
  2. Self-attention encoder so each planet is aware of the whole board.
  3. Masked attention: each *owned* planet (query) scores every planet (key)
     to choose a target, plus a learned "no-op" key meaning "send nothing".
  4. An MLP head regresses the number of ships to send (conditioned on the
     chosen target, since the amount depends on the target's garrison).

Pointer-network machinery from AlphaStar is dropped (the planet set is small
and roughly fixed); we only need to pick a target and a ship count, exactly as
implmentation.md proposes.

Shapes use a batch dimension B so the same module serves training and
inference (inference just uses B=1).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import FEATURE_DIM, MAX_PLANETS

NEG_INF = -1e9


class OrbitNet(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model

        self.embed = nn.Sequential(
            nn.Linear(FEATURE_DIM, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Target-selection attention (query = owned planet, key = any planet).
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        # Launch gate: per owned planet, "do I launch this turn at all?". Kept
        # separate from target selection so the launch decision (a single logit)
        # doesn't have to out-score a launch probability fragmented across every
        # candidate target (which made the old shared-softmax NOOP dominate).
        self.gate_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # Ship head: regresses the FRACTION of the source garrison to send (in
        # [0,1] via sigmoid), conditioned on the (source, target) pair -- humans
        # overwhelmingly "send everything", so a bounded fraction is far easier
        # to fit than the 1..24000 absolute count.
        self.ship_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # RL-only ship head: a Beta(alpha, beta) distribution over the SAME [0,1]
        # fraction, so the ship amount can be SAMPLED and trained by policy gradient.
        # (The sigmoid ship_head above stays for SL + greedy deployment; this one is
        # used by act_rl / main_rl.) alpha,beta>=1 via 1+softplus keeps the density
        # proper; the final bias leans the initial mean high ("send most"), matching
        # the SL prior so early RL exploration doesn't randomise ship amounts.
        self.ship_beta_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
        with torch.no_grad():
            self.ship_beta_head[-1].bias.copy_(torch.tensor([2.0, -1.0]))  # mean ~0.7

        # Value head (RL critic): a board summary -> expected game outcome in
        # [-1, 1] (win-ness). Unused by SL; trained by regression to the return.
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def value(self, h, valid_mask):
        """Critic estimate per board: mean-pool the (valid) planet embeddings and
        map to a scalar in [-1, 1]. h [B,N,D], valid_mask [B,N] -> [B]."""
        mask = valid_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)   # [B,D]
        return torch.tanh(self.value_head(pooled).squeeze(-1))            # [B]

    def forward(self, feats, valid_mask, mine_mask):
        """
        feats      [B, N, FEATURE_DIM] float
        valid_mask [B, N] bool  (real planet)
        mine_mask  [B, N] bool  (owned by me -> a valid source/query)

        Returns:
          target_logits [B, N, N]  per source, score for each target (given launch)
          gate_logits   [B, N]     per source, launch-or-not logit
          h             [B, N, D]  planet embeddings (for the ship head)
        """
        B, N, _ = feats.shape

        h = self.embed(feats)                                   # [B, N, D]
        pad_mask = ~valid_mask                                  # True = ignore
        h = self.encoder(h, src_key_padding_mask=pad_mask)      # [B, N, D]

        q = self.q_proj(h)                                      # [B, N, D]
        k = self.k_proj(h)                                      # [B, N, D]
        scale = 1.0 / math.sqrt(self.d_model)

        target_logits = torch.matmul(q, k.transpose(1, 2)) * scale   # [B, N, N]
        # Mask invalid targets: padded planets, and a planet targeting itself.
        eye = torch.eye(N, dtype=torch.bool, device=feats.device).unsqueeze(0)
        invalid_tgt = (~valid_mask).unsqueeze(1) | eye          # [B, N, N]
        target_logits = target_logits.masked_fill(invalid_tgt, NEG_INF)

        gate_logits = self.gate_head(h).squeeze(-1)             # [B, N]

        return target_logits, gate_logits, h

    def ship_fraction(self, h, b_idx, src_idx, tgt_idx):
        """Predicted fraction of the source garrison to send, in [0,1], for given
        source/target pairs.

        h        [B, N, D]
        b_idx, src_idx, tgt_idx : 1-D index tensors selecting (batch, src, tgt).
        Multiply by the source's available ships for an absolute count (see act).
        """
        src_emb = h[b_idx, src_idx]                  # [M, D]
        tgt_emb = h[b_idx, tgt_idx]                  # [M, D]
        pair = torch.cat([src_emb, tgt_emb], dim=-1)  # [M, 2D]
        return torch.sigmoid(self.ship_head(pair).squeeze(-1))   # [M] in [0,1]

    def ship_beta(self, h, b_idx, src_idx, tgt_idx):
        """Beta(alpha, beta) over the [0,1] ship fraction, for the RL policy. Same
        (src, tgt) conditioning as ship_fraction; alpha,beta>=1 so the density is
        proper. Returns (alpha, beta), each [M]."""
        src_emb = h[b_idx, src_idx]
        tgt_emb = h[b_idx, tgt_idx]
        pair = torch.cat([src_emb, tgt_emb], dim=-1)             # [M, 2D]
        raw = self.ship_beta_head(pair)                          # [M, 2]
        alpha = 1.0 + F.softplus(raw[..., 0])
        beta = 1.0 + F.softplus(raw[..., 1])
        return alpha, beta

    @torch.no_grad()
    def act(self, feats, valid_mask, mine_mask, greedy=True, gate_threshold=0.5):
        """Single-sample inference helper.

        Inputs are unbatched numpy/tensor arrays of shape [N, ...].
        Returns python lists: for each owned planet that the gate decides to fire,
        a (source_idx, target_idx, ship_fraction) tuple. ship_fraction is in [0,1]
        of the source garrison -- the caller multiplies by available ships.
        """
        feats = torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0)
        valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool).unsqueeze(0)
        mine_mask = torch.as_tensor(mine_mask, dtype=torch.bool).unsqueeze(0)

        target_logits, gate_logits, h = self.forward(feats, valid_mask, mine_mask)
        gate_p = torch.sigmoid(gate_logits[0])

        decisions = []
        mine_idx = torch.nonzero(mine_mask[0], as_tuple=False).flatten().tolist()
        for i in mine_idx:
            if float(gate_p[i].item()) < gate_threshold:
                continue                              # gate says don't launch
            logits = target_logits[0, i]
            if greedy:
                tgt = int(torch.argmax(logits).item())
            else:
                tgt = int(torch.distributions.Categorical(logits=logits).sample().item())
            frac = self.ship_fraction(h, torch.tensor([0]), torch.tensor([i]),
                                      torch.tensor([tgt]))
            decisions.append((i, tgt, float(frac.item())))
        return decisions

    # ----------------------------------------------------------------------- #
    # RL: stochastic rollout + action re-evaluation.
    # Samples the gate (Bernoulli), target (Categorical), and ship fraction
    # (Beta) -- all three are part of the policy gradient. log-probs are recomputed
    # in evaluate_actions during the update (standard on-policy actor-critic).
    # ----------------------------------------------------------------------- #
    @torch.no_grad()
    def act_rl(self, feats, valid_mask, mine_mask):
        """Sample one turn's action. Returns (decisions, record):
          decisions: [(src, tgt, frac)] for planets that launched (-> a move)
          record:    [(src, launched(0/1), tgt|-1, frac)] for EVERY owned planet
                     (gate + the sampled ship fraction are part of the action)."""
        dev = next(self.parameters()).device
        ft = torch.as_tensor(feats, dtype=torch.float32, device=dev).unsqueeze(0)
        vm = torch.as_tensor(valid_mask, dtype=torch.bool, device=dev).unsqueeze(0)
        mm = torch.as_tensor(mine_mask, dtype=torch.bool, device=dev).unsqueeze(0)
        target_logits, gate_logits, h = self.forward(ft, vm, mm)
        gate_p = torch.sigmoid(gate_logits[0])

        decisions, record = [], []
        for i in torch.nonzero(mm[0], as_tuple=False).flatten().tolist():
            launched = int(torch.bernoulli(gate_p[i]).item())
            if launched:
                tgt = int(torch.distributions.Categorical(logits=target_logits[0, i]).sample().item())
                idx = torch.tensor([0], device=dev)
                a, b = self.ship_beta(h, idx, torch.tensor([i], device=dev),
                                      torch.tensor([tgt], device=dev))
                frac = float(torch.distributions.Beta(a, b).sample().item())
                decisions.append((i, tgt, frac))
                record.append((i, 1, tgt, frac))
            else:
                record.append((i, 0, -1, 0.0))
        return decisions, record

    def evaluate_actions(self, feats, valid_mask, mine_mask, records):
        """Re-run a batch of stored steps WITH grad and return (log_prob, value,
        entropy), each [B]. `records` is a list (len B) of per-step action records
        from act_rl (one entry per owned planet, in mine_mask order). Fully
        vectorised -- the per-(step, planet) python loop was the RL bottleneck."""
        target_logits, gate_logits, h = self.forward(feats, valid_mask, mine_mask)
        value = self.value(h, valid_mask)                       # [B]
        B = feats.shape[0]
        dev = feats.device

        # Flatten records (already aligned with nonzero(mine_mask), batch-major).
        launched, target, frac = [], [], []
        for rec in records:
            for (_i, la, tg, fr) in rec:
                launched.append(la)
                target.append(tg)
                frac.append(fr)
        launched = torch.tensor(launched, dtype=torch.float32, device=dev)  # [M]
        target = torch.tensor(target, dtype=torch.long, device=dev)         # [M]
        frac = torch.tensor(frac, dtype=torch.float32, device=dev)          # [M]
        bidx, pidx = torch.nonzero(mine_mask, as_tuple=True)                 # [M] owned

        logp = torch.zeros(B, device=dev)
        ent = torch.zeros(B, device=dev)

        # Gate (Bernoulli) over every owned planet.
        glog = gate_logits[bidx, pidx]                                      # [M]
        logp_gate = -F.binary_cross_entropy_with_logits(glog, launched, reduction="none")
        p = torch.sigmoid(glog)
        ent_gate = -(p * torch.log(p + 1e-9) + (1 - p) * torch.log(1 - p + 1e-9))
        logp.index_add_(0, bidx, logp_gate)
        ent.index_add_(0, bidx, ent_gate)

        # Target (Categorical) only where the planet launched.
        m = launched > 0.5
        if m.any():
            tb, tp = bidx[m], pidx[m]
            tl = target_logits[tb, tp]                                     # [L, N]
            logsm = torch.log_softmax(tl, dim=-1)
            logp_tgt = logsm.gather(1, target[m].unsqueeze(1)).squeeze(1)  # [L]
            ent_tgt = -(logsm.exp() * logsm).sum(-1)                       # [L]
            logp.index_add_(0, tb, logp_tgt)
            ent.index_add_(0, tb, ent_tgt)

            # Ship fraction (Beta) only where the planet launched.
            a, b = self.ship_beta(h, tb, tp, target[m])                    # [L], [L]
            bd = torch.distributions.Beta(a, b)
            fr = frac[m].clamp(1e-4, 1 - 1e-4)                             # keep in (0,1)
            logp.index_add_(0, tb, bd.log_prob(fr))
            ent.index_add_(0, tb, bd.entropy())
        return logp, value, ent
