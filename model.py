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
    def __init__(self, d_model=256, n_heads=8, n_layers=4, dropout=0.0):
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
        # On Apple Silicon (MPS) the nested-tensor fast path (taken when a
        # src_key_padding_mask is passed) hits aten::_nested_tensor_from_mask*,
        # which isn't implemented on MPS and raises. Disable it only there;
        # CUDA/CPU keep the default optimized path.
        enable_nested = not torch.backends.mps.is_available()
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=enable_nested,
        )

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

        # Ship head: a 2-component Gaussian MIXTURE over the FRACTION of the source
        # garrison to send (in [0,1]), conditioned on the (source, target) pair.
        # A single Beta (alpha,beta>=1) is unimodal, but the human launch-fraction
        # is bimodal -- a big "send everything" mass near 1.0 plus a separate
        # smaller-commit mode -- so we use two Gaussians to capture both bumps.
        # ONE head serves both regimes:
        #   - SL fits the mixture MEAN by MSE on the human fraction (train_sl);
        #   - RL samples it and trains by policy gradient (act_rl / evaluate_actions);
        #   - greedy deployment uses the mixture mean (see ship_fraction).
        # The 5 outputs are [mu1, mu2, s1, s2, mix]: component means via sigmoid in
        # (0,1), spreads via softplus (+floor), mixing weight via sigmoid. The bias
        # seeds one mode high (~0.9, "send most") and one mid (~0.5), 50/50.
        self.ship_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 5),
        )
        with torch.no_grad():
            self.ship_head[-1].bias.copy_(torch.tensor([2.2, 0.0, -2.0, -2.0, 0.0]))

        # Value head (RL critic): a board summary -> expected (discounted) return.
        # Unused by SL; trained by regression to the GAE return. Output is LINEAR
        # (unbounded) -- the return is dense-shaped (lead-margin deltas + terminal),
        # so it routinely exceeds [-1,1]; a tanh would saturate and bias it.
        # The summary concatenates mean- and max-pooled planet embeddings (2D):
        # mean captures the average board state, max highlights the single most
        # salient planet (e.g. a dominant garrison) that a mean would wash out.
        self.value_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def value(self, h, valid_mask):
        """Critic estimate per board: pool the (valid) planet embeddings (mean +
        max, concatenated) and map to a scalar (unbounded -- it regresses to the
        dense-shaped return, which is not confined to [-1,1]).
        h [B,N,D], valid_mask [B,N] -> [B]."""
        mask = valid_mask.unsqueeze(-1).float()                          # [B,N,1]
        mean_pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [B,D]
        # Mask padded planets to -inf so they never win the max.
        max_pooled = h.masked_fill(~valid_mask.unsqueeze(-1), NEG_INF).max(dim=1).values  # [B,D]
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)            # [B,2D]
        return self.value_head(pooled).squeeze(-1)                       # [B], linear

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
        """Greedy/deterministic fraction of the source garrison to send, in [0,1],
        for given source/target pairs: the MODE of the mixture -- the mean of the
        higher-weight component. NOT the mixture mean w*mu1+(1-w)*mu2: the launch
        fraction is bimodal, so the mixture mean falls in the VALLEY between the two
        bumps (a fraction the model itself deems unlikely). Picking the dominant
        component's mean keeps the output on a real bump while staying deterministic.
        Used by SL-greedy deployment and inference (RL samples instead, see act_rl).

        h        [B, N, D]
        b_idx, src_idx, tgt_idx : 1-D index tensors selecting (batch, src, tgt).
        Multiply by the source's available ships for an absolute count (see act).
        """
        means, _sigmas, w = self.ship_mixture_params(h, b_idx, src_idx, tgt_idx)
        k = w.argmax(-1, keepdim=True)                         # [M,1] dominant mode
        return means.gather(-1, k).squeeze(-1).clamp(0.0, 1.0)  # [M] in [0,1]

    SHIP_SIGMA_FLOOR = 0.03   # keeps the "send everything" component from collapsing to a spike
    # Ceiling on the ship-fraction std. The fraction lives in [0,1], so sigma>~0.5
    # is already near-flat over the support -- past that point the only thing growing
    # sigma buys is differential entropy (0.5*log(2*pi*e*sigma^2), UNBOUNDED above).
    # With an entropy bonus that hands the optimizer a free runaway: inflate sigma to
    # farm entropy, the ship-fraction policy degenerates to noise, win rate collapses.
    # clamp(max) caps it AND zeroes the gradient above the cap, removing the incentive.
    SHIP_SIGMA_CAP = 0.5

    def ship_mixture_params(self, h, b_idx, src_idx, tgt_idx):
        """Raw parameters of the 2-Gaussian ship-fraction mixture for each
        (src, tgt) pair. Returns (means, sigmas, weights):
          means   [M, 2] component means in (0,1)   (sigmoid)
          sigmas  [M, 2] component std devs > 0      (softplus + floor)
          weights [M, 2] mixing probabilities, rows sum to 1 (sigmoid on logit)."""
        src_emb = h[b_idx, src_idx]
        tgt_emb = h[b_idx, tgt_idx]
        pair = torch.cat([src_emb, tgt_emb], dim=-1)             # [M, 2D]
        raw = self.ship_head(pair)                               # [M, 5]
        means = torch.sigmoid(raw[..., 0:2])                     # [M, 2] in (0,1)
        sigmas = (F.softplus(raw[..., 2:4]) + self.SHIP_SIGMA_FLOOR).clamp(max=self.SHIP_SIGMA_CAP)
        w1 = torch.sigmoid(raw[..., 4:5])                        # [M, 1]
        weights = torch.cat([w1, 1.0 - w1], dim=-1)              # [M, 2]
        return means, sigmas, weights

    def ship_mixture(self, h, b_idx, src_idx, tgt_idx):
        """The ship-fraction distribution as a torch MixtureSameFamily of 2 Normals.
        Support is all of R; callers clamp samples to [0,1]. log_prob is exact;
        for entropy use ship_entropy (no closed form for a Gaussian mixture)."""
        means, sigmas, weights = self.ship_mixture_params(h, b_idx, src_idx, tgt_idx)
        mix = torch.distributions.Categorical(probs=weights)
        comp = torch.distributions.Normal(means, sigmas)
        return torch.distributions.MixtureSameFamily(mix, comp), means, sigmas, weights

    def ship_logprob(self, h, b_idx, src_idx, tgt_idx, x):
        """log p(x) under the 2-Gaussian ship-fraction mixture, computed in closed
        form (no torch.distributions objects -> no per-batch construction/validation
        overhead, which matters on the SL hot path). x [M] -> [M]."""
        means, sigmas, weights = self.ship_mixture_params(h, b_idx, src_idx, tgt_idx)
        z = (x.unsqueeze(-1) - means) / sigmas                              # [M,2]
        log_comp = -0.5 * z * z - torch.log(sigmas) - 0.5 * math.log(2 * math.pi)
        # clamp_min before log: a saturated sigmoid makes one weight exactly 0.0,
        # and log(0)=-inf gives an inf gradient (0*inf -> NaN) that poisons the
        # shared encoder. clamp passes gradient only where unclamped, so this caps
        # the weight log-prob without the NaN.
        log_weights = torch.log(weights.clamp_min(1e-8))                    # [M,2]
        return torch.logsumexp(log_weights + log_comp, dim=-1)             # [M]

    @staticmethod
    def ship_entropy(sigmas, weights):
        """Differentiable surrogate for the mixture's differential entropy: the
        weight-averaged per-component Gaussian entropy, sum_k w_k * 0.5*log(2*pi*e*sigma_k^2).
        This is a lower bound (ignores the mixing term) but is analytic and pushes
        component spread up, which is all the entropy bonus needs. [M]."""
        comp_ent = 0.5 * math.log(2.0 * math.pi * math.e) + torch.log(sigmas)  # [M,2]
        return (weights * comp_ent).sum(-1)                                    # [M]

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
        dev = h.device

        mine_idx = torch.nonzero(mine_mask[0], as_tuple=False).flatten()   # [M] owned
        if mine_idx.numel() == 0:
            return []

        # Gate by threshold over all owned planets at once.
        gate_p = torch.sigmoid(gate_logits[0, mine_idx])                  # [M]
        fire = gate_p >= gate_threshold                                   # [M]
        L = int(fire.sum().item())
        if L == 0:
            return []

        # Target + ship fraction for the firing subset, all batched.
        src_l = mine_idx[fire]                                            # [L]
        logits_l = target_logits[0, src_l]                               # [L, N]
        if greedy:
            tgt_l = logits_l.argmax(dim=-1)                              # [L]
        else:
            tgt_l = torch.distributions.Categorical(logits=logits_l).sample()
        b0 = torch.zeros(L, dtype=torch.long, device=dev)
        frac_l = self.ship_fraction(h, b0, src_l, tgt_l)                 # [L] Beta mean

        src_list = src_l.tolist()
        tgt_list = tgt_l.tolist()
        frac_list = frac_l.tolist()
        return [(src_list[k], tgt_list[k], frac_list[k]) for k in range(L)]

    # ----------------------------------------------------------------------- #
    # RL: stochastic rollout + action re-evaluation.
    # Samples the gate (Bernoulli), target (Categorical), and ship fraction
    # (Beta) -- all three are part of the policy gradient. log-probs are recomputed
    # in evaluate_actions during the update (standard on-policy actor-critic).
    # ----------------------------------------------------------------------- #
    @torch.no_grad()
    def act_rl(self, feats, valid_mask, mine_mask, gate_bias=0.0):
        """Sample one turn's action. Returns (decisions, record):
          decisions: [(src, tgt, frac)] for planets that launched (-> a move)
          record:    [(src, launched(0/1), tgt|-1, frac)] for EVERY owned planet
                     (gate + the sampled ship fraction are part of the action).
        gate_bias is added to the gate logit before sampling -- pass -log(pos_weight)
        to undo the SL class-imbalance inflation so the Bernoulli fires at the true
        (calibrated) launch rate instead of the threshold-tuned inflated one."""
        dev = next(self.parameters()).device
        ft = torch.as_tensor(feats, dtype=torch.float32, device=dev).unsqueeze(0)
        vm = torch.as_tensor(valid_mask, dtype=torch.bool, device=dev).unsqueeze(0)
        mm = torch.as_tensor(mine_mask, dtype=torch.bool, device=dev).unsqueeze(0)
        target_logits, gate_logits, h = self.forward(ft, vm, mm)

        mine_idx = torch.nonzero(mm[0], as_tuple=False).flatten()        # [M] owned
        M = mine_idx.numel()
        if M == 0:
            return [], []

        # Gate over all owned planets at once (one Bernoulli draw), de-biased.
        gate_p = torch.sigmoid(gate_logits[0, mine_idx] + gate_bias)     # [M] calibrated
        launched = torch.bernoulli(gate_p).bool()                       # [M]

        tgt_all = torch.full((M,), -1, dtype=torch.long, device=dev)    # -1 = no launch
        frac_all = torch.zeros(M, device=dev)
        # Target + ship fraction only for the launched subset, all batched.
        L = int(launched.sum().item())
        if L > 0:
            src_l = mine_idx[launched]                                  # [L]
            tgt_l = torch.distributions.Categorical(
                logits=target_logits[0, src_l]).sample()               # [L]
            b0 = torch.zeros(L, dtype=torch.long, device=dev)
            dist, _m, _s, _w = self.ship_mixture(h, b0, src_l, tgt_l)  # 2-Gaussian mixture
            frac_l = dist.sample().clamp(0.0, 1.0)                     # [L] in [0,1]
            tgt_all[launched] = tgt_l
            frac_all[launched] = frac_l

        # One device->host sync, then a pure-python assembly loop (no tensor ops).
        mine_list = mine_idx.tolist()
        launched_list = launched.tolist()
        tgt_list = tgt_all.tolist()
        frac_list = frac_all.tolist()

        decisions, record = [], []
        for k in range(M):
            i = mine_list[k]
            if launched_list[k]:
                decisions.append((i, tgt_list[k], frac_list[k]))
                record.append((i, 1, tgt_list[k], frac_list[k]))
            else:
                record.append((i, 0, -1, 0.0))
        return decisions, record

    def evaluate_actions(self, feats, valid_mask, mine_mask, records, teacher=None,
                         gate_bias=0.0):
        """Re-run a batch of stored steps WITH grad and return (log_prob, value,
        entropy), each [B]. `records` is a list (len B) of per-step action records
        from act_rl (one entry per owned planet, in mine_mask order). Fully
        vectorised -- the per-(step, planet) python loop was the RL bottleneck.

        gate_bias MUST match the value passed to act_rl so the re-evaluated gate
        log-prob is for the same (calibrated) Bernoulli that actually sampled.

        If `teacher` (a frozen OrbitNet, e.g. the SL policy) is given, also return
        a 4th value `kl` [B]: the per-step forward KL(teacher || student) summed
        over gate (all owned) + target & ship (launched). Differentiates through
        self only; the teacher forward runs under no_grad. Used as a behavior-
        cloning anchor term in the PPO loss (cf. AlphaStar human_policy_kl_loss)."""
        target_logits, gate_logits, h = self.forward(feats, valid_mask, mine_mask)
        value = self.value(h, valid_mask)                       # [B]
        B = feats.shape[0]
        dev = feats.device

        if teacher is not None:
            with torch.no_grad():
                t_target_logits, t_gate_logits, t_h = teacher.forward(
                    feats, valid_mask, mine_mask)

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
        kl = torch.zeros(B, device=dev) if teacher is not None else None

        # Gate (Bernoulli) over every owned planet, de-biased to match act_rl.
        glog = gate_logits[bidx, pidx] + gate_bias                          # [M]
        logp_gate = -F.binary_cross_entropy_with_logits(glog, launched, reduction="none")
        p = torch.sigmoid(glog)
        ent_gate = -(p * torch.log(p + 1e-9) + (1 - p) * torch.log(1 - p + 1e-9))
        logp.index_add_(0, bidx, logp_gate)
        ent.index_add_(0, bidx, ent_gate)
        if teacher is not None:
            # Bernoulli KL(teacher || student) over all owned planets (both de-biased).
            t_glog = t_gate_logits[bidx, pidx] + gate_bias
            t_p = torch.sigmoid(t_glog)
            kl_gate = (t_p * (F.logsigmoid(t_glog) - F.logsigmoid(glog))
                       + (1 - t_p) * (F.logsigmoid(-t_glog) - F.logsigmoid(-glog)))
            kl.index_add_(0, bidx, kl_gate)

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

            # Ship fraction (2-Gaussian mixture) only where the planet launched.
            # Closed-form log_prob/entropy -- no distribution object on this path.
            _sm, ssig, sw = self.ship_mixture_params(h, tb, tp, target[m])
            fr = frac[m].clamp(0.0, 1.0)
            logp.index_add_(0, tb, self.ship_logprob(h, tb, tp, target[m], fr))
            ent.index_add_(0, tb, self.ship_entropy(ssig, sw))            # analytic surrogate

            if teacher is not None:
                # Categorical KL(teacher || student) over launched targets.
                t_logsm = torch.log_softmax(t_target_logits[tb, tp], dim=-1)
                kl_tgt = (t_logsm.exp() * (t_logsm - logsm)).sum(-1)       # [L]
                kl.index_add_(0, tb, kl_tgt)
                # KL(teacher || student) for the ship fraction. A Gaussian mixture
                # has no closed-form KL, so estimate it by Monte Carlo on a single
                # teacher draw: E_t[log p_t - log p_s] (differentiable through the
                # student only; the teacher runs under no_grad above).
                td, _tm, _ts, _tw = teacher.ship_mixture(t_h, tb, tp, target[m])
                x = td.sample().clamp(0.0, 1.0)                            # [L] teacher draw
                kl_ship = (td.log_prob(x)
                           - self.ship_logprob(h, tb, tp, target[m], x))  # [L]
                kl.index_add_(0, tb, kl_ship)

        if teacher is not None:
            return logp, value, ent, kl
        return logp, value, ent
