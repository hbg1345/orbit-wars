import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import FEATURE_DIM, MAX_PLANETS

NEG_INF = -1e9

class OrbitNet(nn.Module):
    # Ship amount is no longer a learned distribution. Per launch the model only
    # picks an amount MODE -- 0 = minimum-capture (forecast-derived smallest fleet
    # that takes the target), 1 = all-in (send the whole garrison). The concrete
    # ship count + aim angle come from the heuristic (forecast.min_capture_fleet /
    # all available), not the network.
    MODE_MIN, MODE_ALL = 0, 1

    def __init__(self, d_model=256, n_heads=4, n_layers=6, dropout=0.0):
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
        enable_nested = not torch.backends.mps.is_available()
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=enable_nested,
        )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        self.gate_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # Mode head: one logit per (src, tgt) pair = P(all-in) vs minimum-capture.
        self.mode_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def value(self, h, valid_mask):
        mask = valid_mask.unsqueeze(-1).float()
        mean_pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        max_pooled = h.masked_fill(~valid_mask.unsqueeze(-1), NEG_INF).max(dim=1).values
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
        return self.value_head(pooled).squeeze(-1)

    def forward(self, feats, valid_mask, mine_mask):
        B, N, _ = feats.shape

        h = self.embed(feats)
        pad_mask = ~valid_mask
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        q = self.q_proj(h)
        k = self.k_proj(h)
        scale = 1.0 / math.sqrt(self.d_model)

        target_logits = torch.matmul(q, k.transpose(1, 2)) * scale
        eye = torch.eye(N, dtype=torch.bool, device=feats.device).unsqueeze(0)
        invalid_tgt = (~valid_mask).unsqueeze(1) | eye
        target_logits = target_logits.masked_fill(invalid_tgt, NEG_INF)

        gate_logits = self.gate_head(h).squeeze(-1)

        return target_logits, gate_logits, h

    # --- Amount-mode head (min-capture vs all-in) ---
    def mode_logit(self, h, b_idx, src_idx, tgt_idx):
        """Logit of P(all-in) for each (src, tgt) pair -> shape [L]."""
        src_emb = h[b_idx, src_idx]
        tgt_emb = h[b_idx, tgt_idx]
        pair = torch.cat([src_emb, tgt_emb], dim=-1)
        return self.mode_head(pair).squeeze(-1)

    @torch.no_grad()
    def act(self, feats, valid_mask, mine_mask, greedy=True, gate_threshold=0.5, gate_bias=0.0):
        dev = next(self.parameters()).device
        feats = torch.as_tensor(feats, dtype=torch.float32, device=dev).unsqueeze(0)
        valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=dev).unsqueeze(0)
        mine_mask = torch.as_tensor(mine_mask, dtype=torch.bool, device=dev).unsqueeze(0)

        target_logits, gate_logits, h = self.forward(feats, valid_mask, mine_mask)

        mine_idx = torch.nonzero(mine_mask[0], as_tuple=False).flatten()
        if mine_idx.numel() == 0:
            return []

        # gate_bias mirrors act_rl: with the SL pos_weight inflation de-biased away
        # (gate_bias = -log(pos_weight)), greedy fires iff the DE-BIASED prob clears
        # the threshold -- i.e. greedy == the MODE of the training sampling policy, so
        # eval/deploy fire at the same rate training optimises (no train/eval drift).
        gate_p = torch.sigmoid(gate_logits[0, mine_idx] + gate_bias)
        fire = gate_p >= gate_threshold
        L = int(fire.sum().item())
        if L == 0:
            return []

        src_l = mine_idx[fire]
        logits_l = target_logits[0, src_l]
        b0 = torch.zeros(L, dtype=torch.long, device=dev)

        if greedy:
            tgt_l = logits_l.argmax(dim=-1)
            mode_l = (self.mode_logit(h, b0, src_l, tgt_l) >= 0).long()
        else:
            tgt_l = torch.distributions.Categorical(logits=logits_l).sample()
            mp = torch.sigmoid(self.mode_logit(h, b0, src_l, tgt_l))
            mode_l = torch.bernoulli(mp).long()

        src_list = src_l.tolist()
        tgt_list = tgt_l.tolist()
        mode_list = mode_l.tolist()
        # (src, tgt, mode): mode 0 = min-capture, 1 = all-in. The heuristic
        # (forecast.decisions_to_moves) turns this into ship count + angle.
        return [(src_list[k], tgt_list[k], mode_list[k]) for k in range(L)]

    @torch.no_grad()
    def act_rl(self, feats, valid_mask, mine_mask, gate_bias=0.0):
        dev = next(self.parameters()).device
        ft = torch.as_tensor(feats, dtype=torch.float32, device=dev).unsqueeze(0)
        vm = torch.as_tensor(valid_mask, dtype=torch.bool, device=dev).unsqueeze(0)
        mm = torch.as_tensor(mine_mask, dtype=torch.bool, device=dev).unsqueeze(0)
        target_logits, gate_logits, h = self.forward(ft, vm, mm)

        mine_idx = torch.nonzero(mm[0], as_tuple=False).flatten()
        M = mine_idx.numel()
        if M == 0:
            return [], []

        gate_p = torch.sigmoid(gate_logits[0, mine_idx] + gate_bias)
        launched = torch.bernoulli(gate_p).bool()

        tgt_all = torch.full((M,), -1, dtype=torch.long, device=dev)
        mode_all = torch.zeros(M, dtype=torch.long, device=dev)

        L = int(launched.sum().item())
        if L > 0:
            src_l = mine_idx[launched]
            tgt_l = torch.distributions.Categorical(logits=target_logits[0, src_l]).sample()
            b0 = torch.zeros(L, dtype=torch.long, device=dev)
            # Amount mode: Bernoulli over P(all-in)
            mp = torch.sigmoid(self.mode_logit(h, b0, src_l, tgt_l))
            mode_l = torch.bernoulli(mp).long()
            tgt_all[launched] = tgt_l
            mode_all[launched] = mode_l

        mine_list = mine_idx.tolist()
        launched_list = launched.tolist()
        tgt_list = tgt_all.tolist()
        mode_list = mode_all.tolist()

        decisions, record = [], []
        for k in range(M):
            i = mine_list[k]
            if launched_list[k]:
                # (src, tgt, mode): 0 = min-capture, 1 = all-in.
                decisions.append((i, tgt_list[k], mode_list[k]))
                # record keeps the mode bit for the PPO update (evaluate_actions).
                record.append((i, 1, tgt_list[k], mode_list[k]))
            else:
                record.append((i, 0, -1, 0))
        return decisions, record

    def evaluate_actions(self, feats, valid_mask, mine_mask, records, teacher=None, gate_bias=0.0):
        target_logits, gate_logits, h = self.forward(feats, valid_mask, mine_mask)
        value = self.value(h, valid_mask)
        B = feats.shape[0]
        dev = feats.device

        if teacher is not None:
            with torch.no_grad():
                # Only the teacher's target logits are used (target-only KL); the
                # gate/mode teacher outputs are intentionally ignored.
                t_target_logits, _t_gate_logits, _t_h = teacher.forward(feats, valid_mask, mine_mask)

        launched, target, mode = [], [], []
        for rec in records:
            for (_i, la, tg, md) in rec:
                launched.append(la)
                target.append(tg)
                mode.append(md)              # amount mode bit (0=min, 1=all-in)

        launched = torch.tensor(launched, dtype=torch.float32, device=dev)
        target = torch.tensor(target, dtype=torch.long, device=dev)
        mode = torch.tensor(mode, dtype=torch.float32, device=dev)   # [M] BCE target
        bidx, pidx = torch.nonzero(mine_mask, as_tuple=True)

        logp = torch.zeros(B, device=dev)
        ent = torch.zeros(B, device=dev)
        kl = torch.zeros(B, device=dev) if teacher is not None else None

        # Gate (기존과 동일)
        glog = gate_logits[bidx, pidx] + gate_bias
        logp_gate = -F.binary_cross_entropy_with_logits(glog, launched, reduction="none")
        p = torch.sigmoid(glog)
        ent_gate = -(p * torch.log(p + 1e-9) + (1 - p) * torch.log(1 - p + 1e-9))
        logp.index_add_(0, bidx, logp_gate)
        ent.index_add_(0, bidx, ent_gate)
        # NOTE: gate KL intentionally OMITTED. SL no longer supervises the gate
        # (target-only SL), so the teacher's gate is random -- anchoring RL to it
        # would corrupt the gate. Teacher KL is target-only (see below).

        m = launched > 0.5
        if m.any():
            tb, tp = bidx[m], pidx[m]
            # Target
            tl = target_logits[tb, tp]
            logsm = torch.log_softmax(tl, dim=-1)
            logp_tgt = logsm.gather(1, target[m].unsqueeze(1)).squeeze(1)
            ent_tgt = -(logsm.exp() * logsm).sum(-1)
            logp.index_add_(0, tb, logp_tgt)
            ent.index_add_(0, tb, ent_tgt)

            # Amount mode (binary: 0 = min-capture, 1 = all-in)
            mlog = self.mode_logit(h, tb, tp, target[m])
            logp_mode = -F.binary_cross_entropy_with_logits(mlog, mode[m], reduction="none")
            mp = torch.sigmoid(mlog)
            ent_mode = -(mp * torch.log(mp + 1e-9) + (1 - mp) * torch.log(1 - mp + 1e-9))
            logp.index_add_(0, tb, logp_mode)
            ent.index_add_(0, tb, ent_mode)

            if teacher is not None:
                # Target KL ONLY. SL supervises only the target head, so that is the
                # one component whose teacher is meaningful. Gate KL and amount-mode
                # KL are intentionally omitted -- their teacher is random init and
                # would drag RL's gate/mode toward noise.
                t_logsm = torch.log_softmax(t_target_logits[tb, tp], dim=-1)
                kl_tgt = (t_logsm.exp() * (t_logsm - logsm)).sum(-1)
                kl.index_add_(0, tb, kl_tgt)

        if teacher is not None:
            return logp, value, ent, kl
        return logp, value, ent