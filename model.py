import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import FEATURE_DIM, MAX_PLANETS

NEG_INF = -1e9

class OrbitNet(nn.Module):
    # 함선 파견 비율 bin: bin 0~99 = (b+0.5)/100 = 0.5%,1.5%,...,99.5%,
    # bin 100 = 1.0 (100%, 전군 출격). 총 101개 클래스(= ship_head 출력 차원).
    SHIP_BINS = 101

    def __init__(self, d_model=392, n_heads=8, n_layers=6, dropout=0.0):
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

        # Ship head: 이제 20개의 이산적인 Bin(각 5% 구간)에 대한 Logit을 출력합니다.
        self.ship_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.SHIP_BINS),
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

    # --- Discrete Ship Fraction Helpers ---
    def ship_logits(self, h, b_idx, src_idx, tgt_idx):
        """특정 (src, tgt) 쌍에 대한 20개 Bin의 파견 비율 Logit 반환"""
        src_emb = h[b_idx, src_idx]
        tgt_emb = h[b_idx, tgt_idx]
        pair = torch.cat([src_emb, tgt_emb], dim=-1)
        return self.ship_head(pair)

    def bin_to_fraction(self, bin_idx):
        """Bin 인덱스를 0.0~1.0 비율로 변환. bin 0~99 -> (b+0.5)/100
        (0.5%,1.5%,...,99.5%), bin 100 -> 1.0 (100%, 전군 출격)."""
        denom = self.SHIP_BINS - 1                       # = 100
        frac = (bin_idx.float() + 0.5) / denom
        return torch.where(bin_idx >= denom, torch.ones_like(frac), frac).clamp(0.0, 1.0)

    def ship_fraction(self, h, b_idx, src_idx, tgt_idx):
        """Greedy 평가용: 가장 확률이 높은 Bin 선택"""
        logits = self.ship_logits(h, b_idx, src_idx, tgt_idx)
        bin_idx = logits.argmax(dim=-1)
        return self.bin_to_fraction(bin_idx).clamp(0.0, 1.0)

    @torch.no_grad()
    def act(self, feats, valid_mask, mine_mask, greedy=True, gate_threshold=0.5):
        feats = torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0)
        valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool).unsqueeze(0)
        mine_mask = torch.as_tensor(mine_mask, dtype=torch.bool).unsqueeze(0)

        target_logits, gate_logits, h = self.forward(feats, valid_mask, mine_mask)
        dev = h.device

        mine_idx = torch.nonzero(mine_mask[0], as_tuple=False).flatten()
        if mine_idx.numel() == 0:
            return []

        gate_p = torch.sigmoid(gate_logits[0, mine_idx])
        fire = gate_p >= gate_threshold
        L = int(fire.sum().item())
        if L == 0:
            return []

        src_l = mine_idx[fire]
        logits_l = target_logits[0, src_l]
        b0 = torch.zeros(L, dtype=torch.long, device=dev)

        if greedy:
            tgt_l = logits_l.argmax(dim=-1)
            s_logits = self.ship_logits(h, b0, src_l, tgt_l)
            frac_bin = s_logits.argmax(dim=-1)
        else:
            tgt_l = torch.distributions.Categorical(logits=logits_l).sample()
            s_logits = self.ship_logits(h, b0, src_l, tgt_l)
            frac_bin = torch.distributions.Categorical(logits=s_logits).sample()

        frac_l = self.bin_to_fraction(frac_bin)

        src_list = src_l.tolist()
        tgt_list = tgt_l.tolist()
        frac_list = frac_l.tolist()
        return [(src_list[k], tgt_list[k], frac_list[k]) for k in range(L)]

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
        bin_all = torch.zeros(M, device=dev) # Fraction이 아닌 Bin Index를 저장
        
        L = int(launched.sum().item())
        if L > 0:
            src_l = mine_idx[launched]
            tgt_l = torch.distributions.Categorical(logits=target_logits[0, src_l]).sample()
            b0 = torch.zeros(L, dtype=torch.long, device=dev)
            
            # Discrete Sampling
            s_logits = self.ship_logits(h, b0, src_l, tgt_l)
            bin_l = torch.distributions.Categorical(logits=s_logits).sample()
            
            tgt_all[launched] = tgt_l
            bin_all[launched] = bin_l.float()

        mine_list = mine_idx.tolist()
        launched_list = launched.tolist()
        tgt_list = tgt_all.tolist()
        bin_list = bin_all.tolist()

        decisions, record = [], []
        for k in range(M):
            i = mine_list[k]
            if launched_list[k]:
                bin_idx = int(bin_list[k])
                # 환경에 전달할 실제 비율(bin_to_fraction과 동일 스킴):
                # bin 0~99 -> (b+0.5)/100, bin 100 -> 1.0
                denom = self.SHIP_BINS - 1               # = 100
                real_frac = 1.0 if bin_idx >= denom else (bin_idx + 0.5) / denom
                decisions.append((i, tgt_list[k], real_frac))
                # RL 업데이트를 위해 record에는 '이산형 Bin Index'를 저장
                record.append((i, 1, tgt_list[k], bin_idx)) 
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
                t_target_logits, t_gate_logits, t_h = teacher.forward(feats, valid_mask, mine_mask)

        launched, target, frac_bin = [], [], []
        for rec in records:
            for (_i, la, tg, fr_bin) in rec:
                launched.append(la)
                target.append(tg)
                frac_bin.append(fr_bin) # 저장해둔 Bin Index 꺼냄
                
        launched = torch.tensor(launched, dtype=torch.float32, device=dev)
        target = torch.tensor(target, dtype=torch.long, device=dev)
        frac_bin = torch.tensor(frac_bin, dtype=torch.long, device=dev) # [M] Bin Index
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
        if teacher is not None:
            t_glog = t_gate_logits[bidx, pidx] + gate_bias
            t_p = torch.sigmoid(t_glog)
            kl_gate = (t_p * (F.logsigmoid(t_glog) - F.logsigmoid(glog))
                       + (1 - t_p) * (F.logsigmoid(-t_glog) - F.logsigmoid(-glog)))
            kl.index_add_(0, bidx, kl_gate)

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

            # Ship Fraction (Categorical)
            s_logits = self.ship_logits(h, tb, tp, target[m])
            logsm_ship = torch.log_softmax(s_logits, dim=-1)
            logp_ship = logsm_ship.gather(1, frac_bin[m].unsqueeze(1)).squeeze(1)
            ent_ship = -(logsm_ship.exp() * logsm_ship).sum(-1)
            logp.index_add_(0, tb, logp_ship)
            ent.index_add_(0, tb, ent_ship)

            if teacher is not None:
                # Target KL
                t_logsm = torch.log_softmax(t_target_logits[tb, tp], dim=-1)
                kl_tgt = (t_logsm.exp() * (t_logsm - logsm)).sum(-1)
                kl.index_add_(0, tb, kl_tgt)
                
                # Ship Fraction KL (이제 몬테카를로 샘플링 없이 수식으로 완벽히 떨어집니다!)
                t_s_logits = teacher.ship_logits(t_h, tb, tp, target[m])
                t_logsm_ship = torch.log_softmax(t_s_logits, dim=-1)
                kl_ship = (t_logsm_ship.exp() * (t_logsm_ship - logsm_ship)).sum(-1)
                kl.index_add_(0, tb, kl_ship)

        if teacher is not None:
            return logp, value, ent, kl
        return logp, value, ent