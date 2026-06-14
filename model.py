import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import FEATURE_DIM

NEG_INF = -1e9

class OrbitNet(nn.Module):
    # 5% 단위로 0.0 ~ 1.0 공간을 20개 구간으로 분할
    SHIP_BINS = 100

    def __init__(self, d_model=256, n_heads=8, n_layers=4, dropout=0.0,
                 ar_layers=1, ship_emb_dim=32):
        super().__init__()
        self.d_model = d_model
        self.ship_emb_dim = ship_emb_dim

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

        # --- Autoregressive action decoder (GRU) ---
        # 내 행성을 고정 순서(slot index 오름차순)로 처리하며, 각 행성의 결정을
        # 앞 행성들이 '이미 정한 행동'에 조건화한다. GRU의 hidden state가 앞 행동
        # 요약을 들고 간다:
        #   input_k = h[p_k] + action_emb(a_{k-1})   (a_{-1}=start_tok)
        #   z_k = GRU(input_k, hidden_{k-1})
        # 시퀀스가 짧아(K<=MAX_PLANETS) RNN이 적합: 추론이 태생적으로 O(K) incremental
        # (KV-cache 불필요), 학습은 fused nn.GRU로 한 번에 처리. GRU가 자체적으로
        # causal이라 positional encoding/causal mask가 필요 없다.
        self.start_tok = nn.Parameter(torch.zeros(d_model))       # a_{-1} (이전 행동 없음)
        self.ship_act_emb = nn.Embedding(self.SHIP_BINS, ship_emb_dim)
        self.noop_tgt = nn.Parameter(torch.zeros(d_model))        # 미발사 시 target 표현
        self.noop_ship = nn.Parameter(torch.zeros(ship_emb_dim))  # 미발사 시 ship 표현
        # 행동 토큰 인코더: [h[src], target표현, ship표현, launched] -> d_model
        self.action_enc = nn.Sequential(
            nn.Linear(2 * d_model + ship_emb_dim + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.ar_gru = nn.GRU(
            input_size=d_model, hidden_size=d_model, num_layers=ar_layers,
            batch_first=True, dropout=(dropout if ar_layers > 1 else 0.0),
        )

        # Gate+Ship head: AR 디코더 출력 z_k(앞 행동을 본 source 표현)와 선택된 target
        # 임베딩 h[tgt]를 입력받아 파견 여부(gate)와 파견 비율 Bin(ship)을 함께 예측.
        # 출력 = [gate_logit(1), ship_bin_logits(SHIP_BINS)]
        self.gate_ship_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1 + self.SHIP_BINS),
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

    def forward(self, feats, valid_mask, mine_mask=None):
        """인코더만 수행하여 보드 임베딩 h [B,N,D]를 반환한다.
        target/gate/ship logit은 AR 디코더를 거쳐 act/act_rl/evaluate_actions에서 계산.
        (mine_mask는 호환을 위해 받지만 인코더에선 사용하지 않는다.)"""
        h = self.embed(feats)
        h = self.encoder(h, src_key_padding_mask=~valid_mask)
        return h

    def bin_to_fraction(self, bin_idx):
        """Bin 인덱스를 0.0 ~ 1.0 사이의 비율로 변환 (0 -> 0%, BINS-1 -> 100%)."""
        return bin_idx.float() / (self.SHIP_BINS - 1)

    # ===== Autoregressive helpers =====
    def _action_embed(self, h, src_h, launched, target, bin_idx):
        """행동 (launched, target, bin)을 d_model 벡터로 인코딩 (디코더 메모리용).
        h: [B,N,D], src_h: [B,K,D], launched: [B,K], target/bin_idx: [B,K] long.
        미발사 행성은 noop 표현을 사용한다."""
        lb = launched.bool().unsqueeze(-1)                                   # [B,K,1]
        tgt_h = h.gather(1, target.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d_model))
        target_repr = torch.where(lb, tgt_h, self.noop_tgt)                  # [B,K,D]
        ship_e = self.ship_act_emb(bin_idx.clamp(0, self.SHIP_BINS - 1))     # [B,K,ship_emb]
        ship_repr = torch.where(lb, ship_e, self.noop_ship)                  # [B,K,ship_emb]
        a_in = torch.cat([src_h, target_repr, ship_repr, lb.float()], dim=-1)
        return self.action_enc(a_in)

    def _ar_decode(self, h, planet_idx, launched, target, bin_idx, seq_mask=None):
        """패딩된 결정 시퀀스를 GRU로 한 번에 처리해 z [B,K,D] 반환 (학습용 병렬 경로).
        스텝 k 입력 = h[p_k] + action_emb(a_{k-1}) (a_{-1}=start_tok).
        GRU가 자체적으로 causal이라 mask가 필요 없다. 패딩은 각 보드 시퀀스의 suffix라
        prefix(=valid) 출력에 영향을 주지 않고, 사용처에서 valid 위치만 gather한다.
        seq_mask는 호출 호환을 위해 받지만 사용하지 않는다."""
        B = planet_idx.shape[0]
        src_h = h.gather(1, planet_idx.unsqueeze(-1).expand(-1, -1, self.d_model))  # [B,K,D]
        a_emb = self._action_embed(h, src_h, launched, target, bin_idx)             # [B,K,D]
        # 스텝 k의 입력 행동 = a_{k-1} (오른쪽 shift), a_{-1}=start_tok.
        start = self.start_tok.expand(B, 1, self.d_model)
        prev = torch.cat([start, a_emb[:, :-1]], dim=1)                             # [B,K,D]
        inp = src_h + prev                                                          # [B,K,D]
        z, _ = self.ar_gru(inp)                                                     # [B,K,D]
        return z, src_h

    def _target_logits_from(self, z, kh, valid_mask, bidx, pidx):
        """z(쿼리)와 kh=k_proj(h)(키)로 target logit [M,N] 계산 + 유효/자기 마스킹."""
        scale = 1.0 / math.sqrt(self.d_model)
        q = self.q_proj(z)                                       # [M,D]
        logits = (q.unsqueeze(1) * kh[bidx]).sum(-1) * scale     # [M,N]
        logits = logits.masked_fill(~valid_mask[bidx], NEG_INF)
        logits[torch.arange(z.shape[0], device=z.device), pidx] = NEG_INF
        return logits

    def _gate_ship_from(self, src_repr, tgt_repr):
        """source 표현(z_k)과 target 임베딩 h[tgt]로 (gate_logit, ship_logits) 계산."""
        out = self.gate_ship_head(torch.cat([src_repr, tgt_repr], dim=-1))
        return out[..., 0], out[..., 1:]

    @torch.no_grad()
    def _rollout(self, h, kh, valid_mask, mine_idx, pick_sample, gate_sample,
                 gate_bias=0.0, gate_threshold=0.5):
        """단일 보드(batch=0)에 대한 AR 순차 디코딩. (decisions, record) 반환.
        - pick_sample: target/ship을 샘플링할지(True) argmax(False).
        - gate_sample: gate를 bernoulli(+bias)로(True) 또는 threshold로(False) 결정."""
        dev = h.device
        scale = 1.0 / math.sqrt(self.d_model)
        K = mine_idx.numel()
        hidden = None                       # GRU hidden [num_layers,1,D] (None=zeros)
        prev_a = self.start_tok
        decisions, record = [], []
        for k in range(K):
            p = int(mine_idx[k])
            inp = (h[0, p] + prev_a).view(1, 1, self.d_model)                 # 스텝 입력
            out, hidden = self.ar_gru(inp, hidden)                            # O(1) per step
            z = out[0, 0]                                                     # [D]

            logits = (self.q_proj(z).unsqueeze(0) * kh[0]).sum(-1) * scale    # [N]
            logits = logits.masked_fill(~valid_mask[0], NEG_INF)
            logits[p] = NEG_INF
            if pick_sample:
                tgt = int(torch.distributions.Categorical(logits=logits).sample())
            else:
                tgt = int(logits.argmax())

            gate_logit, s_logits = self._gate_ship_from(z, h[0, tgt])
            if gate_sample:
                launched = bool(torch.bernoulli(torch.sigmoid(gate_logit + gate_bias)))
            else:
                launched = bool(torch.sigmoid(gate_logit) >= gate_threshold)

            bin_i = 0
            if launched:
                if pick_sample:
                    bin_i = int(torch.distributions.Categorical(logits=s_logits).sample())
                else:
                    bin_i = int(s_logits.argmax())

            # 다음 토큰 입력용 action_emb (실제로 취한 행동 기준; 미발사면 noop)
            prev_a = self._action_embed(
                h[:1], h[:1, p:p + 1],
                torch.tensor([[float(launched)]], device=dev),
                torch.tensor([[tgt]], dtype=torch.long, device=dev),
                torch.tensor([[bin_i]], dtype=torch.long, device=dev),
            )[0, 0]

            if launched:
                # act_rl(sample)은 bin 중앙값, act(greedy)은 bin_to_fraction 사용 (기존 관례 유지).
                frac = (bin_i + 0.5) / self.SHIP_BINS if pick_sample \
                    else float(self.bin_to_fraction(torch.tensor(bin_i)))
                decisions.append((p, tgt, frac))
                record.append((p, 1, tgt, bin_i))
            else:
                # 미발사라도 (재현용으로) 정해진 target은 기록해 둔다.
                record.append((p, 0, tgt, 0))
        return decisions, record

    @torch.no_grad()
    def act(self, feats, valid_mask, mine_mask, greedy=True, gate_threshold=0.5):
        feats = torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0)
        valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool).unsqueeze(0)
        mine_mask = torch.as_tensor(mine_mask, dtype=torch.bool).unsqueeze(0)

        h = self.forward(feats, valid_mask)
        kh = self.k_proj(h)
        mine_idx = torch.nonzero(mine_mask[0], as_tuple=False).flatten()
        if mine_idx.numel() == 0:
            return []
        # gate는 항상 threshold 기준, target/ship은 greedy면 argmax / 아니면 샘플.
        decisions, _ = self._rollout(
            h, kh, valid_mask, mine_idx,
            pick_sample=not greedy, gate_sample=False, gate_threshold=gate_threshold,
        )
        return decisions

    @torch.no_grad()
    def act_rl(self, feats, valid_mask, mine_mask, gate_bias=0.0):
        dev = next(self.parameters()).device
        ft = torch.as_tensor(feats, dtype=torch.float32, device=dev).unsqueeze(0)
        vm = torch.as_tensor(valid_mask, dtype=torch.bool, device=dev).unsqueeze(0)
        mm = torch.as_tensor(mine_mask, dtype=torch.bool, device=dev).unsqueeze(0)

        h = self.forward(ft, vm)
        kh = self.k_proj(h)
        mine_idx = torch.nonzero(mm[0], as_tuple=False).flatten()
        if mine_idx.numel() == 0:
            return [], []
        return self._rollout(
            h, kh, vm, mine_idx,
            pick_sample=True, gate_sample=True, gate_bias=gate_bias,
        )

    @torch.no_grad()
    def act_debug(self, feats, valid_mask, mine_mask, gate_threshold=0.5):
        """진단용 greedy AR 디코딩. dict { planet_idx: (gate_p, tgt, tgt_prob, frac, launched) }."""
        ft = torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0)
        vm = torch.as_tensor(valid_mask, dtype=torch.bool).unsqueeze(0)
        mm = torch.as_tensor(mine_mask, dtype=torch.bool).unsqueeze(0)
        h = self.forward(ft, vm)
        kh = self.k_proj(h)
        dev = h.device
        scale = 1.0 / math.sqrt(self.d_model)
        mine_idx = torch.nonzero(mm[0], as_tuple=False).flatten()
        hidden = None
        prev_a = self.start_tok
        out = {}
        for k in range(mine_idx.numel()):
            p = int(mine_idx[k])
            inp = (h[0, p] + prev_a).view(1, 1, self.d_model)
            gout, hidden = self.ar_gru(inp, hidden)
            z = gout[0, 0]
            logits = (self.q_proj(z).unsqueeze(0) * kh[0]).sum(-1) * scale
            logits = logits.masked_fill(~vm[0], NEG_INF)
            logits[p] = NEG_INF
            probs = torch.softmax(logits, dim=-1)
            tgt = int(logits.argmax())
            gate_logit, s_logits = self._gate_ship_from(z, h[0, tgt])
            gate_p = float(torch.sigmoid(gate_logit))
            launched = gate_p >= gate_threshold
            bin_i = int(s_logits.argmax())
            frac = float(self.bin_to_fraction(torch.tensor(bin_i)))
            out[p] = (gate_p, tgt, float(probs[tgt]), frac, launched)
            prev_a = self._action_embed(
                h[:1], h[:1, p:p + 1],
                torch.tensor([[float(launched)]], device=dev),
                torch.tensor([[tgt]], dtype=torch.long, device=dev),
                torch.tensor([[bin_i]], dtype=torch.long, device=dev),
            )[0, 0]
        return out

    def evaluate_actions(self, feats, valid_mask, mine_mask, records, teacher=None, gate_bias=0.0):
        h = self.forward(feats, valid_mask)
        kh = self.k_proj(h)
        value = self.value(h, valid_mask)
        B = feats.shape[0]
        dev = feats.device

        bidx, pidx = torch.nonzero(mine_mask, as_tuple=True)
        logp = torch.zeros(B, device=dev)
        ent = torch.zeros(B, device=dev)
        kl = torch.zeros(B, device=dev) if teacher is not None else None
        if bidx.numel() == 0:
            if teacher is not None:
                return logp, value, ent, kl
            return logp, value, ent

        # records 순서 = 보드별 행성 index 오름차순 = nonzero(mine) 순서 = _rollout 처리 순서.
        launched, target, frac_bin = [], [], []
        for rec in records:
            for (_i, la, tg, fr) in rec:
                launched.append(la)
                target.append(tg)
                frac_bin.append(fr)
        launched = torch.tensor(launched, dtype=torch.float32, device=dev)
        target = torch.tensor(target, dtype=torch.long, device=dev)
        frac_bin = torch.tensor(frac_bin, dtype=torch.long, device=dev)

        # 보드별 시퀀스 위치 pos 계산 후 [B,K_max]로 scatter (패딩 디코딩 입력 구성).
        counts = mine_mask.sum(1).long()                         # [B]
        K_max = int(counts.max())
        start = torch.zeros(B, dtype=torch.long, device=dev)
        start[1:] = counts.cumsum(0)[:-1]
        pos = torch.arange(bidx.numel(), device=dev) - start[bidx]   # [M] 보드 내 위치

        def scatter(vals, dtype):
            t = torch.zeros(B, K_max, dtype=dtype, device=dev)
            t[bidx, pos] = vals.to(dtype)
            return t

        planet_pad = scatter(pidx, torch.long)
        launched_pad = scatter(launched, torch.float32)
        target_pad = scatter(target, torch.long)
        bin_pad = scatter(frac_bin, torch.long)
        seq_mask = torch.zeros(B, K_max, dtype=torch.bool, device=dev)
        seq_mask[bidx, pos] = True

        z, _ = self._ar_decode(h, planet_pad, launched_pad, target_pad, bin_pad, seq_mask)
        z_flat = z[bidx, pos]                                    # [M,D]

        # Gate + Ship (기록된 target을 조건으로). gate는 전체, ship은 launched만 사용.
        gate_logit, s_logits_all = self._gate_ship_from(z_flat, h[bidx, target])
        if teacher is not None:
            with torch.no_grad():
                t_h = teacher.forward(feats, valid_mask)
                t_kh = teacher.k_proj(t_h)
                t_z, _ = teacher._ar_decode(t_h, planet_pad, launched_pad, target_pad, bin_pad, seq_mask)
                t_z_flat = t_z[bidx, pos]
                t_gate_logit, t_s_logits_all = teacher._gate_ship_from(t_z_flat, t_h[bidx, target])

        # Gate
        glog = gate_logit + gate_bias
        logp_gate = -F.binary_cross_entropy_with_logits(glog, launched, reduction="none")
        p = torch.sigmoid(glog)
        ent_gate = -(p * torch.log(p + 1e-9) + (1 - p) * torch.log(1 - p + 1e-9))
        logp.index_add_(0, bidx, logp_gate)
        ent.index_add_(0, bidx, ent_gate)
        if teacher is not None:
            t_glog = t_gate_logit + gate_bias
            t_p = torch.sigmoid(t_glog)
            kl_gate = (t_p * (F.logsigmoid(t_glog) - F.logsigmoid(glog))
                       + (1 - t_p) * (F.logsigmoid(-t_glog) - F.logsigmoid(-glog)))
            kl.index_add_(0, bidx, kl_gate)

        m = launched > 0.5
        if m.any():
            tb, tp = bidx[m], pidx[m]
            # Target (z_k 쿼리로 계산)
            tl = self._target_logits_from(z_flat[m], kh, valid_mask, tb, tp)
            logsm = torch.log_softmax(tl, dim=-1)
            logp_tgt = logsm.gather(1, target[m].unsqueeze(1)).squeeze(1)
            ent_tgt = -(logsm.exp() * logsm).sum(-1)
            logp.index_add_(0, tb, logp_tgt)
            ent.index_add_(0, tb, ent_tgt)

            # Ship Fraction (Categorical) — gate와 같은 head 출력에서 slice
            s_logits = s_logits_all[m]
            logsm_ship = torch.log_softmax(s_logits, dim=-1)
            logp_ship = logsm_ship.gather(1, frac_bin[m].unsqueeze(1)).squeeze(1)
            ent_ship = -(logsm_ship.exp() * logsm_ship).sum(-1)
            logp.index_add_(0, tb, logp_ship)
            ent.index_add_(0, tb, ent_ship)

            if teacher is not None:
                t_tl = teacher._target_logits_from(t_z_flat[m], t_kh, valid_mask, tb, tp)
                t_logsm = torch.log_softmax(t_tl, dim=-1)
                kl_tgt = (t_logsm.exp() * (t_logsm - logsm)).sum(-1)
                kl.index_add_(0, tb, kl_tgt)

                t_s_logits = t_s_logits_all[m]
                t_logsm_ship = torch.log_softmax(t_s_logits, dim=-1)
                kl_ship = (t_logsm_ship.exp() * (t_logsm_ship - logsm_ship)).sum(-1)
                kl.index_add_(0, tb, kl_ship)

        if teacher is not None:
            return logp, value, ent, kl
        return logp, value, ent