# OrbitNet 구조 & 데이터 흐름

현재 [model.py](../model.py) 기준. 입력이 어디를 타고 흘러 최종 행동(발사 여부 / 타겟 / 함대 비율)이 되는지 정리.

설정: `d_model=256`, `n_heads=8`, `n_layers=4`(인코더), `ar_layers=1`(AR GRU 층수), `dim_feedforward=4·d_model=1024`, `SHIP_BINS=100`, `ship_emb_dim=32`.

> **핵심 설계**: 한 턴에 내 행성들이 **독립적으로** 결정하지 않는다. 행성을 **고정 순서(slot index 오름차순)** 로 한 줄로 세우고, 각 행성의 결정을 **앞 행성들이 이미 정한 행동**에 조건화한다(autoregressive). 이 조건화는 **GRU**로 구현한다(시퀀스가 짧아 — K≤MAX_PLANETS — RNN이 적합: 추론이 태생적으로 O(K) incremental, KV-cache 불필요) → "옆 행성이 이미 T를 쳤으니 난 다른 데를 친다" 같은 조율이 가능.

---

## 1. 입력

| 텐서 | shape | 의미 |
|---|---|---|
| `feats` | `[B, N, FEATURE_DIM]` | 행성 N개(`N=MAX_PLANETS`)의 피처 |
| `valid_mask` | `[B, N]` | 보드에 실재하는 행성 (패딩 제외) |
| `mine_mask` | `[B, N]` | 내가 소유한 행성 (발사 출발지 후보 = 결정 대상) |

---

## 2. 인코더 트렁크 — 입력이 `h`가 되기까지

보드 상태를 행성별 컨텍스트 임베딩 `h`로 만든다. ([model.py:85-90](../model.py#L85-L90))

```
feats [B,N,F]
   │
   ▼
embed:  Linear(F→d) → ReLU → Linear(d→d)              # 행성별 임베딩
   │
   ▼  h0 [B,N,d]
TransformerEncoder × n_layers(4)                       # 행성 간 상호작용(self-attention)
   │   ├ src_key_padding_mask = ~valid_mask            # 패딩 행성 무시
   │   └ 각 layer: MHA(8 heads) + FFN(4d)
   ▼
  h  [B,N,d]   ← 행성별 "보드 상태가 반영된" 임베딩
```

`h`는 다른 행성들의 **상태**(소유/병력/위치)를 담는다. 다른 행성들의 **이번 턴 행동**은 아래 AR 디코더가 추가로 주입한다.

`forward(feats, valid_mask)`는 이제 `h`만 반환한다. target/gate/ship logit은 모두 AR 경로(아래)에서 계산된다.

---

## 3. AR GRU — 앞 행동을 보고 결정하기

내 행성을 고정 순서 `p_0, p_1, …, p_{K-1}`(slot index 오름차순)로 처리. 스텝 `k`의 입력과 출력:

```
input_k = h[p_k]  +  action_emb(a_{k-1})        # a_{-1} = start_tok
          └ 누구      └ 직전 행성의 행동
z_k = GRU(input_k, hidden_{k-1})                # hidden state가 a_0..a_{k-1} 요약을 들고 감
```

- `z_k`는 **앞 행성들의 행동 `a_0..a_{k-1}`을 hidden state로 본** source 표현. (자기 행동 `a_k`는 안 봄 → 누수 없음)
- GRU가 자체적으로 causal이라 **positional encoding/causal mask가 필요 없다.**
- 학습(teacher forcing)은 모든 행동을 알므로 **fused `nn.GRU`로 한 번에** 처리([`_ar_decode`](../model.py#L107)), 추론은 행동을 정해가며 **`GRUCell`처럼 한 스텝씩**([`_rollout`](../model.py#L139)) — hidden state만 들고 가므로 **O(K) incremental**(prefix 재계산·KV-cache 불필요).

### action_emb — 행동을 토큰 벡터로 ([model.py:98-108](../model.py#L98-L108))
```
cat([ h[src](d), target_repr(d), ship_repr(32), launched(1) ]) (= 2d+33)
   → Linear(→d) → ReLU → Linear(d→d)  →  action_emb (d)
   target_repr = h[tgt]            (미발사면 noop_tgt)
   ship_repr   = ship_act_emb(bin) (미발사면 noop_ship)
```
출력은 `d_model`(토큰에 더해지므로). `ship_emb_dim=32`는 그 안의 ship bin 부품 크기일 뿐.

---

## 4. `z_k`에서 갈라지는 3갈래 (+ value)

```
                 z_k (앞 행동을 본 source 표현, d)
                       │
        ┌──────────────┼───────────────┐
        ▼              ▼                ▼
    [target]       [gate]            [ship]
  q(z_k)·kᵀ    [z_k, h[tgt]] (2d)  [z_k, h[tgt]] (2d)
        │              │                │
        ▼              ▼                ▼
  타겟 logits[N]   발사 logit(1)    ship-bin logits[100]

   (value는 z_k가 아니라 h 전체 풀링에서: pool(h)(2d) → V)
```

### 4-1. target head — "어디를 칠까?" ([model.py:127-134](../model.py#L127-L134))
```
q = W_q·z_k,   k = W_k·h
target_logits = (q·kᵀ)/√d → [N]      # mask: 자기 자신 + invalid → -inf
```
- 쿼리가 `h[src]`가 아니라 **`z_k`** → 타겟 선택이 **앞 행성들의 행동에 조건화됨**.

### 4-2 / 4-3. gate + ship head — "쏠까? 몇 척?" ([model.py:66-70](../model.py#L66-L70), [:136-139](../model.py#L136-L139))
```
out = gate_ship_head( concat(z_k, h[tgt]) )  (2d → d → 1+100)
gate_logit = out[0]          # 발사 여부
ship_logits = out[1:]        # 100개 bin categorical
bin → fraction:  bin/(SHIP_BINS-1)           # bin_to_fraction
```
- **하나의 head가 gate와 ship을 함께** 출력. 입력 source가 `z_k`이므로 gate/ship 둘 다 앞 행동을 본다.
- target이 **먼저** 정해진 뒤 `h[tgt]`를 조건으로 gate/ship을 구한다.

### 4-4. value head — 상태가치 (critic) ([model.py:78-83](../model.py#L78-L83))
```
pooled = concat(mean_pool(h), max_pool(h)) (2d)   # valid_mask로 마스킹, z_k 무관
V = Linear(2d→d) → ReLU → Linear(d→1)
```

---

## 5. 의사결정 흐름 (행성 간 AR + 행성 내 target→gate/ship)

```
 p_0 ─decode→ z_0 → target_0 → gate_0/ship_0 ─┐  행동 a_0
                                              │  action_emb(a_0)
 p_1 ─decode(z_1는 a_0를 봄)→ target_1 → ...  ─┤  행동 a_1
                                              │
 p_2 ─decode(z_2는 a_0,a_1을 봄)→ ...         ─┘  ...
```

- **행성 간**: `p_k`의 모든 결정(target/gate/ship)은 `z_k`를 통해 `a_0..a_{k-1}`에 조건화. → 완전 autoregressive.
- **행성 내**: `target` 먼저 → `(gate, ship)`은 정해진 target을 조건으로(한 head에서 동시).

한 행성 정책: **P(tgt│z_k) · P(launch│z_k,tgt) · P(frac│z_k,tgt)**

---

## 6. 3가지 호출 경로

### `act` — 배포 ([model.py:201](../model.py#L201)) / `_rollout` ([:142](../model.py#L142))
순차 디코딩. target/ship은 greedy면 argmax(아니면 샘플), gate는 `σ(gate_logit) ≥ threshold`.

### `act_rl` — RL 롤아웃 ([model.py:219](../model.py#L219)) / `_rollout`
순차 디코딩, 전부 샘플. gate는 `Bernoulli(σ(gate_logit + gate_bias))`.
```
record(모든 소유 행성, 처리 순서대로):
   발사  → (planet_i, 1, tgt, bin_idx)
   미발사 → (planet_i, 0, tgt, 0)      # gate 재현용으로 정해진 tgt도 보존
```

### `evaluate_actions` — PPO 업데이트 ([model.py:274](../model.py#L274))
기록을 보드별 `[B,K_max]`로 scatter → **`_ar_decode` 한 번(병렬)** 으로 `z` 계산 → logp/entropy/(teacher)KL를 per-sample `index_add`로 누적.
```
gate    : 모든 소유 행성 (z_k, 기록된 tgt 조건)
target  : launched인 것만 (z_k 쿼리)
ship    : launched인 것만 (gate와 같은 head 출력에서 slice)
teacher 있으면 같은 기록으로 병렬 재생 → KL(teacher‖student)
```
반환: `logp [B]`, `value [B]`, `ent [B]`, (옵션)`kl [B]`.

> **정합성**: `_rollout`(스텝별 GRU)과 `_ar_decode`(fused GRU 한 번)는 동일한 입력 구성·동일한 GRU recurrence(같은 초기 hidden=0)를 쓰므로 같은 `z`를 만든다 → 롤아웃에서 기록한 logp가 업데이트 때 그대로 재현된다. 순서는 가중치가 아닌 **상태(slot index)** 에서만 유도되어 업데이트 후에도 불변.

진단용 greedy 경로는 [`act_debug`](../model.py#L235) (행성별 gate_p/target/ship 반환).

---

## 7. 학습 시 주의 (`act_debug` / SL)

- **teacher forcing**: SL([train_sl.py](../train_sl.py) `compute_losses`)과 RL replay는 `z`를 **사람/기록된 행동**으로 만들고, 추론은 **자기 행동**으로 만든다 → exposure bias 존재(1차 구현은 감수).
- SL의 gate 조건 target: 발사 행성은 사람 target, 미발사 행성은 모델 argmax target.
- **체크포인트 비호환**: AR 파라미터(`ar_gru`/`action_enc`/`ship_act_emb`/`start_tok`/`noop_*`, 약 +0.6M)가 새로 생겨 이전(transformer-AR 포함) 체크포인트는 그대로 못 쓴다 → **SL 재학습 필요**.

---

## 8. 한눈 요약

```
feats ─embed─ TransformerEncoder ─→ h ──────────────────────────┐
                                     │ (행성별 보드 상태)         │
   내 행성 고정 순서로 한 줄 세우기                                │
        input_k = h[p_k] + action_emb(a_{k-1})                  │
                     │ GRU (hidden state가 앞 행동 요약)          │
                     ▼                                           ▼
                    z_k ─┬─ target (q(z_k)·kᵀ)     → 어디로?   value
                         ├─ gate   ([z_k,h[tgt]])  → 쏠까?     (pool h)
                         └─ ship   ([z_k,h[tgt]])  → 몇 척?
```
- **공유**: embed + TransformerEncoder 트렁크(`h`). (인코더는 그대로 transformer, AR 행동 head만 GRU)
- **조율의 비결**: `z_k`가 GRU hidden state를 통해 앞 행성들의 행동을 보고, 그게 target/gate/ship 모두에 흘러든다.
```
