# OrbitNet 구조 & 데이터 흐름

현재 [model.py](../model.py) 기준. 입력이 어디를 타고 흘러 최종 행동(발사 여부 / 타겟 / 함대 비율)이 되는지 정리.

설정: `d_model=256`, `n_heads=8`, `n_layers=4`, `dim_feedforward=4·d_model=1024`, `SHIP_BINS=100`.

---

## 1. 입력

| 텐서 | shape | 의미 |
|---|---|---|
| `feats` | `[B, N, FEATURE_DIM]` | 행성 N개(`N=MAX_PLANETS`)의 피처 |
| `valid_mask` | `[B, N]` | 보드에 실재하는 행성 (패딩 제외) |
| `mine_mask` | `[B, N]` | 내가 소유한 행성 (발사 출발지 후보) |

---

## 2. 공유 트렁크 — 입력이 `h`가 되기까지

모든 head가 공유하는 단 하나의 표현 `h`를 만든다. ([model.py:62-80](../model.py#L62-L80))

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
  h  [B,N,d]   ← 행성별 "컨텍스트 반영된" 임베딩
```

이 `h` 하나가 아래 4갈래로 분기한다.

---

## 3. `h`에서 갈라지는 4갈래 (3 head + value)

```
                          h [B,N,d]
                             │
        ┌────────────────┬───┴────────────┬──────────────────┐
        ▼                ▼                 ▼                  ▼
     [gate]          [target]           [ship]            [value]
   h[src] (d)      q·kᵀ attention   [h[src],h[tgt]] (2d)  pool(h) (2d)
        │                │                 │                  │
        ▼                ▼                 ▼                  ▼
  발사 logit (1)   타겟 logits [N]   ship-bin logits[100]   상태가치 V (스칼라)
```

### 3-1. gate head — "이 행성, 쏠까?" ([model.py:36-40](../model.py#L36-L40), [:78](../model.py#L78))
```
h[src] (d) → Linear(d→d) → ReLU → Linear(d→1) → gate_logit
```
- 입력: **소스 행성 임베딩 하나** (타겟은 아직 모름)
- 소유 행성마다 **독립 Bernoulli** → 한 턴에 여러 행성 동시 발사 가능

### 3-2. target head — "어디를 칠까?" ([model.py:69-76](../model.py#L69-L76))
```
q = W_q·h,  k = W_k·h
target_logits = (q·kᵀ) / √d        # [B,N,N], 행성 i→j 점수
   └ mask: 자기 자신(eye) + invalid 행성 → -inf
```
- attention 한 방으로 모든 (소스→타겟) 점수를 동시에 산출
- 소스별로 softmax → 타겟 하나 선택

### 3-3. ship head — "몇 척 보낼까?" ([model.py:43-47](../model.py#L43-L47), [:83-88](../model.py#L83-L88))
```
pair = concat(h[src], h[tgt]) (2d) → Linear(2d→d) → ReLU → Linear(d→100) → ship_logits
bin → fraction:  bin / (SHIP_BINS-1)               # [model.py:90-93]
```
- 입력: **(소스, 타겟) 쌍** → 타겟 의존적 (약한 중립 vs 센 적에 따라 비율 달라짐)
- 100개 bin에 대한 categorical

### 3-4. value head — 상태가치 (critic) ([model.py:49-60](../model.py#L49-L60))
```
pooled = concat(mean_pool(h), max_pool(h)) (2d)    # valid_mask로 마스킹
V = Linear(2d→d) → ReLU → Linear(d→1)
```

---

## 4. 의사결정 흐름 (autoregressive 3단 분해)

소유 행성마다 **gate → target → ship** 순서로 조건부 결정:

```
            소유 행성 src
                 │
                 ▼
        ┌─ P(launch | src) ──────────────┐   gate (h[src])
        │                                │
   launch=0                         launch=1
   (아무것도 안 함)                       │
                                         ▼
                                P(tgt | src)            target attention
                                         │
                                         ▼
                                P(frac | src, tgt)      ship (h[src],h[tgt])
                                         │
                                         ▼
                                 (src, tgt, frac) 발사
```

전체 정책: **P(launch│src) · P(tgt│src) · P(frac│src,tgt)**

핵심: gate는 **타겟을 모르는 상태**에서, ship은 **타겟을 정한 뒤** 호출된다 → 둘이 입력(d vs 2d)과 시점이 다르다.

---

## 5. 3가지 호출 경로

같은 `forward(h)`를 쓰지만 용도가 다르다.

### `act` — greedy 배포 ([model.py:101-138](../model.py#L101-L138))
```
gate_p = σ(gate_logit)
fire   = gate_p ≥ gate_threshold          # 임계값 컷
발사분만 → tgt = argmax(target_logits)
         → frac_bin = argmax(ship_logits)
(greedy=False면 argmax 대신 샘플)
```

### `act_rl` — RL 롤아웃 (항상 샘플) ([model.py:140-188](../model.py#L140-L188))
```
gate_p   = σ(gate_logit + gate_bias)      # bias로 발사율 보정
launched = Bernoulli(gate_p)
발사분만 → tgt      = Categorical(target_logits).sample()
         → bin_idx  = Categorical(ship_logits).sample()

record(모든 소유 행성):
   발사  → (planet_i, 1, tgt, bin_idx)
   미발사 → (planet_i, 0, -1, 0)
```

### `evaluate_actions` — PPO 업데이트 ([model.py:190-263](../model.py#L190-L263))
저장된 record로 logp / entropy / (teacher) KL를 **per-sample(B) 단위 `index_add`**로 누적:

```
gate    : 항상 누적 (모든 소유 행성)        # [model.py:217-222]
target  : launched인 것만 (m = launched>0.5) # [model.py:234-239]
ship    : launched인 것만                    # [model.py:242-247]
teacher 있으면 같은 구조로 KL(teacher‖student):
   gate → Bernoulli KL,  target/ship → categorical KL
```

반환: `logp [B]`, `value [B]`, `ent [B]`, (옵션)`kl [B]`

---

## 6. 한눈 요약

```
feats ─embed─ Transformer ─→ h ─┬─ gate   (h[src])        → 쏠까?
                                ├─ target (q·kᵀ)           → 어디로?
                                ├─ ship   (h[src],h[tgt])  → 몇 척?
                                └─ value  (pool h)         → 이 상태 가치?
```
- **공유**: embed + Transformer 트렁크 (`h`)
- **분리 이유**: gate/target/ship이 autoregressive 분해의 **서로 다른 조건부 레벨**(입력·시점·개수)에 위치
