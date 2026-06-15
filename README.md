# Orbit Wars Agent

행성 점령 게임(Orbit Wars)을 위한 신경망 에이전트. 인간 리플레이로 지도학습(SL) →
self-play 강화학습(RL)으로 이어지는 AlphaStar식 파이프라인.

모든 명령은 conda 환경 `orbit`(Python 3.12)에서 실행한다.

```bash
conda run -n orbit python <script>.py
```

---

## 1. 지도학습 (SL) — 인간 승자 모방

파싱/캐싱과 학습은 **분리된 2단계**다. 먼저 `dataset.py`로 리플레이를 파싱해
디스크 store로 캐싱하고, 그다음 `train_sl.py`가 그 캐시만 읽어 학습한다.
모델이 예측하는 것: 행성별 **발사 여부(gate)** + **타겟** + **함선 수(보유량 대비 비율)**.

### (선택) 데이터 다운로드

`data/manifest.csv` 기준 최신 N개 일자 데이터셋을 zip(압축) 상태로 받는다.

```bash
./download_latest.sh        # 최신 5개 (default)
./download_latest.sh 3      # 최신 3개
```

→ `data/zips/*.zip` 에 저장 (Kaggle CLI 인증 필요: `~/.kaggle/kaggle.json`).

### 1-1. 파싱 → 캐시 (`dataset.py`)

```bash
conda run -n orbit python dataset.py                 # data/*.json + data/zips/*.zip 전부
conda run -n orbit python dataset.py --rebuild       # 캐시 강제 재생성
conda run -n orbit python dataset.py --workers 8     # 파싱 병렬 수
conda run -n orbit python dataset.py --limit 200     # 에피소드 200개만 (빠른 점검)
conda run -n orbit python dataset.py a.json b.zip    # 특정 파일만 (.json/.zip)
```

- 캐시: `data/parsed_cache/` (memmap, owner는 int8 압축). `PARSER_VERSION` 올리면 자동 재생성.
- **zip은 추출하지 않고** 멤버를 메모리에서 한 에피소드씩 읽어 파싱 → 디스크엔 zip + 캐시만.
- `ORBIT_CACHE_DIR` 로 캐시 위치 변경 가능 (feature store가 ~20+ GB).

### 1-2. 학습 (`train_sl.py`)

```bash
conda run -n orbit python train_sl.py                # 위에서 만든 캐시로 학습
```

- 캐시가 없으면 `dataset.py`를 먼저 돌리라고 안내하고 종료한다.
- 출력: `orbit_net.pt` (val loss 최저 best), `train_log.csv`

### SL 모델 점검

한 에피소드에서 gate/타겟/함선 예측 vs 인간 실제 행동 + **게이트 임계값 sweep**:

```bash
conda run -n orbit python inspect_episode.py                 # 첫 리플레이
conda run -n orbit python inspect_episode.py data/X.json     # 특정 리플레이
conda run -n orbit python inspect_episode.py data/X.json --all   # 발사 안 한 step까지
```

---

## 2. 강화학습 (RL) — 스냅샷 풀 league

SL 모델(`orbit_net.pt`)로 초기화한 learner를 **자기 과거 best들의 league**와 붙여 A2C로 학습.
상대 = **최근 best 2개 + 고정 SL teacher**. 매 iter `--games`판을 **2인:4인 = 3:1**로 돌린다
(2인은 셋 중 하나와, 4인은 셋 다와). **SL 가중치(`orbit_net.pt`)는 안 건드린다.**

```bash
conda run -n orbit python train_rl.py --iters 200 --games 8 --workers 8
conda run -n orbit python train_rl.py --games 20 --win-gate 0.55   # 저장 임계값 조절
```

- `--iters` 수집+업데이트 사이클 / `--games` iter당 총 게임(4의 배수 권장) / `--workers` 병렬 롤아웃
- `--win-gate`(기본 0.5): league 승률이 이걸 넘으면 `orbit_net_rl.pt` 저장 + pool에 push
- `SHAPE_COEF`(코드 모듈 상수, 기본 0): 매 턴 **함선 점유율 변화**를 dense 보상(PBRS)으로 (0이면 종료 ±1만). 기본은 run 내내 상수이며, `SHAPE_DECAY_FRAC>0`이면 감쇠
- 출력: `orbit_net_rl.pt`(gated best, main_rl이 로드), `orbit_net_rl_last.pt`(최신), `rl_pool/`(스냅샷), `train_rl_log.csv`(per-iter 지표)
- learner는 `orbit_net_rl.pt` 있으면 거기서 **이어서(resume)** 시작, pool도 디스크에서 복원
- 곡선: `python plot_rl_log.py` → `train_rl_curves.png` (승률 / loss / entropy)

### 학습 로그 읽는 법

매 iter 한 줄로 찍힌다:
```
iter   12  steps=1840  rollout=22s update=2s  win=0.55 (2p=0.67 4p=0.25 vsSL=4/5)  policy=-0.2167 value=0.2835 entropy=2.858
```

| 항목 | 의미 |
|------|------|
| `steps` | 이번 iter에 모은 learner 스텝(학습 샘플) 수 |
| `rollout` / `update` | 게임 수집 / A2C 업데이트에 걸린 초 |
| `win` | league 종합 승률 (2인+4인). 상대가 점점 세져서 ~0.5 근처로 수렴 |
| `2p` / `4p` | 2인 / 4인 승률 (랜덤 baseline 2인 0.5, 4인 0.25) |
| **`vsSL`** | **고정 SL teacher와의 2인전 승/판** (예 4/5). teacher는 안 변하니 **시점 비교 가능한 절대 지표** — 이게 우상향하면 진짜 강해지는 것 |
| `policy` | 정책 손실 `-(logπ·advantage)`. **음수 가능**, 절대값은 의미 작음(대리 목적함수). 성능은 win/vsSL로 판단 |
| `value` | critic 손실 = MSE(return, value). **낮을수록** critic이 결과를 잘 예측 → advantage가 정확해짐 |
| `entropy` | 정책 엔트로피(탐색량). 학습되며 **서서히 감소**가 정상. 너무 빨리 0이면 조기 수렴, 안 줄면 미학습 |

> 핵심 진행 지표는 **`vsSL`**(절대) > `win`(상대). `policy` 수치 자체에 일희일비하지 말 것.

---

## 3. 게임 실행 / 리플레이

`run_game.py`로 한 판 돌리고 `replay.html` 저장 (브라우저로 열람).

```bash
conda run -n orbit python run_game.py                       # SL vs SL (기본)
conda run -n orbit python run_game.py main_rl.py main.py     # RL vs SL
conda run -n orbit python run_game.py main.py random         # SL vs 랜덤
conda run -n orbit python run_game.py main.py main.py main.py main.py   # 4인
```

- `main.py` → `orbit_net.pt`(SL), `main_rl.py` → `orbit_net_rl.pt`(RL) 로드

---

## 4. 캐글 제출

제출물은 **루트에 `main.py`(agent 함수 포함)** 가 있어야 한다. 우리 에이전트는
모델 가중치 + 여러 모듈을 쓰므로 **multi-file `tar.gz` 번들**로 제출한다.

### 준비 (1회)

```bash
pip install kaggle                       # CLI
# 인증: ~/.kaggle/kaggle.json (또는 ~/.kaggle/access_token) 배치
kaggle competitions list -s "orbit wars" # 인증 확인
```

그리고 웹에서 규칙 동의(필수): https://www.kaggle.com/competitions/orbit-wars → "Join Competition".
참가 확인: `kaggle competitions list --group entered`

### 번들 만들기 + 제출

`main.py`가 import하는 파일과 가중치를 한 tar.gz로 묶는다 (main.py가 루트에 오게):

```bash
# SL 에이전트 (orbit_net.pt)
tar -czf submission.tar.gz main.py model.py features.py forecast.py physics.py orbit_net.pt
kaggle competitions submit orbit-wars -f submission.tar.gz -m "SL agent v1"
```

RL 에이전트는 진입점이 `main_rl.py`이고 가중치가 `orbit_net_rl.pt`인데,
**제출 루트의 파일명은 `main.py`여야** 한다. 루트의 `main.py`(SL 에이전트)를
덮어쓰지 않도록, **스테이징 폴더 `rl_submission/`** 에 현재 RL 코드와 가중치를
복사하고 그 안에서 `main_rl.py`를 `main.py`로 넣어 번들링한다:

```bash
mkdir -p rl_submission
# 현재 학습 코드 + 가중치를 스테이징 폴더로 복사 (구버전과 섞이지 않게 매번 복사)
cp model.py features.py forecast.py physics.py orbit_net_rl.pt rl_submission/
cp main_rl.py rl_submission/main.py        # 진입점만 main.py로 (루트 main.py는 그대로)
# 폴더 내부 기준(-C)으로 압축해서 main.py가 tar 루트에 오게 한다
tar -czf submission_rl.tar.gz -C rl_submission \
    main.py model.py features.py forecast.py physics.py orbit_net_rl.pt
kaggle competitions submit orbit-wars -f submission_rl.tar.gz -m "RL agent v1"
```

> 번들에 들어갈 파일은 `main.py`(=`main_rl.py`)의 import(`model`/`features`/`forecast`/`physics`)와
> 로드하는 `*.pt` 가중치다. 새 모듈을 추가하면 복사 목록과 tar 목록 모두에 넣어야 한다.
> **주의:** `rl_submission/`의 `.py`는 반드시 루트(학습 코드)에서 매번 복사해 동기화할 것.
> 안 그러면 폴더에 남은 구버전 `model.py`(예: `n_layers`가 다른)로 가중치가 어긋나 로드되어
> 망가진 에이전트가 제출된다.

### 제출 전 로컬 점검

```bash
conda run -n orbit python run_game.py main.py random   # SL: 에러 없이 한 판 도는지
# RL 번들 점검 (스테이징 폴더에서 그대로 실행 -> 제출과 동일한 코드/가중치):
conda run -n orbit python run_game.py rl_submission/main.py random
```

### 모니터링

```bash
kaggle competitions submissions orbit-wars      # 제출 상태 + submission ID
kaggle competitions episodes <SUBMISSION_ID>    # 플레이한 게임들
kaggle competitions leaderboard orbit-wars -s   # 리더보드 상위
```

---

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `features.py` | obs → feature 텐서 (정적 + HORIZON turn 시계열, 위치 포함) |
| `forecast.py` | 함대 궤적/보드 forward-simulation |
| `physics.py` | 기하/속도/충돌 (`swept_dist`) |
| `replay.py` | 리플레이 → SL 라벨 (gate/타겟/비율) |
| `model.py` | OrbitNet (attention 정책 + gate/target/ship + value 헤드) |
| `dataset.py` | 리플레이(.json/.zip) 파싱 → 디스크 캐시 + `DiskDataset` |
| `train_sl.py` / `train_rl.py` | SL(순수 학습) / RL 학습 |
| `download_latest.sh` | manifest 기준 최신 N일 데이터셋 다운로드 |
| `main.py` / `main_rl.py` | kaggle 에이전트 (SL / RL) |
| `inspect_episode.py` | 모델 예측 점검 + 게이트 임계값 sweep |