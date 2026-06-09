# Orbit Wars 관측(observation) 포맷

게임 실행 시 매 턴 각 플레이어에게 들어오는 `observation` 구조와, 그 안에서
행성·혜성·함대 정보가 어떤 형태로 오는지 정리한 문서. (코드 출처는 각 항목에 표기)

## 게임 상수 (config)

| 키 | 값 | 의미 |
|---|---|---|
| `episodeSteps` | 500 | 게임 최대 턴 수 |
| `actTimeout` | 1 | 턴당 행동 결정 시간(초) |
| `shipSpeed` | 6.0 | 함대 **최대** 속도 (크기에 따라 1.0~6.0, [physics.py:33-35](physics.py#L33-L35)) |
| `cometSpeed` | 4.0 | 혜성 속도 (units/turn, 경로를 따라 이동) |
| `sunRadius` | 10.0 | 태양 반지름 (중심 (50,50), 함대가 닿으면 소멸) |
| `boardSize` | 100.0 | 보드 크기 (100×100, 좌표 0~100) |

회전 한계 `ROTATION_LIMIT = 50.0`: `중심거리 + 행성반지름 < 50`이면 태양을
중심으로 **공전**, 아니면 **고정**. ([physics.py:22](physics.py#L22), [physics.py:122-133](physics.py#L122-L133))

---

## 1. observation 최상위 구조

Kaggle 리플레이 기준 `steps[t][p] = {action, observation, reward, status, info}`이고,
그 중 `observation`은 다음 키를 가짐 ([replay.py:3-7](replay.py#L3-L7)):

```python
observation = {
    "player":            int,     # 이 관측의 시점(내 player id)
    "step":              int,     # 현재 턴 번호 (0..499)
    "angular_velocity":  float,   # 공전 각속도 w (모든 공전 행성 공통, 강체 회전)
    "planets":           [...],   # 행성+혜성의 현재 상태 (아래 2)
    "fleets":            [...],   # 비행 중인 함대 (아래 3)
    "comets":            [...],   # 혜성 경로 정보 (아래 4)
    "comet_planet_ids":  [...],   # 어떤 planet id가 혜성인지 (아래 4)
}
```

> 읽기 헬퍼: obs가 dict든 속성 객체든 `P._obs_get(obs, key, default)`로 접근
> ([physics.py:28-30](physics.py#L28-L30)).

---

## 2. planets — 행성 **및 혜성**의 현재 상태

`planets`는 리스트이고, 각 원소는 7-튜플 ([features.py:117](features.py#L117)):

```python
[ id, owner, x, y, radius, n_ships, production ]
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | int | 바디 고유 id |
| `owner` | int | 소유자 player id. `-1` = 중립 |
| `x, y` | float | **현재** 위치 (0~100) |
| `radius` | float | 충돌/포획 반지름 |
| `n_ships` | float | 현재 주둔 함선 수 (garrison) |
| `production` | float | 턴당 함선 생산량 (소유 시, 1~5) |

**중요:** 혜성도 이 `planets` 리스트에 똑같은 7-튜플로 들어온다. 즉 혜성도
소유자·주둔 함선·생산력을 가지며 행성처럼 포획 가능하다. 차이는 "움직이는
방식"뿐이고, 그 위치는 `x, y`에 **현재 값만** 주어진다(미래 위치는 아래 4의
경로로 따로 계산해야 함).

---

## 3. fleets — 비행 중인 함대

`fleets`는 리스트, 각 원소는 7-튜플 ([replay.py:7](replay.py#L7), [forecast.py:50](forecast.py#L50)):

```python
[ id, owner, x, y, angle, from_planet_id, ships ]
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | int | 함대 id |
| `owner` | int | 소유자 player id |
| `x, y` | float | 현재 위치 |
| `angle` | float | 진행 방향 (라디안) — 직선 이동 |
| `from_planet_id` | int | 출발 행성 (자기 출발지엔 다시 안 내림) |
| `ships` | float | 함대 규모 (속도는 `fleet_speed(ships)`로 결정) |

함대 속도: `1.0 + 5.0 * (log(ships)/log(1000))**1.5`, 최대 6.0 (ships≥1000에서).
([physics.py:33-35](physics.py#L33-L35))

---

## 4. comets / comet_planet_ids — 혜성 경로

행성 리스트는 혜성의 **현재 위치**만 주므로, 미래 위치는 별도 경로로 온다.

`comet_planet_ids`: 혜성인 planet id들의 리스트(set으로 변환해 사용).
([features.py:99-102](features.py#L99-L102))

`comets`: 혜성 "그룹"의 리스트. 각 그룹 `g`의 구조 ([physics.py:81-90](physics.py#L81-L90)):

```python
g = {
    "paths":       [ path_0, path_1, ... ],   # 그룹 내 각 혜성의 경로
    "path_index":  float,                     # 현재 경로상 위치(턴마다 +1, 보간됨)
    "planet_ids":  [ pid_0, pid_1, ... ],     # 경로와 1:1 대응되는 혜성 id
}
# path_i = [ (x0,y0), (x1,y1), ... ]  턴 단위 위치 점들의 리스트
```

해석: 혜성 `pid_i`의 `k`턴 뒤 위치는 `path_i[path_index + k]` (정수 인덱스 사이는
선형 보간). `path_index + k`가 경로 끝을 넘으면 혜성은 **보드를 떠난 것**(`None`).
([physics.py:103-120](physics.py#L103-L120))

---

## 5. 위치 예측 (파생) — physics.predict_fn

세 종류의 바디가 서로 다르게 움직인다 ([physics.py:93-133](physics.py#L93-L133)). `predict_fn`이
이를 통합해 `predict(k) -> (x, y) 또는 None` 함수를 돌려준다:

| 바디 | 미래 위치 |
|---|---|
| **혜성** (`id ∈ comet_planet_ids`) | 경로 `path`를 따라 이동, 끝나면 `None` (소멸) |
| **공전 행성** (`중심거리+반지름 < 50`, `w≠0`) | `θ = θ₀ + w·k`, 반경 R 고정 회전 |
| **고정 행성** | 위치 불변 |

`build_world(obs)`가 모든 바디에 대해 이 predict를 묶어주고, `forecast.simulate`가
이를 이용해 미래 HORIZON턴의 소유권/주둔 함선 궤적을 만든다. ([forecast.py](forecast.py))

---

## 6. action 포맷 (참고)

행동은 발사 명령들의 리스트 ([replay.py:6](replay.py#L6)):

```python
action = [ [from_planet_id, angle, num_ships], ... ]
```

출발 행성 id, 발사 각도(라디안), 보낼 함선 수. 발사가 없으면 빈 리스트.

---

## 7. feature 추출 시 주의 (현재 상태)

`features.extract`가 위 obs를 신경망 입력으로 변환한다 ([features.py:82-](features.py#L82)).
정적 블록(현재 상태) + 시계열 블록(forecast 미래 궤적)으로 구성.

**위치 궤적으로 motion 표현**: 속도(vx,vy) 채널 대신, 시계열 블록의 매 미래 턴마다
그 바디의 위치 `(x, y)`(중심 기준 정규화)를 함께 넣는다. forecast가 이미 매 턴
위치를 계산하므로 추가 비용이 거의 없고, 이 한 가지로:
- **혜성** = 경로를 따라가는 실제 궤적이 그대로 담긴다 (경로가 휘어도 정확).
- **공전 행성** = 원호 궤적이 담긴다.
- **고정 행성** = 전 턴 동일 좌표.

→ 시계열 채널 `TEMP_CH = OWNER_CH + 3` = `[소유권 5 + ships + x + y]` = 8.
   ([features.py](features.py), [forecast.py simulate](forecast.py#L104))
