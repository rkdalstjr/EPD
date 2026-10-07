좋습니다. RSI 철학으로 **기획안 v1.1 재작성 → 코드 반영 → 실행 스크립트 갱신**까지 한 번에 갑니다.

---

# Part 1. `docs/RESEARCH_SPEC.md` v1.1 (전면 재작성)

```markdown
# RESEARCH_SPEC.md — EPD 프로젝트 연구 헌법

> **동결 선언**: 본 문서는 실험 중 수정 금지. 수정 필요 시 `RESEARCH_SPEC_v2.md` 생성.
> **철학**: RSI처럼 **어떤 자산·어떤 시대에도 동일 파라미터로 작동**하는 지표를 목표로 한다.
> **AI/ML 최적화 금지**: 파라미터 탐색·튜닝·ML 피처 결합은 이 지표의 정의에 반한다.

**Version**: 1.1 (RSI 철학 반영)
**Date**: 2026-10-06
**Supersedes**: v1.0

---

## 0. 설계 원칙 (RSI 4원칙)

| 원칙 | 내용 | EPD 구현 |
|---|---|---|
| **P1. 유계 출력** | 출력이 유한 범위. 임계값이 절대값 | `tanh` 변환 → EPD ∈ (-1, +1), EPD_100 ∈ (0, 100) |
| **P2. 이론적 파라미터** | 최적화가 아니라 문헌·주기 근거로 고정 | §3.4 참조, **탐색 금지** |
| **P3. 스케일 불변** | 자산 가격대에 무관 | 로그수익률 + 롤링 z-score |
| **P4. 학습 없음** | 순수 공식. ML 피처 결합 금지 | 회귀·GBM 피처 사용 금지 (사용자 선택은 자유, 지표 정의엔 불포함) |

**P1·P2·P4는 절대 규칙.** 위반 시 EPD는 RSI급 보편성을 주장할 수 없다.

---

## 1. 프로젝트 개요

### 1.1 한 줄 목표

> **"가격 동역학과 순열 엔트로피 동역학의 불일치(divergence)를, 어떤 자산·시대에도 동일 파라미터로 작동하는 단일 지표로 확정한다."**

### 1.2 범위

| 포함 | 제외 |
|---|---|
| EPD 단일 지표 개발·검증 | TRA, KSE (추후) |
| 합성 + 실데이터 + 다자산 보편성 | 3차원 국면 공간 |
| **파라미터 보편성 (튜닝 아님)** | **파라미터 최적화** |
| 지표 스펙 문서화 | ML 피처 결합, 백테스트 자동화 |

### 1.3 산출물

```
① RESEARCH_SPEC.md         — 본 문서 (동결)
② EPD_SPEC.md              — 지표 스펙 (Stage 3 후)
③ epd_core/                — 패키지 (Stage 2)
④ synthetic_report.md      — Stage 1 결과
⑤ universality_report.md   — Stage 3 결과 (신규)
⑥ realdata_report.md       — Stage 4~5 결과
⑦ walkforward_report.md    — Stage 6 결과
⑧ final_decision.md        — 생존/폐기 판정
```

---

## 2. 연구 질문과 가설

### 2.1 Research Questions

| ID | 질문 | Stage |
|---|---|---|
| **RQ1** | 가격-엔트로피 divergence가 안정적 구조로 존재하는가? | 1, 4 |
| **RQ2** | 그 divergence가 momentum·volatility·PE와 **직교**하는 정보를 갖는가? | 5 |
| **RQ3** | **동일 파라미터**로 여러 자산·시대에 이식 가능한가? | 3, 6 |

### 2.2 가설

| 가설 | 내용 | 검증 |
|---|---|---|
| **H1** | EPD_100 극단값(>70 또는 <30) 이후 추세 지속성이 약화된다 | 분위별 forward return 단조성 |
| **H2** | \|EPD\| 극단값은 미래 변동성 증가와 연결된다 | quintile vs future vol |
| **H3** | EPD는 기존 지표와 **직교**한다 (중복 아님) | `corr(EPD, MOM) < 0.5` AND `corr(EPD, VOL) < 0.5` |

**Primary endpoint**: H1 ∧ H3
**Secondary**: H2

**H3 재정의(v1.1)**: "회귀 계수 유의성"이 아니라 **"직교성"**이 목적. RSI가 그렇듯, 다른 지표를 대체하는 게 아니라 **다른 것을 본다**는 증거면 충분.

### 2.3 실패 조건 (사전 고정, v1.1 강화)

| 조건 | 판정 |
|---|---|
| **A. 구조적 중복** | `corr(EPD, MOM) > 0.5` OR `corr(EPD, VOL) > 0.5` |
| **B. Null FP** | White noise 1000블록 중 \|IC\|>0.05 가 5% 이상 |
| **C. OOS 소멸** | IS Sharpe 개선 > 0.2, OOS 개선 < 0.05 |
| **D. 파라미터 비보편성** ★ | **36조합 중 90% 미만이 동일 방향이면 FAIL** |
| **E. 비용 후 소멸** | 왕복 0.4354% 반영 후 초과수익 < 0 |

**D가 v1.1에서 대폭 강화됨.** v1.0은 "로버스트 영역의 중심값"을 골랐지만, 이는 최적화의 변종이다. v1.1은 **모든 조합이 같은 결론**이어야 한다. RSI가 14에서만 되고 20에서는 안 되면 RSI가 아니다.

**E가 가장 강력한 폐기 조건.**

---

## 3. EPD 정의 (v2, 확정)

### 3.1 최종 공식

```python
# 1) 로그수익률
r_t = log(P_t / P_{t-1})

# 2) 순열 엔트로피 (Bandt-Pompe 2002)
PE_t = permutation_entropy(r[t-w_pe : t], m, tau)

# 3) 엔트로피 변화율
ΔPE_t = PE_t - PE_{t-1}

# 4) Causal 롤링 z-score (shift 1 적용, 미래 정보 누출 금지)
Z_r(t) = (r_t - μ_r(t)) / σ_r(t)        # μ,σ from r[t-w_z : t-1]
Z_e(t) = (ΔPE_t - μ_e(t)) / σ_e(t)

# 5) 원시 divergence
EPD_raw = Z_r - Z_e

# 6) RSI식 유계 변환 ★ (v1.1 신규)
EPD     = tanh(EPD_raw / 2)                # ∈ (-1, +1)
EPD_100 = 50 * (1 + tanh(EPD_raw / 2))     # ∈ (0, 100)  ← 주 사용 스케일
```

### 3.2 스케일 근거

`tanh(x/2)`는 자의적 상수가 아니다:
- EPD_raw는 이론적으로 `N(0, √2)` 근사 (Z_r, Z_e 독립 가정 시)
- `tanh(x/2)`에서 `x=2` → 0.76 → **EPD_100 = 88** (극단)
- `x=1` → 0.46 → **EPD_100 = 73** (과열 진입)
- `x=0` → 0 → **50** (중립)
- `x=-1` → **27**, `x=-2` → **12**

→ **±1σ가 73/27, ±2σ가 88/12** 로 RSI의 70/30 관례와 자연스럽게 정렬. `tanh_scale=2`는 **변환 상수**이며 튜닝 대상이 아니다. (RSI의 "100 스케일"과 동일 성격)

### 3.3 부호 규약 (Stage 4에서 최종 확정)

| EPD_100 | 해석 (가설) | RSI 유사 |
|---|---|---|
| **> 70** | 가격이 엔트로피보다 강함 → **과열 위험** | 과매수 |
| 30 ~ 70 | 정상 | 중립 |
| **< 30** | 엔트로피가 가격보다 강함 → **구조 형성 중** | 과매도 |

**주의**: v1.0과 동일하게 Stage 4에서 뒤집힐 수 있음. 데이터가 결정.

### 3.4 파라미터 (이론적 고정, 탐색 금지)

| 파라미터 | **고정값** | 이론적 근거 |
|---|---|---|
| `m` (PE order) | **3** | Bandt-Pompe 2002 원논문. 3! = 6 패턴, 통계적 추정 안정. m=4는 24패턴으로 과다. |
| `tau` (delay) | **1** | 일봉 표준. 인접 관측 사용. |
| `w_pe` (PE window) | **60** | m=3에서 6패턴 × 최소 10관측 = 60. 분기(quarter) 주기와 일치. |
| `w_z` (z window) | **252** | 1년 거래일. 자연 주기. |
| `tanh_scale` | **2** | §3.2 참조. 변환 상수. |

**이 값들은 튜닝 대상이 아니다.** Stage 3은 이 값들의 **보편성을 검증**하는 것이지, 최적값을 찾는 게 아니다.

### 3.5 출력 스키마

```python
{
    "epd": EPD,               # ∈ (-1, +1)
    "epd_100": EPD_100,       # ∈ (0, 100)  ← 주 사용
    "epd_raw": EPD_raw,       # 원시값 (디버깅용)
    "epd_abs": |EPD|,
    "alignment": sign(Z_r × Z_e),
    "return_z": Z_r,
    "entropy_z": Z_e,
    "pe": PE_t,
    "dpe": ΔPE_t,
    "params": {...}
}
```

---

## 4. 개발 단계 (Stage 0 ~ 8)

### Stage 0 — Research Spec (1일)
**산출물**: 본 문서. **동결.**

---

### Stage 1 — Synthetic Validation (2일)

**목적**: EPD가 "의도한 현상"을 측정하는지 확인.

**Test A: White Noise** — EPD_100과 future return/vol 무관. |IC|>0.05 비율 < 5% → PASS
**Test B: GARCH(1,1)** — return IC ≈ 0 (vol 반응은 정상)
**Test C: Regime Switching** — 국면 전환 ±5일 내 |EPD_100−50|>20 발생률 ≥ 50%
**Test D: Synthetic Divergence (핵심)** — 후반부 EPD_100이 유의하게 50에서 이탈

**판정**: 4개 모두 PASS → Stage 2. D 실패 시 공식 재설계.

**산출물**: `synthetic_report.md`

---

### Stage 2 — EPD MVP 구현 (1일)

**산출물**: `packages/epd_core/` 패키지
```
packages/epd_core/
├── pyproject.toml
└── src/epd_core/
    ├── __init__.py
    ├── permutation.py
    ├── _core.py
    └── epd.py       ← compute_epd(bounded=True) 기본
```

**API**:
```python
def compute_epd(returns, close=None,
                w_pe=60, m=3, tau=1, w_z=252,
                bounded=True, tanh_scale=2.0) -> dict
```

**Causal 보장**: `rolling_zscore`는 `min_periods=w_z`, `shift(1)`.
**테스트**: `tests/test_epd.py`.

---

### Stage 3 — Universality Check (2일) ★ v1.1 재정의

**v1.0 (폐기)**: 36조합 → "로버스트 영역의 중심값"을 기본으로.
**v1.1 (신규)**: 36조합 **전부가 같은 방향**이어야 함.

**절차**:
1. `w_pe ∈ {20, 60, 120}`, `m ∈ {3, 4}`, `w_z ∈ {60, 252, 504}`, `price ∈ {return, slope}` = 36조합
2. 각 조합에 대해 5분위 forward return 계산
3. 단조성 점수 (Spearman rank corr of quintile means) 부호 확인
4. **판정**:
   - **PASS**: 36개 중 **33개 이상(≥90%)** 이 동일 부호
   - **FAIL**: 하나라도 방향이 다르면 실패 조건 D

**"중심값 선택" 없음.** 실패 시 그냥 폐기.

**산출물**: `universality_report.md` + 히트맵

---

### Stage 4 — Real Data Discovery (3일)

**데이터**: KOSPI(005930), KOSDAQ, S&P500, NASDAQ / 2005-2025 / 일봉

**분석 (v1.1 변경)**:
```python
# 기존: 5분위만
# 수정: 5분위 + 절대 임계값 분석 병행

# (a) 임계값 기반
mask_high = EPD_100 > 70
mask_low  = EPD_100 < 30
fwd_ret_high = fwd_ret[mask_high].mean()
fwd_ret_low  = fwd_ret[mask_low].mean()

# (b) 5분위 (참고용)
quintiles = qcut(EPD_100, 5)
```

**판정 (H1)**: 임계값 기반 forward return 차이가 p<0.05 (Newey-West) + 5분위 단조성 |ρ|>0.7

**핵심**: 임계값 70/30이 **자산 바뀌어도 동일하게** 작동해야 함. 자산별로 임계값 조정했다면 즉시 FAIL.

**산출물**: `realdata_report.md`

---

### Stage 5 — Orthogonality Check (3일) ★ 재정의

**v1.0**: `future_ret ~ MOM + EPD` 회귀, 계수 유의성.
**v1.1**: **직교성** 확인. 계수 유의성은 부차적.

**분석**:
```python
# 1) 상관계수 (핵심)
corr_epd_mom = corr(EPD_100, MOM_20d)
corr_epd_vol = corr(EPD_100, VOL_20d)
corr_epd_pe  = corr(EPD_100, PE_60)

# 2) 회귀 (참고)
M1: future_ret ~ MOM_20d
M4: future_ret ~ MOM_20d + EPD_100
```

**판정 (H3)**:
- `|corr(EPD, MOM)| < 0.5` **AND**
- `|corr(EPD, VOL)| < 0.5` **AND**
- `|corr(EPD, PE)| < 0.5`

**모두 만족 → 직교성 PASS.** 회귀 R² 증가는 보너스일 뿐, 필수 아님.

**이유**: RSI도 MOM과 완전 독립은 아니다. 그러나 **0.5 미만이면 다른 것을 본다**고 주장 가능.

**산출물**: 상관행렬 + 회귀표

---

### Stage 6 — Walk-Forward Validation (2일)

**설계**:
```
Train: 2005-2014 (파라미터 변경 없음, 그냥 확인)
Val:   2015-2017 (임계값 70/30 확정)
Test:  2018-2025 (절대 손대지 않음)
```

**v1.1 핵심**: **모든 자산에 동일 파라미터·동일 임계값** 적용. 자산별 커스터마이징 금지.

**평가**:
- IC (Spearman)
- Q5−Q1 Sharpe
- **거래비용 반영 후 Sharpe** (왕복 0.4354%)

**판정**:
- 비용 전 Sharpe > 0.5 AND 비용 후 Sharpe > 0.2 → PASS
- 비용 후 Sharpe < 0 → 실패 조건 E

**산출물**: `walkforward_report.md`

---

### Stage 7 — Final Decision (1일)

**최종 판정 매트릭스**:

| 조건 | 결과 |
|---|---|
| Stage 1 통과 (4/4) | ✅ 필수 |
| Stage 3 보편성 ≥90% | ✅ 필수 |
| Stage 4 H1 통과 (임계값 기반) | ✅ 필수 |
| Stage 5 직교성 (모든 corr < 0.5) | ✅ 필수 |
| Stage 6 비용 후 Sharpe > 0.2 | ✅ 필수 |
| 실패 조건 A~E 회피 | ✅ 필수 |

**모두 ✅**: `EPD_SPEC.md` v1.0 확정
**하나라도 ❌**: 원인 문서화 → 재설계 최대 2회 → 3회 실패 시 폐기, TRA로 이동.

---

### Stage 8 — EPD_SPEC 확정 (1일)

```markdown
# EPD_SPEC.md v1.0

## 정의
EPD_raw = Z_r - Z_e
EPD_100 = 50 * (1 + tanh(EPD_raw / 2))

## 파라미터 (고정, 변경 금지)
- w_pe = 60, m = 3, tau = 1
- w_z = 252, tanh_scale = 2

## 임계값 (자산 무관)
- EPD_100 > 70: 과열 위험
- EPD_100 < 30: 구조 형성 중

## 출력 스키마
{epd, epd_100, epd_raw, ...}

## 사용법
- 리스크 필터: EPD_100>70 진입 회피
- 독립 신호: 임계값 크로스
- ML 피처: ❌ 금지 (지표 철학 위반)

## 검증 결과 요약
- 다자산 IC, 비용 후 Sharpe, ...

## 한계
- ...
```

---

## 5. 일정 (총 16일)

| Stage | 기간 | 누적 |
|---|---|---|
| 0. Spec | 1 | 1 |
| 1. Synthetic | 2 | 3 |
| 2. MVP | 1 | 4 |
| 3. **Universality** | 2 | 6 |
| 4. Real Data | 3 | 9 |
| 5. **Orthogonality** | 3 | 12 |
| 6. Walk-Forward | 2 | 14 |
| 7. Decision | 1 | 15 |
| 8. SPEC | 1 | 16 |

**병행 시 3~4주.**

---

## 6. 리스크

| 리스크 | 확률 | 대응 |
|---|---|---|
| Stage 1 White noise FP | 중 | PE 파라미터 재검토 (단, 튜닝 아님) |
| **Stage 3 보편성 실패** | **높음** | **폐기 검토** (RSI급 아님) |
| Stage 4 H1 기각 | 중 | 부호 규약 재정의 |
| Stage 5 직교성 실패 (corr>0.5) | 중 | 폐기 |
| Stage 6 비용 후 소멸 | 높음 | 폐기 |
| **전체 실패** | **40~50%** | TRA로 이동 |

**핵심 인식**: RSI급 보편성 요구는 성공 확률을 **60~70% → 40~50%로 낮춘다.** 이건 정직한 기준이다.

---

## 7. 판정 흐름도

```
Stage 1 (Synthetic, 4 tests)
   └─ 모두 PASS → Stage 2
Stage 2 (MVP) → Stage 3 (Universality ≥90%)
   └─ FAIL → 폐기 검토
Stage 4 (Real Data, 임계값 기반 H1)
   └─ FAIL → 폐기
Stage 5 (Orthogonality, corr<0.5)
   └─ FAIL → 폐기
Stage 6 (Walk-Forward, 비용 후 Sharpe>0.2)
   └─ FAIL → 폐기
Stage 7 → EPD_SPEC 확정
```

**재설계 최대 2회. 3회 실패 → EPD 아이디어 폐기.**

---

## 8. 성공 시 다음 단계 (v1.1 수정)

```
1. EPD_100을 리스크 관리 필터로 사용 (진입 회피)
2. /api/quant/pipeline 노출
3. 프론트 시각화 (RSI처럼 보조지표 창)

❌ 삭제: "EPD를 ML 피처로 추가 (LightGBM)"

→ 6개월 후
4. TRA 개발
5. KSE 개발
6. 3축 독립성 검증 (corr < 0.5)
7. 3차원 국면 공간
```

**ML 피처 결합은 EPD 철학 위반.** 사용자가 개인적으로 시도하는 건 자유지만, EPD_SPEC에 명시하지 않는다.

---

## 9. 지금 할 일

```bash
mkdir -p docs/epd_project
# 파일 생성
# RESEARCH_SPEC.md ← 본 문서
# EPD_SPEC.md     ← DRAFT
# synthetic_report.md, universality_report.md, realdata_report.md
# walkforward_report.md, final_decision.md
```

**지금은 코드 작성까지 진행** (본 세션).

---

## 10. 최종 요약

| 항목 | 값 |
|---|---|
| **목표** | RSI급 보편 지표로 EPD 확정 |
| **출력** | EPD_100 ∈ (0, 100) |
| **임계값** | 70 / 30 (자산 무관) |
| **파라미터** | **고정**. 탐색 금지. |
| **Primary endpoint** | H1 (추세 약화) ∧ H3 (직교성) |
| **성공 확률** | **40~50%** (기존 60~70%에서 하향) |
| **실패 시** | TRA로 이동 |

**핵심 원칙**:

> **"EPD가 RSI처럼 어디서나 작동하는지 정직하게 판정한다. 특정 자산·특정 파라미터에서만 되면 그건 RSI가 아니다."**
```

---

# Part 2. 코드 반영

## 2.1 패키지 구조로 재배치

```bash
cd epd_project
mkdir -p packages/epd_core/src/epd_core
mkdir -p packages/epd_core/tests

# (기존 app/services/entropy/ 파일이 있다면 이동, 없다면 새로 생성)
```

## 2.2 `packages/epd_core/pyproject.toml`

```toml
[project]
name = "epd-core"
version = "0.1.0"
description = "Entropy-Price Divergence (RSI-style bounded indicator)"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/epd_core"]
```

## 2.3 `packages/epd_core/src/epd_core/permutation.py`

(v1.0과 동일 — 변경 없음)

```python
"""순열 엔트로피 (Bandt-Pompe 2002)."""
from __future__ import annotations
import numpy as np
from math import factorial, log


def _ordinal_pattern(vec: np.ndarray) -> tuple:
    return tuple(np.argsort(vec, kind="stable").tolist())


def permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    """
    정규화된 순열 엔트로피 ∈ [0, 1].
    - 모두 같은 패턴 → 0
    - 균일 분포 → 1
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    span = (m - 1) * tau + 1
    if n < span:
        return np.nan

    counts: dict[tuple, int] = {}
    total = 0
    for i in range(n - span + 1):
        vec = x[i : i + span : tau]
        key = _ordinal_pattern(vec)
        counts[key] = counts.get(key, 0) + 1
        total += 1

    if total == 0:
        return np.nan

    p = np.fromiter(counts.values(), dtype=float, count=len(counts)) / total
    h = -np.sum(p * np.log(p))
    h_max = log(factorial(m))
    return float(h / h_max) if h_max > 0 else np.nan
```

## 2.4 `packages/epd_core/src/epd_core/_core.py`

```python
"""롤링 유틸: PE, causal z-score, slope."""
from __future__ import annotations
import numpy as np
from .permutation import permutation_entropy


def rolling_pe(returns: np.ndarray,
               w_pe: int = 60,
               m: int = 3,
               tau: int = 1) -> np.ndarray:
    """PE_t = permutation_entropy(r[t-w_pe:t]). Causal."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    out = np.full(n, np.nan)
    for t in range(w_pe, n):
        out[t] = permutation_entropy(r[t - w_pe : t], m=m, tau=tau)
    return out


def rolling_zscore(x: np.ndarray,
                   w_z: int = 252,
                   min_periods: int | None = None) -> np.ndarray:
    """Causal rolling z-score: z_t = (x_t - μ_{t-w_z:t}) / σ_{t-w_z:t}."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if min_periods is None:
        min_periods = w_z
    out = np.full(n, np.nan)
    for t in range(n):
        lo = t - w_z
        if lo < 0:
            continue
        win = x[lo:t]
        win = win[~np.isnan(win)]
        if win.size < min_periods:
            continue
        mu = win.mean()
        sd = win.std(ddof=1)
        if not np.isfinite(sd) or sd == 0.0:
            out[t] = 0.0
        else:
            out[t] = (x[t] - mu) / sd
    return out


def rolling_slope(y: np.ndarray, w: int = 60) -> np.ndarray:
    """y[t-w+1 : t+1] 단순선형회귀 기울기."""
    y = np.asarray(y, dtype=float)
    n = y.size
    out = np.full(n, np.nan)
    xgrid = np.arange(w, dtype=float)
    xmean = xgrid.mean()
    xvar = ((xgrid - xmean) ** 2).sum()
    for t in range(w - 1, n):
        win = y[t - w + 1 : t + 1]
        if np.isnan(win).any():
            continue
        ymean = win.mean()
        out[t] = ((xgrid - xmean) * (win - ymean)).sum() / xvar
    return out
```

## 2.5 `packages/epd_core/src/epd_core/epd.py` ★ 핵심 변경

```python
"""EPD (Entropy-Price Divergence) — RSI-style bounded indicator."""
from __future__ import annotations
import numpy as np
from ._core import rolling_pe, rolling_zscore

# 이론적 고정 상수 (변경 금지)
_DEFAULT_W_PE = 60
_DEFAULT_M = 3
_DEFAULT_TAU = 1
_DEFAULT_W_Z = 252
_DEFAULT_TANH_SCALE = 2.0


def compute_epd(returns: np.ndarray,
                close: np.ndarray | None = None,
                w_pe: int = _DEFAULT_W_PE,
                m: int = _DEFAULT_M,
                tau: int = _DEFAULT_TAU,
                w_z: int = _DEFAULT_W_Z,
                bounded: bool = True,
                tanh_scale: float = _DEFAULT_TANH_SCALE) -> dict:
    """
    EPD (Entropy-Price Divergence).

    원시 정의:
        EPD_raw = Z_r - Z_e

    RSI식 유계 변환 (bounded=True, 기본):
        EPD     = tanh(EPD_raw / tanh_scale)              ∈ (-1, +1)
        EPD_100 = 50 * (1 + tanh(EPD_raw / tanh_scale))   ∈ (0, 100)

    임계값:
        EPD_100 > 70 → 과열 위험
        EPD_100 < 30 → 구조 형성 중

    Parameters
    ----------
    returns     : 1D 로그수익률
    close       : 미사용 (V1 return 고정)
    w_pe, m, tau, w_z : §3.4 고정값. 변경은 연구 목적 외 금지.
    bounded     : True면 EPD_100 스케일 사용. False면 원시 z-score.
    tanh_scale  : 변환 상수 (기본 2.0). 튜닝 대상 아님.

    Returns
    -------
    dict: epd, epd_100, epd_raw, epd_abs, alignment,
          return_z, entropy_z, pe, dpe, params
    """
    r = np.asarray(returns, dtype=float)
    n = r.size

    pe = rolling_pe(r, w_pe=w_pe, m=m, tau=tau)

    dpe = np.full(n, np.nan)
    if n >= 2:
        dpe[1:] = pe[1:] - pe[:-1]

    z_r = rolling_zscore(r, w_z=w_z)
    z_e = rolling_zscore(dpe, w_z=w_z)

    epd_raw = z_r - z_e

    if bounded:
        epd = np.tanh(epd_raw / tanh_scale)
        epd_100 = 50.0 * (1.0 + epd)
    else:
        epd = epd_raw
        epd_100 = epd_raw

    epd_abs = np.abs(epd)

    with np.errstate(invalid="ignore"):
        prod = z_r * z_e
    alignment = np.sign(prod)
    alignment[np.isnan(prod)] = np.nan

    return {
        "epd": epd,
        "epd_100": epd_100,
        "epd_raw": epd_raw,
        "epd_abs": epd_abs,
        "alignment": alignment,
        "return_z": z_r,
        "entropy_z": z_e,
        "pe": pe,
        "dpe": dpe,
        "params": {
            "w_pe": w_pe, "m": m, "tau": tau, "w_z": w_z,
            "bounded": bounded, "tanh_scale": tanh_scale,
        },
    }
```

## 2.6 `packages/epd_core/src/epd_core/__init__.py`

```python
"""EPD Core — Entropy-Price Divergence, RSI-style bounded indicator."""
from .epd import compute_epd
from .permutation import permutation_entropy
from ._core import rolling_pe, rolling_zscore, rolling_slope

__all__ = [
    "compute_epd",
    "permutation_entropy",
    "rolling_pe",
    "rolling_zscore",
    "rolling_slope",
]

__version__ = "0.1.0"

# 이론적 고정 상수 재노출 (참조용)
DEFAULT_W_PE = 60
DEFAULT_M = 3
DEFAULT_TAU = 1
DEFAULT_W_Z = 252
DEFAULT_TANH_SCALE = 2.0
THRESHOLD_HIGH = 70
THRESHOLD_LOW = 30
```

## 2.7 `packages/epd_core/tests/test_epd.py` (갱신)

```python
"""EPD 구현 검증 — RSI 철학 반영."""
import numpy as np
import pytest
from epd_core import (
    compute_epd, permutation_entropy, rolling_pe, rolling_zscore,
    DEFAULT_W_PE, DEFAULT_M, DEFAULT_TAU, DEFAULT_W_Z,
    THRESHOLD_HIGH, THRESHOLD_LOW,
)


def test_pe_constant_is_zero():
    assert permutation_entropy(np.ones(50), m=3, tau=1) == pytest.approx(0.0)


def test_pe_monotonic_is_zero():
    assert permutation_entropy(np.arange(100, dtype=float), m=3, tau=1) == pytest.approx(0.0)


def test_pe_bounded():
    rng = np.random.default_rng(0)
    for _ in range(5):
        h = permutation_entropy(rng.standard_normal(200), m=3, tau=1)
        assert 0.0 <= h <= 1.0


def test_rolling_pe_causal():
    rng = np.random.default_rng(1)
    pe = rolling_pe(rng.standard_normal(300), w_pe=50)
    assert np.all(np.isnan(pe[:50]))
    assert np.isfinite(pe[50:]).all()


def test_rolling_zscore_causal():
    x = np.arange(1.0, 101.0)
    z = rolling_zscore(x, w_z=20)
    assert np.all(np.isnan(z[:20]))
    assert z[20] > 1.0
    z_short = rolling_zscore(x[:60], w_z=20)
    assert np.allclose(z[:60], z_short, equal_nan=True)


def test_epd_100_bounded_range():
    """★ RSI 철학 P1: 출력은 반드시 (0, 100)."""
    rng = np.random.default_rng(42)
    r = rng.standard_normal(3000)
    out = compute_epd(r)
    valid = np.isfinite(out["epd_100"])
    assert valid.sum() > 0
    assert (out["epd_100"][valid] > 0).all()
    assert (out["epd_100"][valid] < 100).all()


def test_epd_bounded_in_neg1_pos1():
    rng = np.random.default_rng(43)
    r = rng.standard_normal(2000)
    out = compute_epd(r)
    valid = np.isfinite(out["epd"])
    assert (out["epd"][valid] > -1).all()
    assert (out["epd"][valid] < 1).all()


def test_epd_100_neutral_at_zero_raw():
    """EPD_raw=0 → EPD_100=50 (중립)."""
    r = np.zeros(1000)  # 상수 수익률 → 모든 z=0 (혹은 NaN)
    out = compute_epd(r, w_pe=30, w_z=100)
    valid = np.isfinite(out["epd_100"])
    if valid.any():
        assert np.allclose(out["epd_100"][valid], 50.0)


def test_epd_default_params():
    """★ RSI 철학 P2: 기본 파라미터가 이론적 고정값."""
    rng = np.random.default_rng(44)
    out = compute_epd(rng.standard_normal(1000))
    p = out["params"]
    assert p["w_pe"] == DEFAULT_W_PE == 60
    assert p["m"] == DEFAULT_M == 3
    assert p["tau"] == DEFAULT_TAU == 1
    assert p["w_z"] == DEFAULT_W_Z == 252
    assert p["bounded"] is True
    assert p["tanh_scale"] == 2.0


def test_epd_thresholds_are_absolute():
    """★ 임계값이 상수로 고정되어 있어야 함."""
    assert THRESHOLD_HIGH == 70
    assert THRESHOLD_LOW == 30


def test_epd_100_matches_formula():
    """EPD_100 = 50*(1 + tanh(EPD_raw/2))."""
    rng = np.random.default_rng(45)
    r = rng.standard_normal(1500)
    out = compute_epd(r)
    expected = 50.0 * (1.0 + np.tanh(out["epd_raw"] / 2.0))
    mask = np.isfinite(expected)
    assert np.allclose(out["epd_100"][mask], expected[mask])


def test_epd_unbounded_option():
    """bounded=False면 raw 값 반환."""
    rng = np.random.default_rng(46)
    out = compute_epd(rng.standard_normal(500), bounded=False)
    mask = np.isfinite(out["epd"])
    assert np.allclose(out["epd"][mask], out["epd_raw"][mask])
    # raw는 1을 넘을 수 있음
    assert (np.abs(out["epd"][mask]) > 1).any()


def test_epd_bounded_never_exceeds_100():
    """극단적 스파이크에도 EPD_100 < 100."""
    r = np.zeros(1000)
    r[500] = 50.0  # 극단 스파이크
    out = compute_epd(r, w_pe=30, w_z=100)
    valid = np.isfinite(out["epd_100"])
    assert (out["epd_100"][valid] < 100).all()
    assert (out["epd_100"][valid] > 0).all()
```

## 2.8 `scripts/stage1_synthetic.py` 갱신 (EPD_100 사용)

```python
"""
Stage 1: Synthetic Validation — RSI-style bounded EPD.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))

from epd_core import compute_epd  # noqa: E402

RNG = np.random.default_rng(20260930)
N = 10_000


def gen_white_noise(n=N):
    return RNG.standard_normal(n)


def gen_garch(n=N, alpha=0.10, beta=0.85, seed=7):
    rng = np.random.default_rng(seed)
    omega = 1.0 - alpha - beta
    r = np.zeros(n)
    s2 = np.zeros(n)
    s2[0] = omega / max(1e-9, (1 - alpha - beta))
    eps = rng.standard_normal(n)
    for t in range(1, n):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * eps[t]
    return r


def gen_regime(n=N, p_stay=0.98, seed=11):
    rng = np.random.default_rng(seed)
    state = 0
    states = np.zeros(n, dtype=int)
    for t in range(n):
        if rng.random() > p_stay:
            state = 1 - state
        states[t] = state
    sig = np.where(states == 0, 1.0, 3.0)
    return rng.standard_normal(n) * sig, states


def gen_divergence(n=N, seed=13):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n) * 0.5
    phi = np.linspace(0.0, 0.85, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.05 + phi[t] * r[t - 1] + eps[t]
    return r


def forward_return(r, h):
    fr = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fr[t] = r[t + 1 : t + 1 + h].sum()
    return fr


def forward_vol(r, h):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fv[t] = r[t + 1 : t + 1 + h].std(ddof=1)
    return fv


def spearman_ic(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 100:
        return np.nan
    xr = np.argsort(np.argsort(x[mask]))
    yr = np.argsort(np.argsort(y[mask]))
    return float(np.corrcoef(xr, yr)[0, 1])


def ic_blocks(x, y, block=1000):
    ics = []
    for i in range(0, len(x), block):
        ics.append(spearman_ic(x[i:i+block], y[i:i+block]))
    return np.array([v for v in ics if np.isfinite(v)])


def test_A_white_noise():
    r = gen_white_noise()
    out = compute_epd(r)
    epd = out["epd_100"]
    fr = forward_return(r, 5)
    fv = forward_vol(r, 5)

    ic_ret = spearman_ic(epd, fr)
    ic_vol = spearman_ic(epd, fv)
    blk = ic_blocks(epd, fr, block=1000)
    frac_fp = float(np.mean(np.abs(blk) > 0.05)) if blk.size else np.nan

    return {"name": "A_white_noise",
            "ic_ret_5d": ic_ret, "ic_vol_5d": ic_vol,
            "frac_blocks_absIC>0.05": frac_fp,
            "pass": bool(frac_fp < 0.05)}


def test_B_garch():
    r = gen_garch()
    out = compute_epd(r)
    epd = out["epd_100"]
    ic_ret = spearman_ic(epd, forward_return(r, 5))
    ic_vol = spearman_ic(epd, forward_vol(r, 5))
    return {"name": "B_garch",
            "ic_ret_5d": ic_ret, "ic_vol_5d": ic_vol,
            "pass": bool(abs(ic_ret) < 0.05)}


def test_C_regime():
    r, states = gen_regime()
    out = compute_epd(r)
    epd100 = out["epd_100"]
    trans = np.where(np.diff(states) != 0)[0] + 1
    hits = 0
    for t in trans:
        lo, hi = max(0, t - 5), min(len(epd100), t + 5)
        if np.any(np.abs(epd100[lo:hi] - 50) > 20):
            hits += 1
    frac = hits / max(1, len(trans))
    return {"name": "C_regime",
            "n_transitions": int(len(trans)),
            "hit_rate_|epd100-50|>20_within_5d": frac,
            "pass": bool(frac >= 0.5)}


def test_D_divergence():
    r = gen_divergence()
    out = compute_epd(r)
    epd100 = out["epd_100"]
    half = len(epd100) // 2
    first = epd100[:half][np.isfinite(epd100[:half])]
    second = epd100[half:][np.isfinite(epd100[half:])]
    delta = second.mean() - first.mean()
    sd = np.nanstd(epd100)
    return {"name": "D_divergence",
            "epd100_mean_first_half": float(first.mean()),
            "epd100_mean_second_half": float(second.mean()),
            "delta_over_sigma": float(delta / sd) if sd > 0 else np.nan,
            "pass": bool(abs(delta) > 0.3 * sd)}


def main():
    results = []
    for fn in (test_A_white_noise, test_B_garch, test_C_regime, test_D_divergence):
        res = fn()
        results.append(res)
        flag = "PASS" if res["pass"] else "FAIL"
        print(f"[Stage1:{res['name']}] {flag}  "
              f"{ {k:v for k,v in res.items() if k not in ('name','pass')} }")

    passed = all(r["pass"] for r in results)
    print(f"\n=== Stage 1 Result: {'PASS' if passed else 'FAIL'} ===")

    rpt = ROOT / "docs" / "synthetic_report.md"
    lines = ["# Synthetic Report — Stage 1 (v1.1, RSI-style)",
             "", "Fixed params: w_pe=60, m=3, tau=1, w_z=252, tanh_scale=2",
             ""]
    for res in results:
        lines.append(f"## {res['name']}")
        for k, v in res.items():
            if k == "name":
                continue
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append(f"**Overall**: {'PASS' if passed else 'FAIL'}")
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {rpt}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

---

# Part 3. 실행

```bash
cd epd_project

# 1) 패키지 설치 (editable)
pip install -e packages/epd_core

# 2) 단위 테스트 (특히 test_epd_100_bounded_range, test_epd_thresholds_are_absolute)
python -m pytest packages/epd_core/tests/ -v

# 3) Stage 1 합성 검증
python scripts/stage1_synthetic.py
```

---

# Part 4. 이번 세션에서 확정된 것

| 항목 | v1.0 | **v1.1 (현재)** |
|---|---|---|
| 출력 | 무한대 (z-score) | **유계 (0, 100)** |
| 임계값 | ±1.5 / ±2 (자산별 조정 여지) | **70 / 30 (절대값)** |
| 파라미터 | 36조합 탐색 후 중심값 | **이론적 고정** |
| Stage 3 | Robustness (중심값 선택) | **Universality (≥90% 동일)** |
| H3 | 회귀 계수 유의성 | **직교성 (corr < 0.5)** |
| ML 피처 | §8에 계획 | **삭제 (철학 위반)** |
| 성공 확률 | 60~70% | **40~50%** |

**다음 결정 포인트**:
1. `w_pe=60` 근거가 충분한가? (20=월, 252=년 중 재검토 여지)
2. `tanh_scale=2` 근거가 충분한가? (1이 더 RSI스러울 수도)
3. Stage 1 결과 → 통과 시 `scripts/stage3_universality.py` 작성

원하시면 이어서 **Stage 3 보편성 스크립트**와 **Stage 4 다자산 로더**까지 바로 깔아드립니다. 어디까지 갈까요?