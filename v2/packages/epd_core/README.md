# EPD — Entropy-Price Divergence

> **가격 움직임과 순열 엔트로피 구조의 divergence를 측정하는 시장 상태 지표**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![numpy](https://img.shields.io/badge/numpy-1.24%2B-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 개요

**EPD (Entropy-Price Divergence)** 는 가격 스트레스의 크기와, 가격으로 설명되지 않는 엔트로피 이탈의 크기 사이의 **magnitude divergence**를 측정하는 신규 시장 상태 지표입니다.

```
EPD_100 = 100 · tanh( ||Z_r| − |Z_e|| / 2 )     ∈ (0, 100)
```

**한 줄 정의**:
> "가격 충격의 크기와, 그 가격 충격으로 설명되지 않는 시장 구조의 이탈 크기 사이의 불일치"

기존 기술적 지표(RSI, MACD, ATR 등)와 **통계적으로 독립**이며, **미래 변동성에 대한 추가 정보**를 제공합니다.

---

## ⚠️ 주의사항

EPD는 **방향 예측 지표가 아닙니다**.

- ✅ **시장 상태 변수** (구조적 긴장 강도)
- ✅ **미래 변동성 예측 피처**
- ✅ **Regime 라벨러**
- ❌ **방향 예측** (엔트로피는 무질서도이므로 방향성 없음)
- ❌ **매매 신호** (전략 성과는 별도 검증)
- ❌ **RSI 대체** (다른 카테고리)

**RSI가 "어디로 갈까"라면, EPD는 "얼마나 흔들릴까".**

---

## 왜 필요한가

### 기존 지표 체계의 공백

| 축 | 커버 지표 |
|---|---|
| 방향 | RSI, MACD |
| 위치 | Bollinger %B |
| 크기 | ATR |
| 기대 변동 | VIX |
| **구조-가격 괴리** | **(공백)** |

> "같은 +2% 상승이라도, 시장 구조가 다류면 다른 사건이다."

가격만 보는 지표로는 **"가격은 튀는데 구조는 잠잠한 상태"** 와 **"가격과 구조가 함께 움직이는 상태"** 를 구분할 수 없습니다.

EPD는 이 **5번째 축**을 채웁니다.

---

## 설치

```bash
pip install epd-core

# 또는 개발 모드
pip install -e packages/epd_core
```

의존성: `numpy >= 1.24` (그 외 없음)

---

## 사용법

```python
import numpy as np
from epd_core import compute_epd, THRESHOLD_HIGH, THRESHOLD_LOW

# 종가 시계열 (양수)
close = np.array([...])

# EPD 계산
out = compute_epd(close)

epd = out["epd_100"]     # 0~100, 주 사용
mag = out["epd_mag"]     # raw magnitude
z_r = out["z_r"]         # 가격 z-score
z_e = out["z_e"]         # 엔트로피 residual z-score

# 임계값 기반 해석 (자산 무관)
low_tension  = epd < THRESHOLD_LOW    # < 30
high_tension = epd > THRESHOLD_HIGH   # > 70 → 변동성↑ 예고
```

### 시각화 (선택)

```bash
pip install epd-core[viz]   # plotly, yfinance
```

```python
# 나비형 다이어그램 (Z_r vs Z_e)
from scripts.epd_ui import make_chart
```

---

## 정의 (v3.1)

### 수식

```
r_t        = log(P_t / P_{t−1})
PE_t       = permutation_entropy(r[t−60:t], m=3, tau=1)

Z_r(t)     = causal_zscore(r, 252)
E[PE|Z_r]  = rolling_bin_mean(PE | |Z_r|, 504)
Z_e(t)     = causal_zscore(PE_t − E[PE_t|Z_r], 252)

EPD_mag    = ||Z_r| − |Z_e||
EPD_100    = 100 · tanh(EPD_mag / 2)
```

### 파라미터 (고정)

| 파라미터 | 값 | 근거 |
|---|---|---|
| `w_pe` | 60 | Bandt-Pompe 표준 (m=3에서 6패턴 × 10관측) |
| `m` | 3 | Bandt-Pompe 2002 |
| `tau` | 1 | 일봉 표준 |
| `w_z` | 252 | 1년 거래일 |
| `resid_win` | 504 | 2년 (회귀 기반 기대값) |
| `tanh_scale` | 2.0 | RSI 70/30 정렬 |

### 임계값

| EPD_100 | 상태 | Forward Vol |
|---|---|---|
| < 30 | 낮은 긴장 | baseline |
| 30 ~ 70 | 중간 | +0.5% |
| **≥ 70** | **높은 긴장** | **+30%** |

---

## 검증 결과

### 독립성 (Stage 3)

| 비교 지표 | corr(EPD, X) |
|---|---|
| RSI(14) | −0.23 |
| ATR(14) | +0.15 |
| MACD | +0.15 |
| Realized Vol(20) | +0.10 |
| Autocorr(20) | ~0.00 |
| Permutation Entropy | −0.21 |

**모든 기존 지표와 corr < 0.6** → EPD는 독립적인 새 축.

### 미래 변동성 예측력 (Stage 7, 10)

| Horizon | Partial corr (|Z_r| 통제 후) |
|---|---|
| 5일 | +0.089 |
| 10일 | +0.084 |
| 20일 | +0.086 |

**|Z_r| 통제 후에도 유의한 추가 정보 제공**.

### 시간 안정성 (Stage 10)

**4기간 × 8자산 × 3 horizon = 96 케이스**:
- 양수 비율: **82.3%**
- 4개 기간 모두 평균 partial > 0

### 자산별 High/Low Forward Vol Ratio (Stage 11)

| 자산 | Ratio |
|---|---|
| SPX | **1.59** |
| NASDAQ | **1.60** |
| KOSDAQ | **1.45** |
| KOSPI | 1.22 |
| 한국 개별주 | 1.13~1.19 |

**EPD_100 ≥ 70 구간에서 미래 변동성 평균 30% 이상 증가**.

---

## 자산별 성능

| 자산군 | 판정 |
|---|---|
| 미국 대형 지수 (SPX, NASDAQ) | ✅ 강함 |
| 한국 중소형 지수 (KOSDAQ) | ✅ 강함 |
| 한국 대형 지수 (KOSPI) | ⚠️ 중간 |
| 한국 개별주 | ⚠️ 약함 (편차 큼) |

---

## 한계 (정직)

1. **방향 예측 불가** — 구조적 (엔트로피는 무질서도)
2. **개별주 편차** — SK하이닉스 등 일부 자산 약함
3. **GFC 구간 예외** — SPX 2005-2010에서 음의 관계
4. **"보편적 안정성" 아님** — "시간적 안정성 근거 확보" 수준
5. **단독 리스크 필터 부적합** — 변동성 ≠ 수익 방향

---

## 검증 이력 (11 Stages)

| Stage | 검증 | 결과 |
|---|---|---|
| 1 | 합성 데이터 (4 tests) | ✅ PASS |
| 2 | 시장 구조 (12자산) | ⚠️ 조걸부 |
| 3 | Distinctiveness (7 지표) | ✅ 12/12 PASS |
| 4 | 4분면 상태 분석 | ❌ 재설계 유도 |
| 5 | Robustness | ⚠️ C1/C4 |
| 6 | Representation | ✅ R4 발견 |
| 7 | R1~R6 비교 | ✅ R6 최우수 |
| 8 | 정보 검증 | ✅ R6 확인 |
| 9 | 방향성 (Ze 부호) | ✅ 부호 폐기 |
| 10 | 시간 안정성 | ✅ 82.3% PASS |
| 11 | Normalization | ✅ N1 (tanh) 채택 |

---

## 프로젝트 구조

```
epd_project/
├── packages/
│   └── epd_core/                 # ★ 배포 패키지
│       ├── pyproject.toml
│       ├── README.md
│       ├── src/
│       │   └── epd_core/
│       │       ├── __init__.py
│       │       ├── permutation.py   # 순열 엔트로피
│       │       ├── _core.py         # 롤링 유틸
│       │       └── epd.py           # ★ compute_epd
│       └── tests/
│           └── test_epd.py
├── scripts/                      # 연구 스크립트
│   ├── stage_1_synthetic.py
│   ├── stage_2_market_structure.py
│   ├── ...
│   ├── stage_11_normalization.py
│   └── epd_ui.py                # 시각화
├── docs/
│   ├── EPD_SPEC.md              # 지표 스펙 v3.1
│   ├── FINAL_DECISION.md        # 최종 판정
│   └── charts/                  # 시각화 결과
└── README.md
```

---

## 시각화

EPD는 **나비형 다이어그램**으로 시각화됩니다:

```
┌─────────────────────────────┐
│  가격 (캔들스틱)             │  ← 배경: 긴장도 히트맵
├─────────────────────────────┤
│  Z_r ↗  |  ↖ -Z_e            │  ← 나비형: divergence 시각화
│      0 ─┼─                   │
├─────────────────────────────┤
│  EPD_100 (라인, 30/50/70)    │  ← magnitude
└─────────────────────────────┘
```

```bash
python scripts/epd_ui.py --ticker 005930.KS --start 2020-01-01
```

---

## 다른 지표와의 관계

| | RSI | EPD |
|---|---|---|
| 재는 것 | 가격 모멘텀 | 구조적 긴장 |
| 출력 | 0~100 | 0~100 |
| 임계값 | 70/30 | 70/30 |
| 방향 예측 | 부분적 | ❌ |
| 변동성 예측 | ❌ | ✅ |
| 자산 보편성 | 높음 | 중간 |

**EPD는 RSI의 대체가 아니라 보완.**

**방향(RSI) + 구조(EPD)의 조합**이 시장에 대한 더 완전한 그림을 제공합니다.

---

## 활용 경로

| 용도 | 방법 |
|---|---|
| **시각화** | 상태 대시보드 |
| **변동성 예측 피처** | GARCH / ML 입력 |
| **Regime 라벨러** | "안정 vs 불안정" 구간 표시 |
| **리스크 경보** | EPD_100 ≥ 70 → 포지션 조정 |
| **3차원 상태공간** | TRA, KSE와 결합 |
| ❌ 방향 신호 | 사용 금지 |
| ❌ 단독 전략 | 검증 안 됨 |

---

## 다음 단계

- **TRA** (Trend-Range-Ambiguity) 프로젝트 — 별도
- **KSE** (Kurtosis-Skew-Entropy) 프로젝트 — 별도
- **3차원 상태공간**: EPD × TRA × KSE

---

## 인용

```
@software{epd_core,
  title  = {EPD: Entropy-Price Divergence},
  author = {EPD Research},
  year   = {2026},
  version = {3.1.0},
  note   = {Magnitude divergence market-state indicator}
}
```

---

## 라이선스

MIT

---

## 참고문헌

- Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time series. *Physical Review Letters*, 88(17), 174102.
- 관련 문서: [`docs/EPD_SPEC.md`](docs/EPD_SPEC.md), [`docs/FINAL_DECISION.md`](docs/FINAL_DECISION.md)
