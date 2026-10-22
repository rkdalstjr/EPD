# EPD_SPEC.md v3.1.0

**확정일**: 2026-10-06
**상태**: **FROZEN**
**버전**: 3.1.0

---

## 1. 정의
```
r_t = log(P_t / P_{t−1})
PE_t = permutation_entropy(r[t−60:t], m=3, tau=1)

Z_r(t) = causal_zscore(r_t, 252)
E[PE|Z_r] = rolling_bin_mean(PE | |Z_r|, 504) # 5분위 bin
Z_e(t) = causal_zscore(PE_t − E[PE_t|Z_r], 252)

EPD_mag = ||Z_r| − |Z_e||
EPD_100 = 100 · tanh(EPD_mag / 2) ∈ (0, 100)
```


### 의미

> **EPD는 가격 스트레스 크기(|Z_r|)와, 가격으로 설명되지 않는
>  엔트로피 이탈 크기(|Z_e|) 사이의 magnitude divergence를
>  측정하는 시장 상태 변수다.**

- **Z_r의 부호**는 magnitude에 사용 안 함 (Stage 9 확정)
- **Z_e의 부호**는 magnitude에 사용 안 함 (Stage 9 확정)
- **magnitude only 지표**

---

## 2. 파라미터 (동결)

| 이름 | 값 | 근거 |
|---|---|---|
| `w_pe` | 60 | Bandt-Pompe 표준 |
| `m` | 3 | 원논문 (m=4는 통계 신뢰성 부족) |
| `tau` | 1 | 일봉 표준 |
| `w_z` | 252 | 1년 거래일 |
| `resid_win` | 504 | 2년 (회귀 기반 기대값) |
| `tanh_scale` | 2.0 | RSI 70/30 정렬 |
| `n_bins` | 5 | \|Z_r\| 분위 |

**변경 금지**. 튜닝 대상 아님.

---

## 3. 임계값

| EPD_100 | 상태 | Forward Vol |
|---|---|---|
| < 30 | 낮은 긴장 | baseline |
| 30 ~ 70 | 중간 | +0.5% |
| **≥ 70** | **높은 긴장** | **+30% (평균)** |

**자산별 High/Low forward vol ratio** (Stage 11):
- SPX: 1.59, NASDAQ: 1.60
- KOSDAQ: 1.45, KOSPI: 1.22
- KR 개별주: 1.13~1.19

---

## 4. 출력 스키마

```python
{
    "epd_100":   np.ndarray,   # 0-100 (주 사용)
    "epd_mag":   np.ndarray,   # ||Z_r| - |Z_e||
    "epd_dir":   np.ndarray,   # sign (진단용)
    "z_r":       np.ndarray,   # = Z_r
    "z_e":       np.ndarray,   # = Z_e
    "price_z":   np.ndarray,   # |Z_r|
    "entropy_z": np.ndarray,   # |Z_e|
    "pe":        np.ndarray,   # 순열 엔트로피
    "resid":     np.ndarray,   # PE - E[PE|Zr]
    "params":    dict,
}
```

## 5. 검증 이력

| **Stage** | **검증** | **결과** |
| :--- | :--- | :--- |
| 1 | 합성 (White, GARCH, Regime, Divergence) | ✅ 4/4 PASS |
| 2 | 시장 구조 (12자산) | ⚠️ 조건부 |
| 3 | Distinctiveness (7 지표) | ✅ 12/12 PASS |
| 4 | State (4분면) | ❌ |
| 4.5 | Z_e variants | ✅ E4 확정 |
| 5 | Robustness | ⚠️ C1 우세 |
| 6 | Representation | ✅ R4 발견 |
| 7 | R1~R6 비교 | ✅ R6 |
| 8 | 정보 검증 | ✅ R6 확인 |
| 9 | Ze 부호 | ✅ 무정보 |
| 10 | Time stability | ✅ 82.3% |
| 11 | Normalization | ✅ N1 (tanh) |
| 대안 1 | Weighted PE | ❌ 무효 |
| 대안 2 | Dispersion Entropy | ❌ 무효 |
| 대안 3 | 3일 RMS Z_r | ❌ RMS 프록시 |
| 대안 4 | Regime-conditional | ❌ 개선 없음 |

---

## 6. 자산별 성능

### 강함

| **자산** | **R6 partial (Stage 10)** | **Q5/Q1 (Stage 2)** |
| :--- | :--- | :--- |
| SPX | +0.098 | 1.343 |
| NASDAQ | +0.125 | 1.312 |
| KOSDAQ | +0.147 | 1.271 |

### 약함

| **자산** | **R6 partial** | **문제** |
| :--- | :--- | :--- |
| SK_Hynix | -0.040 | Ze 역효과 |
| Samsung | +0.006 | 2020-2025 약화 |
| KOSPI | +0.034 | 상대적 약함 |

---

## 7. 한계

1. **방향 예측 불가**: 엔트로피는 무질서도 (구조적)
2. **자산 의존**: KR 개별주 편차 큼
3. **고변동 regime 편향**: 저변동에서 무효 (A, D 공통)
4. **"보편적 안정성" 아님**: "시간적 안정성" 수준
5. **단독 리스크 필터 부적합**: 변동성 ≠ 수익 방향
6. **엔트로피 대안 무효**: Weighted PE, Dispersion Entropy 모두 나쁨

---

## 8. 사용법

### Python

```python
from epd_core import compute_epd, THRESHOLD_HIGH, THRESHOLD_LOW

out = compute_epd(close)

epd = out["epd_100"]

high_tension = epd > THRESHOLD_HIGH  # > 70
low_tension = epd < THRESHOLD_LOW    # < 30
```

### 시각화

- **EPD_100 라인** (0-100) + 30/70 밴드
- **|Zr| vs |Ze|** 컴포넌트 (선택)
- **나비형 UI**: Z_r과 -Z_e 비교

### Pine Script

- `docs/EPD_pine_v6.txt` 참조

---

## 9. 대안 검토 결과 (참고, 2026-10)

| **대안** | **결과** | **이유** |
| :--- | :--- | :--- |
| **Weighted PE** | 기각 | 진폭 가중치가 노이즈 증폭 |
| **Dispersion Entropy** | 기각 | A와 동등, 새 정보 없음 |
| **3일 RMS Z_r** | 기각 | raw corr 우위는 RMS 프록시 |
| **Regime-conditional** | 기각 | A 대비 개선 없음 |

**모두 v3.1을 대체할 근거 부족**.

---

## 10. 다음 단계

- **TRA 프로젝트** (별도)
- **KSE 프로젝트** (별도)
- **3차원 상태공간**: EPD × TRA × KSE
- **고변동 regime 지표 재정의** (선택, 별도 연구)

---

## 11. 동결 선언

> **이 문서는 EPD v3.1.0의 최종 스펙이며,**
>
> **이후 변경은 v4.0으로 새로 작성한다.**

**변경 금지 항목**:

- 정의 (수식)
- 파라미터 (60/3/1/252/504/2.0)
- 임계값 (30/70)
- 출력 스키마

**확정일**: 2026-10-06  
**상태**: FROZEN  
**다음 검토**: 6개월 후 (TRA/KSE 통합 시)

---

## 12. 코드 최종 정리

### `packages/epd_core/src/epd_core/__init__.py`

```python
"""EPD Core v3.1.0 — Final frozen."""

from .epd import (
    compute_epd,
    DEFAULT_W_PE,
    DEFAULT_M,
    DEFAULT_TAU,
    DEFAULT_W_Z,
    DEFAULT_RESID_WIN,
    DEFAULT_TANH_SCALE,
    DEFAULT_Z_R_MODE,
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
)

from .permutation import permutation_entropy
from ._core import rolling_pe, rolling_zscore, rolling_slope

__all__ = [
    "compute_epd",
    "permutation_entropy",
    "rolling_pe",
    "rolling_zscore",
    "rolling_slope",
    "DEFAULT_W_PE",
    "DEFAULT_M",
    "DEFAULT_TAU",
    "DEFAULT_W_Z",
    "DEFAULT_RESID_WIN",
    "DEFAULT_TANH_SCALE",
    "DEFAULT_Z_R_MODE",
    "THRESHOLD_HIGH",
    "THRESHOLD_LOW",
]

__version__ = "3.1.0"
```