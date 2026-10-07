# EPD_SPEC.md v3.1

**확정일**: 2026-10-06
**상태**: FROZEN
**버전**: 3.1.0

---

## 1. 정의
r_t = log(P_t / P_{t−1})
PE_t = permutation_entropy(r[t−60:t], m=3, tau=1)
Z_r(t) = causal_zscore(r, 252)
E[PE|Z_r] = rolling_bin_mean(PE | |Z_r|, 504)
Z_e(t) = causal_zscore(PE_t − E[PE_t|Z_r], 252)

EPD_mag = ||Z_r| − |Z_e||
EPD_100 = 100 · tanh(EPD_mag / 2) ∈ (0, 100)

text

### 의미
> **EPD는 가격 스트레스 크기(|Z_r|)와 가격으로 설명되지 않는 엔트로피 이탈 크기(|Z_e|) 사이의 magnitude divergence를 측정하는 시장 상태 변수다.**

**Z_e의 부호는 정보 없음** (Stage 9 확정). 따라서 magnitude만 사용.

---

## 2. 파라미터 (고정)

| 이름 | 값 |
|---|---|
| `w_pe` | 60 |
| `m` | 3 |
| `tau` | 1 |
| `w_z` | 252 |
| `resid_win` | 504 |
| `tanh_scale` | 2.0 |

---

## 3. 임계값 (N1 tanh 기반)

| EPD_100 | 상태 | Forward Vol |
|---|---|---|
| < 30 | 낮은 긴장 | baseline |
| 30 ~ 70 | 중간 | +0.5% |
| **≥ 70** | **높은 긴장** | **+30% (mean ratio 1.31)** |

**자산별 High/Low ratio**:
- SPX: 1.59, NASDAQ: 1.60 (강함)
- KOSDAQ: 1.45, KOSPI: 1.22
- KR 개별주: 1.13~1.19

---

## 4. 출력 스키마

```python
{
    "epd_100":   np.ndarray,   # 0-100 (주 사용)
    "epd_mag":   np.ndarray,   # ||Z_r| - |Z_e||
    "epd_dir":   np.ndarray,   # sign 진단용
    "epd_raw":   np.ndarray,   # Z_r - Z_e (legacy)
    "z_r":       np.ndarray,
    "z_e":       np.ndarray,
    "price_z":   np.ndarray,   # |Z_r|
    "entropy_z": np.ndarray,   # |Z_e|
    "pe":        np.ndarray,
    "resid":     np.ndarray,
    "params":    dict,
}
```
5. 검증 이력
Stage	검증	결과
1	합성 (4 tests)	PASS
2	시장 구조 (12자산)	조건부 PASS
3	Distinctiveness (7 지표)	12/12 PASS
4	State (4분면)	FAIL → E4 발견
4.5	Z_e variants	E4 residual 확정
5	Robustness	C1/C4
6	Representation	R4 발견
7	R1~R6 비교	R6 최우수
8	정보 검증	R6 확인
9	Ze 부호	부호 무정보
10	Time stability	82.3% 양수 PASS
11	Normalization	N1 (tanh) 채택
6. 자산별 성능
강한 자산 (SPX, NASDAQ, KOSDAQ)
R6 partial > 0.05

High/Low forward vol ratio > 1.4

약한 자산 (SK하이닉스, 일부 KR 개별주)
R6 partial 0.01~0.03

2020-2025 약화 (75% 양수)

예외 (명시)
SPX 2005-2010: 3/3 음수 (GFC, R1 압도)

SK하이닉스 2020-2025: 5/6 음수

7. 한계 (정직)
Z_e 부호 무정보: sign component 폐기

개별주 편차: KR 개별주 약함

GFC/최근 regime: 특정 구간 약화

방향 예측 불가: 구조적

"보편적 안정성" 아님: "시간적 안정성 근거 확보"

8. 사용법
시각화
EPD_100 라인 (0-100)

임계선 30/70

배경 히트맵: EPD_100 강도

실전
변동성 예측 피처

Regime 신호 (≥70 = 높은 긴장)

❌ 방향 예측

❌ 단독 전략

기존 지표와 관계
RSI와 corr ~0.23

|Z_r| 통제 후 partial +0.05 (Stage 10)

9. 최종 표현
EPD = ||Z_r| − |Z_e||, scaled by tanh

RSI가 "가격 모멘텀 상태"를 표시하듯,
EPD는 "가격-엔트로피 magnitude divergence 상태"를 표시한다.

이 문서는 EPD v3.1의 최종 스펙이며, 이후 변경은 v4.0으로 새로 작성한다.
