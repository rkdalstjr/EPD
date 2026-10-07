# STAGE 0 — Research Spec (동결)

**프로젝트**: EPD (Entropy-Price Divergence) v2.0
**동결일**: 2026-10-06
**원칙**: 본 문서는 실험 전 확정. 수정 시 `STAGE_0_SPEC_v2.md` 새로 생성.

---

## 1. 연구 목표

> **"가격 움직임과 순열 엔트로피 변화의 어긋남(divergence)이
> 미래 시장 특성에 대해 기존 지표가 설명하지 못하는 정보를
> 반복 가능하게 제공하는가?"**

**예측 대상**: **미래 변동성** (방향 아님).
방향 예측은 v1에서 실패 확정 (엔트로피는 무질서도 → 방향성 없음).

---

## 2. EPD 정의 (v2.0, 동결)
r_t = log(P_t / P_{t−1})
PE_t = permutation_entropy(r[t−w_pe:t], m, tau)
ΔPE_t = PE_t − PE_{t−1}
Z_r(t) = (r_t − μ_r[t−w_z:t]) / σ_r[t−w_z:t]
Z_e(t) = (ΔPE_t − μ_e[t−w_z:t]) / σ_e[t−w_z:t]
EPD_raw = Z_r − Z_e
EPD_ema = EMA(EPD_raw, ema_span)
EPD_100 = 50 · (1 + tanh(EPD_ema / tanh_scale))

text

**모든 z-score는 causal** (t 시점은 t−1까지의 데이터만 사용).

---

## 3. 파라미터 (이론적 고정, 튜닝 금지)

| 파라미터 | 값 | 근거 |
|---|---|---|
| `w_pe` | 60 | m=3에서 6패턴 × 10관측, 분기 주기 |
| `m` | 3 | Bandt-Pompe 2002 표준 |
| `tau` | 1 | 일봉 표준 |
| `w_z` | 252 | 1년 거래일 |
| `tanh_scale` | 2.0 | RSI 70/30 정렬 |
| `ema_span` | 14 | RSI Wilder smoothing과 동일 |

**Stage 2의 parameter robustness는 "이 값 주변에서 안정한가" 검증이며,
"최적값 탐색"이 아니다.**

---

## 4. 임계값 (자산 무관, 절대값)

| EPD_100 | 해석 |
|---|---|
| > 70 | 과열 (가격이 엔트로피보다 강함) |
| 30 ~ 70 | 중립 |
| < 30 | 구조 형성 중 (엔트로피가 가격보다 강함) |

---

## 5. 검증 대상 (Stage별)

| Stage | 질문 | 대상 |
|---|---|---|
| 1 | 합성 데이터에서 의도대로 작동? | 4 synthetic tests |
| 2 | seed/sample/파라미터 변화에도 안정? | 150 조합 |
| 3 | 실데이터에서 변동성 예측? | 12자산 |
| 4 | 기존 변동성 지표 통제 후에도? | ΔR² |
| 5 | OOS에서도 유지? | Train/Val/Test |

---

## 6. Pass/Fail 기준 (사전 고정)

### Stage 1 — Synthetic (4 tests)

| Test | PASS 기준 |
|---|---|
| A. White noise | IC ≈ 0, 블록 2σ 초과 < 30% |
| B. GARCH | return IC < 0.05 |
| C. Regime | 전환 ±5일 내 \|epd_raw_100 − 50\| > 20 비율 ≥ 50% |
|            | (raw EPD 사용 — EMA는 스무딩용이므로 전환 감지엔 부적합) |
| D. Divergence | 이벤트 구간 극단값 비율 > 배경의 2배 |

**판정**: 4/4 PASS

### Stage 2 — Robustness (v2.0)

검증 항목 (real market 중심):
1. **Sample robustness**: 3자산 × 3기간 = 9조합
   PASS: |rho| > 0.05 비율 ≥ 80%
2. **Parameter sensitivity**: SPX, 27조합
   PASS: |rho| > 0.05 비율 ≥ 80%

참고 (판정 제외):
3. Seed robustness (synthetic): 진단 정보만.
   이유: EPD의 대상은 real market 변동성. 합성 GARCH/divergence는
   EPD의 부가가치 대상이 아님 (Stage 1에서 정의 검증 완료).

### Stage 3 — Real Market

| 자산군 | 기준 |
|---|---|
| US index (SPX, NASDAQ) | Q5/Q1 > 1.10, rho > 0.10 |
| KR index (KOSPI, KOSDAQ) | Q5/Q1 > 1.10, rho > 0.05 |
| KR stock (7종목) | Q5/Q1 > 1.10, rho > 0.05 |

**PASS 기준**: 각 자산군 **70% 이상 통과**

### Stage 4 — Incremental
M1: future_vol ~ RV20
M2: future_vol ~ RV20 + |EPD_raw|

text

**PASS 기준**: 
- ΔR²(M2−M1) > 0.005 (평균)
- EPD 계수 p<0.05 비율 ≥ 70%

### Stage 5 — Walk-Forward

**분할**: Train 05-14 / Val 15-17 / Test 18-25

**PASS 기준 (Test)**:
- |rho| > 0.10
- IS → OOS 열화 < 30%

---

## 7. 실패 조건 (즉시 폐기)

| 조건 | 정의 |
|---|---|
| F1. 정의 문제 | Stage 1 실패 |
| F2. Robustness 없음 | Stage 2 |rho|>0.10 비율 < 50% |
| F3. 자산 특화 | Stage 3에서 1개 자산군만 통과 |
| F4. 기존 지표와 중복 | Stage 4 ΔR² < 0.002 |
| F5. OOS 소멸 | Stage 5 열화 > 50% |

**F2, F3, F5는 "조건부 생존" 판정 가능** (범위 축소).

---

## 8. 산출물

| Stage | 문서 | 스크립트 |
|---|---|---|
| 0 | 이 문서 | — |
| 1 | STAGE_1_SYNTHETIC.md | stage_1_synthetic.py |
| 2 | STAGE_2_ROBUSTNESS.md | stage_2_robustness.py |
| 3 | STAGE_3_REAL_MARKET.md | stage_3_real_market.py |
| 4 | STAGE_4_INCREMENTAL.md | stage_4_incremental.py |
| 5 | STAGE_5_WALKFORWARD.md | stage_5_walkforward.py |
| — | FINAL_REPORT.md | — |

---

## 9. 재현 환경

- Python >= 3.10
- numpy >= 1.24
- yfinance >= 0.2 (Stage 3~5)
- pytest >= 7.0 (Stage 1)

```bash
pip install -e v2/packages/epd_core
python v2/scripts/stage_1_synthetic.py
python v2/scripts/stage_2_robustness.py
```
이 문서는 실험 시작 전 확정되며, 이후 결과에 따라 수정되지 않는다.

text

---

## 3. 다음 단계

**파일 1을 저장하고 알려주세요.** 그 다음 **파일 2 (`STAGE_1_SYNTHETIC.md`) + `stage_1_synthetic.py`** 순서로 갑니다.

**진행 순서**:
1. ✅ `v2/docs/STAGE_0_SPEC.md` ← 지금
2. `v2/scripts/stage_1_synthetic.py`
3. `v2/docs/STAGE_1_SYNTHETIC.md`
4. `v2/scripts/stage_2_robustness.py` ← **핵심 신규**
5. `v2/docs/STAGE_2_ROBUSTNESS.md`
6. ... (Stage 3, 4, 5)
7. `v2/docs/FINAL_REPORT.md`
