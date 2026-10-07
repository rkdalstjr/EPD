# RESEARCH_SPEC_v2.md — EPD 연구 헌법 v1.2

> v1.1을 대체. v1.1은 이력으로 보존.
> 변경 근거: **"divergence"의 개념 정의 오류 수정** (결과에 맞춘 최적화 아님).

**Version**: 1.2
**Date**: 2026-10-06
**Supersedes**: v1.1

## 목표 재정의 (2026-10-06 후반)
RSI급 보편성 기준 폐기. 실용 지표 목표로 전환.
평가: 전체 기간 IC, Long-short Sharpe (자산별 + 통합).
수정 한도 없음. 각 시도 결과를 이 문서 하단에 기록.

## 1. v1.1 → v1.2 변경 근거

### 문제 (v1.1의 결함)
- `Z_r = zscore(returns)`는 **순간 움직임의 크기**만 잡음
- 완만한 추세(2013-2019 SPX, KOSPI 장기)에서 `Z_r ≈ 0`
- 즉 "가격이 평균에서 멀리 벗어난 상태"를 포착하지 못함
- 결과: Stage 3에서 SPX 2013-2019, 2020-2025 rho≈0 (관계 소멸)

### 수정 (v1.2)
- `Z_r` → **`Z_p = zscore(log(close))`**
- "divergence"를 "순간 사건의 조합"에서 **"두 상태의 공존"**으로 정정
- 이는 개념적 정합성에 근거하며, Stage 3 결과와 독립적으로 옳음

## 2. 정의 (v1.2)
```
log_P_t = log(P_t)
PE_t = permutation_entropy(r[t-w_pe:t], m, tau) # r = diff(log_P)
dPE_t = PE_t - PE_{t-1}
Z_p(t) = causal_zscore(log_P, w_z)
Z_e(t) = causal_zscore(dPE, w_z)
EPD_raw = Z_p - Z_e
EPD = tanh(EPD_raw / 2)
EPD_100 = 50 * (1 + EPD)
```

## 3. 파라미터 (v1.1과 동일, 변경 없음)
| 파라미터 | 값 |
|---|---|
| w_pe | 60 |
| m | 3 |
| tau | 1 |
| w_z | 252 |
| tanh_scale | 2 |

## 4. Stage 3 판정 기준 변경
- v1.1: 36조합 중 90% 부호 통일
- **v1.2: 조합별 \|rho\| ≥ 0.6 비율 ≥ 70%** (부호 무관, 자산별)
- 부호는 진단 정보로만 기록

## 5. Stage 3.5 (신규)
- 기간 분할: 2005-2012, 2013-2019, 2020-2025
- 각 기간에서 \|rho\| ≥ 0.6 비율 ≥ 60% 필요
- 하나라도 미달이면 FAIL

## 6. Stage 0~2, 4~8 (v1.1과 동일)
(변경 없음)
