# Entropy Alternatives Research (2026-10)

**상태**: 완료, 기각
**결론**: 모든 대안이 v3.1을 대체하지 못함

## 후보 및 결과

| 후보 | 검증 | 결과 |
|---|---|---|
| Weighted PE (variance) | Stage 2/10 | mean partial 0.021 (A: 0.037) → **나쁨** |
| Dispersion Entropy | Stage 2/10 | mean partial 0.032 (A: 0.037) → 동등 |
| 3일 RMS Z_r | Stage 2/10 + 비선형 통제 | raw corr 우위, partial 열화 → **RMS 프록시** |
| Regime-conditional (A/D 혼합) | Stage 2/10 + 시간 순서 | partial A 대비 -0.0002 → **개선 없음** |

## 주요 교훈

1. **rank-collinearity**: `rank(RMS) = rank(RMS²)` → Spearman 이중 통제는 raw/binned로 재검증 필수
2. **프록시 통제**: "개선" 발견 시 변동성/모멘텀 프록시 통제부터
3. **시간 스케일 정렬**: 이론적 타당성만으로 개선 보장 안 됨
4. **PE 정의는 안정적**: amplitude blindness가 결함이 아니라 특성

## 결론

> 표준 PE (m=3, w=60) + 일간 Z_r 조합이 현재 검증 범위에서 최선.
> v3.1 확정.