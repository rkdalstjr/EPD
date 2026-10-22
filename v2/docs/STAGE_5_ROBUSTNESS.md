# Stage 5 — Robustness

**판정일**: 2026-10-06
**스크립트**: `scripts/stage_5_robustness.py`
**결과**: **조건부 PASS**

## 검증 항목

1. **Parameter neighborhood**: w_pe, w_z, resid_win 변화 시 성능 유지?
2. **Time stability**: 4기간 분할

## 결과 요약

- Parameter: 27조합 중 80%+ 안정
- Time: 4기간 모두 평균 partial > 0

## 결론

v3.1 파라미터 (60/3/1/252/504/2.0)는 **로버스트 영역의 중심값**으로 확정.