# Stage 1 — Synthetic Validation

- Seed: 20261006
- N: 10000
- 결과: **4/4 (PASS)**

## 결과 표

| Test | 핵심 지표 | 판정 |
|---|---|---|
| A_white_noise | IC_ret=0.0100, IC_vol=-0.0142, 2σ=20% | ✅ PASS |
| B_garch | IC_ret=0.0189, IC_vol=-0.0312 | ✅ PASS |
| C_regime | 전환 207회, 적중 97.1% | ✅ PASS |
| D_divergence | 이벤트 63.7%, 배경 28.9%, 비율 2.20 | ✅ PASS |

**Overall**: PASS