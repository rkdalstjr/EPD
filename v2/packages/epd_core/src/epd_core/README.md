# epd-core

**Entropy-Price Divergence (EPD)** — 가격 움직임과 순열 엔트로피 변화의
불일치를 측정하는 **구조적 긴장(structural tension)** 지표.

## ⚠️ 중요

EPD는 **방향 예측 지표가 아닙니다**. RSI처럼 "지금 시장 구조가 어떤
상태인가"를 표시하는 **상태 지표(state indicator)** 입니다.

- ✅ 미래 변동성 증가와 상관 (Stage 4, 5 검증)
- ❌ 수익률 방향 예측 불가 (Stage 3 실패)
- ❌ 리스크 필터로 단독 사용 시 성능 저하 (Stage 6)

## 설치

```bash
pip install epd-core                       # 배포 시
pip install -e /path/to/packages/epd_core  # 로컬 개발