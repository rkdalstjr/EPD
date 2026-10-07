```markdown
# FINAL DECISION — EPD 프로젝트

**판정일**: 2026-10-06
**판정**: **조건부 생존 (CONDITIONAL PASS)**

---

## 1. 최종 결론

> **EPD v3.1 = `100 · tanh(||Z_r| − |Z_e|| / 2)`**
> 
> 가격 스트레스와 엔트로피 이탈의 magnitude divergence를 측정하는
> **시장 상태 변수**. RSI와 유사한 카테고리지만 다른 축.

---

## 2. 성공한 것

1. **엔트로피 표현 개선**: dPE → residual (E4)
2. **절대값 도입**: signed → \|·\|
3. **Magnitude 확정**: R6 = `||Z_r| − |Z_e||`
4. **시간 안정성**: 82.3% 양수 (Stage 10)
5. **Distinctiveness**: RSI/ATR/MACD와 독립
6. **N1 normalization**: monotonicity 완전 유지

## 3. 실패한 것

1. **방향 예측**: 구조적 불가
2. **dPE**: 슬라이드 노이즈
3. **Ze 부호**: 정보 없음
4. **RSI급 보편성**: 미달
5. **개별주 편차**: KR 개별주 약함

## 4. 핵심 발견

> **"엔트로피 변화(dPE)는 forward 정보를 담지 않는다.
> 그러나 엔트로피 이탈 magnitude(|Z_e|)는 가격 스트레스와 함께
> 새로운 상태 정보를 제공한다."**

## 5. 최종 산출물

- `epd_core v3.1` (pip installable)
- `EPD_SPEC.md v3.1`
- 시각화 UI
- 11 Stages 검증 데이터

## 6. 다음 프로젝트

- **TRA** (Trend-Range-Ambiguity)
- **KSE** (Kurtosis-Skew-Entropy)
- **3차원 상태공간**: EPD × TRA × KSE

---

**이 문서는 EPD 프로젝트의 최종 기록이다.**