"""Stage 4 — State Interpretability.

질문: EPD가 서로 다른 시장 상태를 구분하는가?

상태 정의 (4분면):
  A: Z_r > +1, Z_e < -1  (가격↑ + 규칙성↑) → "정렬된 상승"
  B: Z_r > +1, Z_e > +1  (가격↑ + 규칙성↓) → "구조 흐트러진 상승"
  C: Z_r < -1, Z_e < -1  (가격↓ + 규칙성↑) → "정렬된 하락"
  D: Z_r < -1, Z_e > +1  (가격↓ + 규칙성↓) → "구조 붕괴"

측정: 각 상태에서 forward 통계
  - forward volatility (5, 10, 20일)
  - forward autocorr (5일)
  - forward |return| (5일)

PASS:
  - 4 상태 간 forward vol 평균 ANOVA p < 0.05
  - 최소 3개 상태 쌍에서 pairwise 유의 (Bonferroni)
  - 자산군별 통과율 >= 70%
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd

warnings.filterwarnings("ignore")

TICKERS = [
    ("^GSPC", "SPX", "US_index"),
    ("^IXIC", "NASDAQ", "US_index"),
    ("^KS11", "KOSPI", "KR_index"),
    ("^KQ11", "KOSDAQ", "KR_index"),
    ("005930.KS", "삼성전자", "KR_stock"),
    ("000660.KS", "SK하이닉스", "KR_stock"),
    ("005380.KS", "현대차", "KR_stock"),
    ("035420.KS", "NAVER", "KR_stock"),
    ("051910.KS", "LG화학", "KR_stock"),
    ("005490.KS", "POSCO홀딩스", "KR_stock"),
    ("068270.KS", "셀트리온", "KR_stock"),
    ("373220.KS", "LG엔솔", "KR_stock"),
]

Z_THRESHOLD = 1.0
H_VOL = 5
ANOVA_ALPHA = 0.05
BONFERRONI_PAIRS = 6  # C(4,2)
PAIRWISE_ALPHA = 0.05 / BONFERRONI_PAIRS  # ≈ 0.0083
GROUP_PASS_RATE = 0.70
N_PAIRWISE_MIN = 3


# ---- 통계 유틸 ----
def anova_f(groups):
    """One-way ANOVA F-test (자체 구현)."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) >= 5]
    if len(groups) < 2:
        return np.nan, np.nan

    k = len(groups)
    n_total = sum(len(g) for g in groups)
    if n_total <= k:
        return np.nan, np.nan

    grand_mean = np.concatenate(groups).mean()
    ssb = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b = k - 1
    df_w = n_total - k
    if df_w <= 0 or ssw == 0:
        return np.nan, np.nan
    msb = ssb / df_b
    msw = ssw / df_w
    F = msb / msw if msw > 0 else np.nan

    # p-value (F distribution 상단 꼬리)
    try:
        from scipy.stats import f as f_dist

        p = float(1 - f_dist.cdf(F, df_b, df_w))
    except ImportError:
        # scipy 없으면 대략적 판정 (F > 3.0 크면 유의)
        p = 1.0 if F < 3.0 else 0.0

    return float(F), p


def pairwise_t(g1, g2):
    """Two-sample Welch's t-test (자체 구현)."""
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    if len(g1) < 5 or len(g2) < 5:
        return np.nan
    m1, m2 = g1.mean(), g2.mean()
    v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
    n1, n2 = len(g1), len(g2)
    se = np.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return np.nan
    t = (m1 - m2) / se
    try:
        from scipy.stats import t as t_dist

        # Welch-Satterthwaite df
        df = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        )
        p = float(2 * (1 - t_dist.cdf(abs(t), df)))
    except ImportError:
        # 근사
        p = 1.0 if abs(t) < 2.5 else 0.0
    return p


def forward_vol(r, h):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        win = r[t + 1 : t + 1 + h]
        win = win[np.isfinite(win)]
        if win.size >= 3:
            fv[t] = win.std(ddof=1)
    return fv


def classify_state(z_r, z_e, thr=Z_THRESHOLD):
    """4분면 상태 라벨."""
    state = np.full(len(z_r), -1, dtype=int)  # -1 = none
    valid = np.isfinite(z_r) & np.isfinite(z_e)
    A = valid & (z_r > +thr) & (z_e < -thr)
    B = valid & (z_r > +thr) & (z_e > +thr)
    C = valid & (z_r < -thr) & (z_e < -thr)
    D = valid & (z_r < -thr) & (z_e > +thr)
    state[A] = 0
    state[B] = 1
    state[C] = 2
    state[D] = 3
    return state


STATE_NAMES = [
    "A(가격↑규칙↑)",
    "B(가격↑규칙↓)",
    "C(가격↓규칙↑)",
    "D(가격↓규칙↓)",
]


def main():
    import yfinance as yf

    print("=" * 120)
    print("Stage 4 — State Interpretability")
    print(f"Z threshold=±{Z_THRESHOLD}, H_vol={H_VOL}")
    print(f"ANOVA α={ANOVA_ALPHA}, Bonferroni α={PAIRWISE_ALPHA:.4f}")
    print("=" * 120)

    all_rows = []

    for tkr, name, grp in TICKERS:
        try:
            df = yf.download(
                tkr,
                start="2005-01-01",
                end="2025-12-31",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df is None or len(df) < 500:
                continue
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            if close.size < 500:
                continue
        except Exception as e:
            print(f"{name} err: {e}")
            continue

        out = compute_epd(close)
        z_r = out["return_z"]
        z_e = out["entropy_z"]

        r = np.full_like(close, np.nan)
        r[1:] = np.diff(np.log(close))
        fv = forward_vol(r, H_VOL)

        state = classify_state(z_r, z_e)

        # 각 상태의 forward vol 분포
        groups = []
        counts = []
        means = []
        for s in range(4):
            mask = (state == s) & np.isfinite(fv)
            g = fv[mask]
            groups.append(g)
            counts.append(len(g))
            means.append(g.mean() if len(g) > 0 else np.nan)

        F, p_anova = anova_f(groups)

        # Pairwise (Bonferroni)
        pairwise_sig = 0
        pairwise_p = np.full((4, 4), np.nan)
        for i in range(4):
            for j in range(i + 1, 4):
                p_ij = pairwise_t(groups[i], groups[j])
                pairwise_p[i, j] = p_ij
                if np.isfinite(p_ij) and p_ij < PAIRWISE_ALPHA:
                    pairwise_sig += 1

        passed = (
            np.isfinite(p_anova)
            and p_anova < ANOVA_ALPHA
            and pairwise_sig >= N_PAIRWISE_MIN
            and all(c >= 20 for c in counts)
        )

        all_rows.append(
            {
                "name": name,
                "grp": grp,
                "counts": counts,
                "means": means,
                "F": F,
                "p_anova": p_anova,
                "pairwise_sig": pairwise_sig,
                "passed": passed,
            }
        )

        # 출력
        print(f"\n--- {name} ({grp}) ---")
        print(
            f"  상태별 개수: A={counts[0]:5d}  B={counts[1]:5d}  "
            f"C={counts[2]:5d}  D={counts[3]:5d}"
        )
        print(f"  forward vol (5d):")
        for s in range(4):
            print(f"    {STATE_NAMES[s]:>16s}: {means[s]:.6f}")
        print(f"  ANOVA F={F:.3f}, p={p_anova:.4f}")
        print(
            f"  유의 pair (Bonferroni {PAIRWISE_ALPHA:.4f}): "
            f"{pairwise_sig}/{BONFERRONI_PAIRS}"
        )
        print(f"  → {'✅ PASS' if passed else '❌ FAIL'}")

    # ----- 자산군별 요약 -----
    print()
    print("=" * 120)
    print("자산군별 요약")
    print("=" * 120)
    print(
        f"{'group':>12s}  {'n':>3s}  "
        f"{'mean F':>8s}  {'mean p':>8s}  "
        f"{'mean sig pairs':>15s}  {'통과율':>8s}  {'판정':>6s}"
    )
    print("-" * 90)

    group_res = {}
    for grp in ["US_index", "KR_index", "KR_stock"]:
        sub = [r for r in all_rows if r["grp"] == grp]
        if not sub:
            continue
        Fs = [r["F"] for r in sub if np.isfinite(r["F"])]
        ps = [r["p_anova"] for r in sub if np.isfinite(r["p_anova"])]
        sigs = [r["pairwise_sig"] for r in sub]
        npass = sum(1 for r in sub if r["passed"])
        frac = npass / len(sub)
        ok = frac >= GROUP_PASS_RATE
        group_res[grp] = {"frac": frac, "ok": ok}
        print(
            f"{grp:>12s}  {len(sub):>3d}  "
            f"{np.mean(Fs):>8.3f}  {np.mean(ps):>8.4f}  "
            f"{np.mean(sigs):>15.2f}  "
            f"{frac:>7.1%}  {'✅' if ok else '❌':>6s}"
        )

    overall = all(g["ok"] for g in group_res.values())
    print()
    print(f"=== Stage 4 Result: {'PASS' if overall else 'FAIL'} ===")

    # ----- 리포트 -----
    rpt = ROOT / "docs" / "STAGE_4_STATE_INTERPRETABILITY.md"
    L = [
        "# Stage 4 — State Interpretability",
        "",
        f"- Z threshold: ±{Z_THRESHOLD}",
        f"- forward window: {H_VOL}일",
        f"- ANOVA α={ANOVA_ALPHA}, Bonferroni α={PAIRWISE_ALPHA:.4f}",
        "",
        "## 상태 정의",
        "",
        "| 상태 | 조건 | 의미 |",
        "|---|---|---|",
        "| A | Z_r>+1, Z_e<-1 | 가격↑ + 규칙성↑ (정렬된 상승) |",
        "| B | Z_r>+1, Z_e>+1 | 가격↑ + 규칙성↓ (구조 흐트러진 상승) |",
        "| C | Z_r<-1, Z_e<-1 | 가격↓ + 규칙성↑ (정렬된 하락) |",
        "| D | Z_r<-1, Z_e>+1 | 가격↓ + 규칙성↓ (구조 붕괴) |",
        "",
        "## 자산별 결과",
        "",
        "| Asset | Group | A_n | B_n | C_n | D_n | "
        "A_vol | B_vol | C_vol | D_vol | F | p | sig_pairs | Pass |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in all_rows:
        c = r["counts"]
        m = r["means"]
        L.append(
            f"| {r['name']} | {r['grp']} | "
            f"{c[0]} | {c[1]} | {c[2]} | {c[3]} | "
            f"{m[0]:.5f} | {m[1]:.5f} | {m[2]:.5f} | {m[3]:.5f} | "
            f"{r['F']:.2f} | {r['p_anova']:.4f} | "
            f"{r['pairwise_sig']} | "
            f"{'✅' if r['passed'] else '❌'} |"
        )

    L += [
        "",
        "## 자산군 요약",
        "",
        "| Group | n | mean F | mean p | mean sig | 통과율 | 판정 |",
        "|---|---|---|---|---|---|---|",
    ]
    for grp, g in group_res.items():
        sub = [r for r in all_rows if r["grp"] == grp]
        Fs = [r["F"] for r in sub if np.isfinite(r["F"])]
        ps = [r["p_anova"] for r in sub if np.isfinite(r["p_anova"])]
        sigs = [r["pairwise_sig"] for r in sub]
        L.append(
            f"| {grp} | {len(sub)} | {np.mean(Fs):.2f} | "
            f"{np.mean(ps):.4f} | {np.mean(sigs):.2f} | "
            f"{g['frac']:.1%} | {'✅' if g['ok'] else '❌'} |"
        )

    L += ["", f"**Overall**: {'PASS' if overall else 'FAIL'}"]
    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\nreport -> {rpt}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
