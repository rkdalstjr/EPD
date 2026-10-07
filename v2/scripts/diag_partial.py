"""Z_r 통제 후 Z_e의 조건부 정보 검증.

세 가지 접근:
1. Partial correlation: corr(Z_e, fwd | Z_r)
2. Multiple regression: fwd ~ Z_r + Z_e (+ interaction)
3. Conditional bins: Z_r 구간별 Z_e 효과
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
    ("^GSPC", "SPX", "US"),
    ("^IXIC", "NASDAQ", "US"),
    ("^KS11", "KOSPI", "KR"),
    ("^KQ11", "KOSDAQ", "KR"),
    ("005930.KS", "삼성전자", "KR"),
    ("000660.KS", "SK하이닉스", "KR"),
    ("035420.KS", "NAVER", "KR"),
    ("051910.KS", "LG화학", "KR"),
]


def forward_vol(r, h=5):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.std(ddof=1)
    return fv


def ols(y, X):
    """OLS, returns (beta, se, r2, t_stats, p_values)."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    rss = (resid**2).sum()
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - rss / tss if tss > 0 else np.nan

    sigma2 = rss / (n - k)
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    return beta, se, r2, t


def partial_corr(x, y, z):
    """corr(x, y | z) — z를 통제한 x, y의 편상관."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 100:
        return np.nan
    x, y, z = x[m], y[m], z[m]

    # x ~ z 잔차
    Z = np.column_stack([np.ones(len(z)), z])
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]
    rx = x - Z @ bx

    # y ~ z 잔차
    by = np.linalg.lstsq(Z, y, rcond=None)[0]
    ry = y - Z @ by

    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    import yfinance as yf

    print("=" * 130)
    print("Z_r 통제 후 Z_e의 조건부 정보 검증")
    print("=" * 130)

    results = []

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
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            if close.size < 500:
                continue
        except Exception:
            continue

        out = compute_epd(close)
        z_r, z_e = out["return_z"], out["entropy_z"]
        r = np.full_like(close, np.nan)
        r[1:] = np.diff(np.log(close))
        fv = forward_vol(r, 5)

        m = np.isfinite(z_r) & np.isfinite(z_e) & np.isfinite(fv)
        y = fv[m]
        xr = z_r[m]
        xe = z_e[m]
        n = m.sum()

        print(f"\n--- {name} ({grp}), n={n} ---")

        # 1. 단순 상관
        rho_r = float(np.corrcoef(xr, y)[0, 1])
        rho_e = float(np.corrcoef(xe, y)[0, 1])
        rho_re = float(np.corrcoef(xr, xe)[0, 1])
        print(f"\n  [1] 단순 상관")
        print(f"      corr(Z_r, fwd_vol) = {rho_r:+.4f}")
        print(f"      corr(Z_e, fwd_vol) = {rho_e:+.4f}")
        print(f"      corr(Z_r, Z_e)     = {rho_re:+.4f}")

        # 2. Partial correlation
        pc = partial_corr(xe, y, xr)
        print(f"\n  [2] 편상관")
        print(f"      corr(Z_e, fwd_vol | Z_r) = {pc:+.4f}")

        # 3. Multiple regression
        c = np.ones(n)
        X1 = np.column_stack([c, xr])
        X2 = np.column_stack([c, xr, xe])
        X3 = np.column_stack([c, xr, xe, xr * xe])

        beta1, _, r2_1, t1 = ols(y, X1)
        beta2, _, r2_2, t2 = ols(y, X2)
        beta3, _, r2_3, t3 = ols(y, X3)

        print(f"\n  [3] 회귀")
        print(f"      M1: fwd ~ Z_r              R²={r2_1:.4f}")
        print(f"      M2: fwd ~ Z_r + Z_e        R²={r2_2:.4f}  ΔR²={r2_2-r2_1:+.4f}")
        print(
            f"      M3: fwd ~ Z_r + Z_e + Z_r×Z_e  R²={r2_3:.4f}  ΔR²={r2_3-r2_2:+.4f}"
        )
        print(f"      t(Z_e) in M2       = {t2[2]:+.3f}")
        print(f"      t(Z_r×Z_e) in M3   = {t3[3]:+.3f}")

        # 4. Z_r 구간별 Z_e 효과
        print(f"\n  [4] Z_r 분위별 Z_e 효과 (상관)")
        qs = np.quantile(xr, [0, 0.25, 0.5, 0.75, 1.0])
        for i in range(4):
            mask = (xr >= qs[i]) & (xr <= qs[i + 1])
            if mask.sum() < 50:
                continue
            sub_e = xe[mask]
            sub_y = y[mask]
            if sub_e.std() == 0 or sub_y.std() == 0:
                continue
            rho = float(np.corrcoef(sub_e, sub_y)[0, 1])
            print(
                f"      Z_r Q{i+1} [{qs[i]:+.2f}, {qs[i+1]:+.2f}]: "
                f"n={mask.sum():5d}, corr(Z_e, fwd)={rho:+.4f}"
            )

        results.append(
            {
                "name": name,
                "grp": grp,
                "rho_r": rho_r,
                "rho_e": rho_e,
                "partial": pc,
                "dr2_m2": r2_2 - r2_1,
                "dr2_m3": r2_3 - r2_2,
                "t_e": t2[2],
                "t_int": t3[3],
            }
        )

    # ----- 요약 -----
    print()
    print("=" * 130)
    print("요약")
    print("=" * 130)
    print(
        f"{'asset':>14s}  {'grp':>4s}  "
        f"{'corr(Ze,fwd)':>14s}  {'partial(Ze|Zr)':>16s}  "
        f"{'ΔR²(M2)':>10s}  {'t(Ze)':>8s}  {'t(int)':>8s}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r['name']:>14s}  {r['grp']:>4s}  "
            f"{r['rho_e']:>+14.4f}  {r['partial']:>+16.4f}  "
            f"{r['dr2_m2']:>+10.4f}  {r['t_e']:>+8.3f}  {r['t_int']:>+8.3f}"
        )

    # 판정
    parts = [abs(r["partial"]) for r in results if np.isfinite(r["partial"])]
    dr2s = [r["dr2_m2"] for r in results if np.isfinite(r["dr2_m2"])]
    ts = [abs(r["t_e"]) for r in results if np.isfinite(r["t_e"])]
    t_ints = [abs(r["t_int"]) for r in results if np.isfinite(r["t_int"])]

    print()
    print(f"  mean |partial(Ze|Zr)| = {np.mean(parts):.4f}")
    print(f"  mean ΔR²(M2)          = {np.mean(dr2s):+.4f}")
    print(f"  mean |t(Z_e)|         = {np.mean(ts):.3f}")
    print(f"  mean |t(Z_r×Z_e)|     = {np.mean(t_ints):.3f}")

    print()
    print("판정:")
    print("  - |partial| > 0.05 → Z_e가 조건부 정보 있음")
    print("  - t(Z_e) > 2 (평균) → 회귀에서 유의")
    print("  - t(int) > 2 → 상호작용 존재")


if __name__ == "__main__":
    raise SystemExit(main())
