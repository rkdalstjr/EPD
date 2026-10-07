"""Z_e variants 비교 — 4가지 엔트로피 표현.

E1 (현재):  Ze = Z(dPE), dPE = PE_t - PE_{t-1}
E2 (level): Ze = Z(PE)
E3 (trend): Ze = Z(EMA(PE,10) - EMA(PE,10)[5])
E4 (resid): Ze = Z(PE - E[PE | |Z_r|])   (causal rolling regression)

측정 (동일 조건):
  - corr(Ze, fwd_vol_5d)
  - corr(Ze, fwd_persist_5d)
  - partial corr (Z_r 통제 후)
  - 회귀 ΔR² (Z_r + Ze)
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import rolling_pe, rolling_zscore

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

W_PE = 60
W_Z = 252
EMA_K = 10
SLOPE_LAG = 5
RESID_WIN = 504  # E4 rolling regression window


# ============================================================
#  유틸
# ============================================================
def ema_np(x, span):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    alpha = 2.0 / (span + 1.0)
    prev = np.nan
    for t in range(len(x)):
        xt = x[t]
        if not np.isfinite(xt):
            out[t] = prev
        else:
            prev = xt if not np.isfinite(prev) else alpha * xt + (1 - alpha) * prev
            out[t] = prev
    return out


def forward_vol(r, h=5):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.std(ddof=1)
    return fv


def forward_persist(r, h=5):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        if not np.isfinite(r[t]) or r[t] == 0:
            continue
        cs = np.sign(r[t])
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size < 3:
            continue
        fv[t] = (np.sign(w) == cs).mean()
    return fv


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    xr = np.argsort(np.argsort(x[m]))
    yr = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(xr, yr)[0, 1])


def partial_corr(x, y, z):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 100:
        return np.nan
    x, y, z = x[m], y[m], z[m]
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def ols_delta_r2(y, x1, x2):
    """M1: y ~ 1 + x1, M2: y ~ 1 + x1 + x2. Returns ΔR², t(x2)."""
    m = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    if m.sum() < 100:
        return np.nan, np.nan
    y, x1, x2 = y[m], x1[m], x2[m]
    n = len(y)
    c = np.ones(n)
    X1 = np.column_stack([c, x1])
    X2 = np.column_stack([c, x1, x2])

    def r2(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        return 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum(), beta, resid

    r2_1, _, _ = r2(X1)
    r2_2, beta2, resid2 = r2(X2)
    # t for x2
    sigma2 = (resid2**2).sum() / (n - 3)
    cov = sigma2 * np.linalg.inv(X2.T @ X2)
    se = np.sqrt(np.diag(cov))
    t_x2 = beta2[2] / se[2] if se[2] > 0 else np.nan
    return r2_2 - r2_1, t_x2


# ============================================================
#  Z_e variants
# ============================================================
def compute_ze_variants(z_r, pe, r):
    """4가지 Z_e 반환."""
    n = len(pe)
    eps = 1e-9

    # --- E1: Z(dPE) ---
    dpe = np.full(n, np.nan)
    dpe[1:] = pe[1:] - pe[:-1]
    ze1 = rolling_zscore(dpe, w_z=W_Z)

    # --- E2: Z(PE level) ---
    ze2 = rolling_zscore(pe, w_z=W_Z)

    # --- E3: Z(EMA(PE) - EMA(PE)[lag]) ---
    pe_ema = ema_np(pe, EMA_K)
    pe_mom = np.full(n, np.nan)
    pe_mom[SLOPE_LAG:] = pe_ema[SLOPE_LAG:] - pe_ema[:-SLOPE_LAG]
    ze3 = rolling_zscore(pe_mom, w_z=W_Z)

    # --- E4: Z(PE - E[PE | |Z_r|]) ---
    # |Z_r|를 5분위 bin으로 나눠, 각 bin 내 rolling mean of PE를 기대값으로
    abs_zr = np.abs(z_r)
    resid = np.full(n, np.nan)
    # rolling window에서 abs_zr 분위 경계 계산 → 각 PE에 대해 같은 bin의 평균 차감
    for t in range(RESID_WIN, n):
        win_zr = abs_zr[t - RESID_WIN : t]
        win_pe = pe[t - RESID_WIN : t]
        mask = np.isfinite(win_zr) & np.isfinite(win_pe)
        if mask.sum() < 100:
            continue
        w_z = win_zr[mask]
        w_p = win_pe[mask]
        if not np.isfinite(abs_zr[t]) or not np.isfinite(pe[t]):
            continue
        # 5분위 경계
        qs = np.quantile(w_z, np.linspace(0, 1, 6))
        qs[0] -= eps
        qs[-1] += eps
        # 현재값의 bin
        b = int(np.searchsorted(qs, abs_zr[t], side="right") - 1)
        b = max(0, min(4, b))
        # bin 평균
        bin_mask = (w_z >= qs[b]) & (w_z < qs[b + 1])
        if bin_mask.sum() < 5:
            continue
        expected = w_p[bin_mask].mean()
        resid[t] = pe[t] - expected

    ze4 = rolling_zscore(resid, w_z=W_Z)

    return {
        "E1_dPE": ze1,
        "E2_level": ze2,
        "E3_trend": ze3,
        "E4_residual": ze4,
    }


# ============================================================
#  Main
# ============================================================
def main():
    import yfinance as yf

    print("=" * 130)
    print("Z_e variants — E1(dPE), E2(level), E3(trend), E4(residual)")
    print("=" * 130)

    results = {}  # (asset, variant) -> dict of metrics

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
            if close.size < 1000:
                continue
        except Exception:
            continue

        log_p = np.log(close)
        r = np.full_like(close, np.nan)
        r[1:] = np.diff(log_p)

        pe = rolling_pe(r, w_pe=W_PE, m=3, tau=1)
        z_r = rolling_zscore(r, w_z=W_Z)

        fv = forward_vol(r, 5)
        fp = forward_persist(r, 5)

        variants = compute_ze_variants(z_r, pe, r)

        print(f"\n### {name} ({grp})")
        print(
            f"  {'variant':>16s}  "
            f"{'rho(fv)':>10s}  {'rho(fp)':>10s}  "
            f"{'part(fv)':>10s}  {'part(fp)':>10s}  "
            f"{'dR2(fv)':>9s}  {'t_Ze(fv)':>10s}"
        )
        print("  " + "-" * 95)

        for vname, ze in variants.items():
            rho_fv = spearman(ze, fv)
            rho_fp = spearman(ze, fp)
            part_fv = partial_corr(ze, fv, z_r)
            part_fp = partial_corr(ze, fp, z_r)
            dr2_fv, t_fv = ols_delta_r2(fv, z_r, ze)

            print(
                f"  {vname:>16s}  "
                f"{rho_fv:>+10.4f}  {rho_fp:>+10.4f}  "
                f"{part_fv:>+10.4f}  {part_fp:>+10.4f}  "
                f"{dr2_fv:>+9.4f}  {t_fv:>+10.3f}"
            )

            results[(name, vname)] = {
                "rho_fv": rho_fv,
                "rho_fp": rho_fp,
                "part_fv": part_fv,
                "part_fp": part_fp,
                "dr2_fv": dr2_fv,
                "t_fv": t_fv,
            }

    # ----- 요약 (자산 평균) -----
    print()
    print("=" * 130)
    print("자산 평균")
    print("=" * 130)
    print(
        f"{'variant':>16s}  "
        f"{'mean|rho_fv|':>14s}  {'mean|rho_fp|':>14s}  "
        f"{'mean|part_fv|':>15s}  {'mean dR2':>10s}  "
        f"{'mean t_Ze':>11s}  {'t>2 count':>11s}"
    )
    print("-" * 110)

    for vname in ["E1_dPE", "E2_level", "E3_trend", "E4_residual"]:
        vals = [v for (a, vn), v in results.items() if vn == vname]
        if not vals:
            continue
        rho_fvs = [abs(v["rho_fv"]) for v in vals if np.isfinite(v["rho_fv"])]
        rho_fps = [abs(v["rho_fp"]) for v in vals if np.isfinite(v["rho_fp"])]
        part_fvs = [abs(v["part_fv"]) for v in vals if np.isfinite(v["part_fv"])]
        dr2s = [v["dr2_fv"] for v in vals if np.isfinite(v["dr2_fv"])]
        ts = [v["t_fv"] for v in vals if np.isfinite(v["t_fv"])]
        n_sig = sum(1 for t in ts if abs(t) > 2)

        print(
            f"{vname:>16s}  "
            f"{np.mean(rho_fvs):>14.4f}  {np.mean(rho_fps):>14.4f}  "
            f"{np.mean(part_fvs):>15.4f}  {np.mean(dr2s):>+10.4f}  "
            f"{np.mean(ts):>+11.3f}  {n_sig:>6d}/{len(ts):<4d}"
        )

    print()
    print("판정:")
    print("  - mean|part_fv| > 0.05 → 조건부 정보 있음")
    print("  - mean t_Ze > 2 → 회귀 유의")
    print("  - E4가 가장 높으면 → 사용자 제안 정당")


if __name__ == "__main__":
    raise SystemExit(main())
