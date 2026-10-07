"""곱셈 EPD variants + E4-residual 결합 비교.

Z_e 종류:
  Z_e_dPE:  Z(dPE)                    (기존, E1)
  Z_e_res:  Z(PE − E[PE | |Z_r|])     (E4)

Candidates (Z_e_res 기반):
  C1: Z_r − Z_e_res                  (뺄셈, 참조선)
  C2: Z_r × Z_e_res                  (signed product)
  C3: Z_r × |Z_e_res|                (magnitude-gated)
  C4: sign(Z_r − Z_e_res) × |Z_r| × |Z_e_res|   (divergence direction)
  C5: Z_r                            (baseline)

참조 (dPE 기반):
  C1_E1: Z_r − Z_e_dPE               (원래 EPD)
  C3_E1: Z_r × |Z_e_dPE|

측정 (동일 조건):
  - corr(C, fwd_vol_5d)
  - corr(C, fwd_persist_5d)
  - partial(C | Z_r)                 ← Z_r 통제 후 추가 정보
  - ΔR² (M1: Z_r, M2: Z_r + C)
  - t(C) in M2
  - t(Z_r × C) interaction
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
RESID_WIN = 504
H = 5
EPS = 1e-9


# ============================================================
#  유틸
# ============================================================
def forward_vol(r, h=H):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.std(ddof=1)
    return fv


def forward_persist(r, h=H):
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


def ols_delta_r2(y, x1, x2, with_interaction=False):
    """M1: y ~ 1+x1, M2: y ~ 1+x1+x2 (+ interaction). Returns (ΔR², t_x2, t_int)."""
    m = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    if m.sum() < 100:
        return np.nan, np.nan, np.nan
    y, x1, x2 = y[m], x1[m], x2[m]
    n = len(y)
    c = np.ones(n)
    X1 = np.column_stack([c, x1])
    if with_interaction:
        X2 = np.column_stack([c, x1, x2, x1 * x2])
    else:
        X2 = np.column_stack([c, x1, x2])

    def fit(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        rss = (resid**2).sum()
        tss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - rss / tss if tss > 0 else np.nan
        # t stats
        df = n - X.shape[1]
        sigma2 = rss / df if df > 0 else np.nan
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.full(X.shape[1], np.nan)
        t = beta / se
        return r2, t

    r2_1, _ = fit(X1)
    r2_2, t2 = fit(X2)
    t_x2 = t2[2]
    t_int = t2[3] if with_interaction else np.nan
    return r2_2 - r2_1, t_x2, t_int


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


# ============================================================
#  Z_e 계산 — dPE vs residual
# ============================================================
def compute_ze_dpe(pe):
    n = len(pe)
    dpe = np.full(n, np.nan)
    dpe[1:] = pe[1:] - pe[:-1]
    return rolling_zscore(dpe, w_z=W_Z)


def compute_ze_residual(pe, z_r):
    """E4: PE − E[PE | |Z_r|]."""
    n = len(pe)
    abs_zr = np.abs(z_r)
    resid = np.full(n, np.nan)
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
        qs = np.quantile(w_z, np.linspace(0, 1, 6))
        qs[0] -= EPS
        qs[-1] += EPS
        b = int(np.searchsorted(qs, abs_zr[t], side="right") - 1)
        b = max(0, min(4, b))
        bin_mask = (w_z >= qs[b]) & (w_z < qs[b + 1])
        if bin_mask.sum() < 5:
            continue
        expected = w_p[bin_mask].mean()
        resid[t] = pe[t] - expected
    return rolling_zscore(resid, w_z=W_Z)


# ============================================================
#  Candidates
# ============================================================
def build_candidates(z_r, ze_dpe, ze_res):
    c = {}
    # 참조 (dPE 기반)
    c["C1_E1_dPE_sub"] = z_r - ze_dpe
    c["C3_E1_dPE_mul"] = z_r * np.abs(ze_dpe)

    # E4-residual 기반
    c["C1_res_sub"] = z_r - ze_res
    c["C2_res_signed_mul"] = z_r * ze_res
    c["C3_res_gated"] = z_r * np.abs(ze_res)
    c["C4_res_div_dir"] = np.sign(z_r - ze_res) * np.abs(z_r) * np.abs(ze_res)

    # Baseline
    c["C5_Zr_only"] = z_r
    c["C0_Ze_res_only"] = ze_res
    return c


# ============================================================
#  Main
# ============================================================
def main():
    import yfinance as yf

    print("=" * 140)
    print("곱셈 EPD variants + E4-residual 비교")
    print("=" * 140)

    results = {}  # (asset, cname) -> dict

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

        ze_dpe = compute_ze_dpe(pe)
        ze_res = compute_ze_residual(pe, z_r)

        fv = forward_vol(r, H)
        fp = forward_persist(r, H)

        cands = build_candidates(z_r, ze_dpe, ze_res)

        print(f"\n### {name} ({grp})")
        print(
            f"  {'candidate':>20s}  "
            f"{'rho(fv)':>9s}  {'rho(fp)':>9s}  "
            f"{'part(fv)':>10s}  {'ΔR2(fv)':>9s}  "
            f"{'t(C)':>8s}  {'t(int)':>8s}"
        )
        print("  " + "-" * 95)

        for cname, cval in cands.items():
            rho_fv = spearman(cval, fv)
            rho_fp = spearman(cval, fp)
            part = partial_corr(cval, fv, z_r)
            dr2, t_c, t_int = ols_delta_r2(fv, z_r, cval, with_interaction=True)

            print(
                f"  {cname:>20s}  "
                f"{rho_fv:>+9.4f}  {rho_fp:>+9.4f}  "
                f"{part:>+10.4f}  {dr2:>+9.4f}  "
                f"{t_c:>+8.3f}  {t_int:>+8.3f}"
            )

            results[(name, cname)] = {
                "rho_fv": rho_fv,
                "rho_fp": rho_fp,
                "partial": part,
                "dr2": dr2,
                "t_c": t_c,
                "t_int": t_int,
            }

    # ----- 요약 -----
    print()
    print("=" * 140)
    print("자산 평균 (8자산)")
    print("=" * 140)
    print(
        f"{'candidate':>20s}  "
        f"{'mean|rho_fv|':>14s}  {'mean|rho_fp|':>14s}  "
        f"{'mean|part|':>12s}  {'mean ΔR2':>11s}  "
        f"{'mean|t_C|':>11s}  {'mean|t_int|':>13s}"
    )
    print("-" * 120)

    cnames = list(
        build_candidates(np.array([0.0]), np.array([0.0]), np.array([0.0])).keys()
    )

    for cname in cnames:
        vals = [v for (a, cn), v in results.items() if cn == cname]
        if not vals:
            continue
        rho_fvs = [abs(v["rho_fv"]) for v in vals if np.isfinite(v["rho_fv"])]
        rho_fps = [abs(v["rho_fp"]) for v in vals if np.isfinite(v["rho_fp"])]
        parts = [abs(v["partial"]) for v in vals if np.isfinite(v["partial"])]
        dr2s = [v["dr2"] for v in vals if np.isfinite(v["dr2"])]
        t_cs = [abs(v["t_c"]) for v in vals if np.isfinite(v["t_c"])]
        t_ints = [abs(v["t_int"]) for v in vals if np.isfinite(v["t_int"])]

        print(
            f"{cname:>20s}  "
            f"{np.mean(rho_fvs):>14.4f}  {np.mean(rho_fps):>14.4f}  "
            f"{np.mean(parts):>12.4f}  {np.mean(dr2s):>+11.4f}  "
            f"{np.mean(t_cs):>11.3f}  {np.mean(t_ints):>13.3f}"
        )

    print()
    print("판정:")
    print("  - Z_r_only 대비 partial이 유의하게 높으면 → 새 정보 있음")
    print("  - ΔR²가 Z_r_only보다 크면 → 회귀 개선")
    print("  - t_int > 2 → Z_r과 상호작용 있음")


if __name__ == "__main__":
    raise SystemExit(main())
