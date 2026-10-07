"""EPD v3.1 — |EPD| 독립성 검증.

핵심 질문:
  1. |EPD|, |Z_r|, |Z_e_res| 중 뭐가 forward vol을 잘 예측?
  2. |EPD|가 |Z_r| 통제 후에도 추가 정보 있나?
  3. 4분면 (Z_r × Z_e_res 부호) 분석

Forward targets:
  fwd_vol, fwd_abs_ret, fwd_range, fwd_max_ret
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd_v3

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

H = 5


# ============================================================
#  Forward targets
# ============================================================
def forward_vol(r, h=H):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.std(ddof=1)
    return fv


def forward_abs_ret(r, h=H):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        fv[t] = np.nansum(np.abs(w))
    return fv


def forward_range(r, h=H):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.max() - w.min()
    return fv


def forward_max_abs(r, h=H):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = np.abs(r[t + 1 : t + 1 + h])
        w = w[np.isfinite(w)]
        if w.size >= 1:
            fv[t] = w.max()
    return fv


TARGETS = {
    "fwd_vol": forward_vol,
    "fwd_abs_ret": forward_abs_ret,
    "fwd_range": forward_range,
    "fwd_max_abs": forward_max_abs,
}


# ============================================================
#  통계
# ============================================================
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


# ============================================================
#  Main
# ============================================================
def main():
    import yfinance as yf

    print("=" * 130)
    print("EPD v3.1 — |EPD| 독립성 검증")
    print("=" * 130)

    # 자산별 결과 수집
    asset_results = {}

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
            if df is None or len(df) < 1000:
                continue
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            if close.size < 1000:
                continue
        except Exception as e:
            print(f"{name} err: {e}")
            continue

        r = np.full_like(close, np.nan)
        r[1:] = np.diff(np.log(close))

        out = compute_epd_v3(close)
        c1 = out["C1_raw"]
        z_r = out["z_r"]
        z_e = out["z_e"]

        abs_epd = np.abs(c1)
        abs_zr = np.abs(z_r)
        abs_ze = np.abs(z_e)

        print(f"\n### {name} ({grp})")
        print(f"  [1] |EPD| vs |Z_r| vs |Z_e_res| → forward targets\n")
        print(
            f"  {'target':>13s}  "
            f"{'|EPD|':>9s}  {'|Z_r|':>9s}  {'|Z_e_res|':>11s}  "
            f"{'partial(|EPD|,tgt| |Z_r|)':>28s}"
        )
        print("  " + "-" * 90)

        targets_data = {}
        for tname, fn in TARGETS.items():
            tgt = fn(r, H)
            rho_epd = spearman(abs_epd, tgt)
            rho_zr = spearman(abs_zr, tgt)
            rho_ze = spearman(abs_ze, tgt)
            pc = partial_corr(abs_epd, tgt, abs_zr)

            print(
                f"  {tname:>13s}  "
                f"{rho_epd:>+9.4f}  {rho_zr:>+9.4f}  {rho_ze:>+11.4f}  "
                f"{pc:>+28.4f}"
            )

            targets_data[tname] = {
                "rho_epd": rho_epd,
                "rho_zr": rho_zr,
                "rho_ze": rho_ze,
                "partial": pc,
            }

        # [2] 4분면 분석 (fwd_vol 기준)
        print(f"\n  [2] 4분면 분석 (Z_r 부호 × Z_e_res 부호) → fwd_vol\n")
        fv = forward_vol(r, H)
        m = np.isfinite(z_r) & np.isfinite(z_e) & np.isfinite(fv)

        # Z_r ± , Z_e ±
        q1 = m & (z_r > 0) & (z_e > 0)
        q2 = m & (z_r < 0) & (z_e > 0)
        q3 = m & (z_r < 0) & (z_e < 0)
        q4 = m & (z_r > 0) & (z_e < 0)

        quadrants = {
            "Q1(Zr+,Ze+)": q1,
            "Q2(Zr-,Ze+)": q2,
            "Q3(Zr-,Ze-)": q3,
            "Q4(Zr+,Ze-)": q4,
        }
        print(
            f"  {'quadrant':>14s}  {'n':>6s}  {'fwd_vol':>10s}  " f"{'|EPD| mean':>11s}"
        )
        print("  " + "-" * 55)
        quad_vols = []
        for qn, mask in quadrants.items():
            if mask.sum() < 20:
                continue
            vol_mean = fv[mask].mean()
            epd_mean = abs_epd[mask].mean()
            quad_vols.append(vol_mean)
            print(
                f"  {qn:>14s}  {mask.sum():>6d}  "
                f"{vol_mean:>10.6f}  {epd_mean:>11.4f}"
            )

        # 4분면 분산 (그룹 간 차이)
        if len(quad_vols) == 4:
            quad_spread = max(quad_vols) - min(quad_vols)
            print(f"  4분면 vol spread = {quad_spread:.6f}")

        asset_results[name] = {
            "grp": grp,
            "targets": targets_data,
            "quad_vols": quad_vols,
        }

    # ============================================================
    # 요약
    # ============================================================
    print("\n" + "=" * 130)
    print("요약 (8자산 평균)")
    print("=" * 130)

    print(f"\n  [A] |EPD| vs |Z_r| vs |Z_e_res|")
    print(
        f"  {'target':>13s}  "
        f"{'|EPD|':>9s}  {'|Z_r|':>9s}  {'|Z_e_res|':>11s}  "
        f"{'partial(|EPD|||Z_r|)':>21s}"
    )
    print("  " + "-" * 75)

    for tname in TARGETS.keys():
        vals = [a["targets"][tname] for a in asset_results.values()]
        rho_epd = np.mean(
            [abs(v["rho_epd"]) for v in vals if np.isfinite(v["rho_epd"])]
        )
        rho_zr = np.mean([abs(v["rho_zr"]) for v in vals if np.isfinite(v["rho_zr"])])
        rho_ze = np.mean([abs(v["rho_ze"]) for v in vals if np.isfinite(v["rho_ze"])])
        partial = np.mean([v["partial"] for v in vals if np.isfinite(v["partial"])])

        print(
            f"  {tname:>13s}  "
            f"{rho_epd:>9.4f}  {rho_zr:>9.4f}  {rho_ze:>11.4f}  "
            f"{partial:>+21.4f}"
        )

    # 판정
    print()
    print("=" * 130)
    print("판정")
    print("=" * 130)

    # fwd_vol 기준 partial
    partials = [
        a["targets"]["fwd_vol"]["partial"]
        for a in asset_results.values()
        if np.isfinite(a["targets"]["fwd_vol"]["partial"])
    ]

    mean_partial = np.mean(np.abs(partials)) if partials else np.nan
    n_sig = sum(1 for p in partials if abs(p) > 0.05)

    print(f"  fwd_vol 기준:")
    print(f"    mean |partial(|EPD|, fwd_vol | |Z_r|)| = {mean_partial:.4f}")
    print(f"    유의(|partial|>0.05) 자산: {n_sig}/{len(partials)}")

    if mean_partial > 0.05:
        print(f"\n  ✅ |EPD|는 |Z_r| 통제 후에도 유의한 추가 정보 있음")
        print(f"     → EPD magnitude는 독립 지표")
    elif mean_partial > 0.02:
        print(f"\n  ⚠️ |EPD|의 독립 정보 미약 (marginal)")
        print(f"     → 조건부 유효")
    else:
        print(f"\n  ❌ |EPD|는 |Z_r|의 재포장")
        print(f"     → EPD magnitude 폐기, |Z_r| 단독 지표로 전환")


if __name__ == "__main__":
    raise SystemExit(main())
