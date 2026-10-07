"""로그 비율/나눗셈 조합 테스트.

비교 대상 (forward target: fwd_vol, fwd_regime_persist):
  1. Z_r                              (baseline, Z_e 없음)
  2. Z_r - Z_e                        (현재 EPD)
  3. sign(Z_r) * log((1+|Z_r|)/(1+|Z_e|))   (signed log ratio)
  4. (Z_r - Z_e) / (|Z_r| + |Z_e| + 0.01)   (normalized diff)
  5. Z_r / (1 + |Z_e|)                (단순 나눗셈)
  6. Z_r * (1 - tanh(|Z_e|))          (Z_e로 attenuate)
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
    ("^GSPC", "SPX"),
    ("^KS11", "KOSPI"),
    ("005930.KS", "삼성전자"),
]


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    xr = np.argsort(np.argsort(x[m]))
    yr = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(xr, yr)[0, 1])


def forward_vol(r, h=5):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.std(ddof=1)
    return fv


def forward_regime_persist(r, h=5):
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


def formulas(z_r, z_e):
    eps = 0.01
    out = {}
    out["Z_r"] = z_r
    out["Z_r - Z_e"] = z_r - z_e
    out["sign*log_ratio"] = np.sign(z_r) * np.log(
        (1 + np.abs(z_r)) / (1 + np.abs(z_e) + eps)
    )
    out["norm_diff"] = (z_r - z_e) / (np.abs(z_r) + np.abs(z_e) + eps)
    out["Z_r / (1+|Z_e|)"] = z_r / (1 + np.abs(z_e))
    out["Z_r * (1-tanh|Ze|)"] = z_r * (1 - np.tanh(np.abs(z_e)))
    return out


def main():
    import yfinance as yf

    targets = {"fwd_vol": forward_vol, "fwd_persist": forward_regime_persist}
    H = 5

    print("=" * 120)
    print("로그 비율/나눗셈 조합 테스트 (H=5)")
    print("=" * 120)

    all_results = {}

    for tkr, name in TICKERS:
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

        print(f"\n### {name}")

        for tname, fn in targets.items():
            tgt = fn(r, H)
            print(f"\n  Target: {tname}")
            print(f"  {'formula':>22s}  {'|rho|':>8s}  {'rho':>8s}")
            print("  " + "-" * 42)
            for fname, fval in formulas(z_r, z_e).items():
                rho = spearman(fval, tgt)
                print(f"  {fname:>22s}  {abs(rho):>8.4f}  {rho:>+8.4f}")
                all_results.setdefault((name, tname, fname), []).append(rho)

    # 요약
    print()
    print("=" * 120)
    print("요약 (3자산 평균 |rho|)")
    print("=" * 120)
    print(f"{'formula':>22s}  {'fwd_vol':>10s}  {'fwd_persist':>12s}")
    print("-" * 60)

    fnames = list(formulas(np.array([0.0]), np.array([0.0])).keys())
    for fname in fnames:
        vals_v = [
            abs(v)
            for k, vs in all_results.items()
            if k[1] == "fwd_vol" and k[2] == fname
            for v in vs
        ]
        vals_p = [
            abs(v)
            for k, vs in all_results.items()
            if k[1] == "fwd_persist" and k[2] == fname
            for v in vs
        ]
        print(f"{fname:>22s}  " f"{np.mean(vals_v):>10.4f}  {np.mean(vals_p):>12.4f}")

    print()
    print("판정: Z_r 대비 개선이 있는가?")
    print("  - 개선 없음 → Z_e는 노이즈, 어떤 조합도 무의미")
    print("  - 개선 있음 → 특정 비선형 조합이 상호작용을 잡음")


if __name__ == "__main__":
    raise SystemExit(main())
