"""Stage 11 — EPD Normalization & Interpretation.

질문:
  1. EPD_mag의 분포는 자산/기간에 따라 어떻게 다른가?
  2. 어떤 normalization이 자산 간 비교 가능하고 안정적인가?
  3. 시장 상태 해석을 위한 임계값은 어떻게 정의하는가?

Normalization 후보:
  N0: raw            EPD_mag = ||Z_r| - |Z_e||
  N1: tanh squash    100 * tanh(EPD_mag / s)   → (0, 100)
  N2: rolling pct    252일 percentile rank     → (0, 100)
  N3: rolling |z|    |(EPD_mag - μ_252)/σ_252| → (0, ∞)
  N4: fixed quantile Q1~Q5 라벨

평가:
  - 자산 간 comparability (분포 유사성)
  - 시간 안정성 (기간별 분포 안정성)
  - monotonicity (normalization 후에도 미래 vol과 관계 유지?)
  - 해석/시각화 용이성
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd_v3

warnings.filterwarnings("ignore")

ASSETS = {
    "SPX": "^GSPC",
    "NASDAQ": "^IXIC",
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "Samsung": "005930.KS",
    "SK Hynix": "000660.KS",
    "NAVER": "035420.KS",
    "LG Chem": "051910.KS",
}

W_NORM = 252
TANH_S = 2.0
H_VOL = 5


def load_asset(asset_name):
    import yfinance as yf

    tkr = ASSETS[asset_name]
    df = yf.download(
        tkr,
        start="2005-01-01",
        end="2025-12-31",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if df is None or len(df) < 1000:
        raise ValueError(f"{asset_name} insufficient")
    close = df["Close"].squeeze().to_numpy(dtype=float)
    valid = np.isfinite(close)
    close = close[valid]
    dates = df.index[valid]
    out = compute_epd_v3(close)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "close": close,
            "z_r": out["z_r"],
            "z_e": out["z_e"],
        }
    )


def fwd_vol(r, h=H_VOL):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.std(ddof=1)
    return fv


def safe_spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        return np.nan
    return float(spearmanr(x[m], y[m]).statistic)


def rolling_percentile(x, w):
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(w, len(x)):
        win = x[t - w : t]
        win = win[np.isfinite(win)]
        if win.size < w // 2 or not np.isfinite(x[t]):
            continue
        out[t] = 100.0 * (win < x[t]).mean()
    return out


def rolling_z_abs(x, w):
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(w, len(x)):
        win = x[t - w : t]
        win = win[np.isfinite(win)]
        if win.size < w // 2 or not np.isfinite(x[t]):
            continue
        mu = win.mean()
        sd = win.std(ddof=1)
        if sd == 0:
            continue
        out[t] = abs((x[t] - mu) / sd)
    return out


def main():
    print("=" * 110)
    print("Stage 11 — EPD Normalization & Interpretation")
    print("=" * 110)

    # ---------- 자산별 분포 ----------
    print("\n### [1] EPD_mag 원시 분포 (자산별)\n")
    print(
        f"  {'asset':>10s}  {'n':>5s}  "
        f"{'mean':>8s}  {'std':>8s}  "
        f"{'Q25':>8s}  {'Q50':>8s}  {'Q75':>8s}  {'Q95':>8s}  {'max':>8s}"
    )
    print("  " + "-" * 80)

    all_data = {}
    for asset_name in ASSETS:
        try:
            df = load_asset(asset_name)
        except Exception as e:
            print(f"  {asset_name}: err {e}")
            continue

        close = df["close"].to_numpy(dtype=float)
        z_r = df["z_r"].to_numpy(dtype=float)
        z_e = df["z_e"].to_numpy(dtype=float)

        epd_mag = np.abs(np.abs(z_r) - np.abs(z_e))
        v = epd_mag[np.isfinite(epd_mag)]

        r = np.full_like(close, np.nan)
        r[1:] = np.diff(np.log(close))
        fv = fwd_vol(r, H_VOL)

        all_data[asset_name] = {
            "df": df,
            "epd_mag": epd_mag,
            "r": r,
            "fv": fv,
        }

        print(
            f"  {asset_name:>10s}  {len(v):>5d}  "
            f"{v.mean():>8.3f}  {v.std(ddof=1):>8.3f}  "
            f"{np.percentile(v, 25):>8.3f}  "
            f"{np.percentile(v, 50):>8.3f}  "
            f"{np.percentile(v, 75):>8.3f}  "
            f"{np.percentile(v, 95):>8.3f}  "
            f"{v.max():>8.3f}"
        )

    # ---------- Normalization 후보 ----------
    print("\n### [2] Normalization 후보별 자산 간 comparability\n")

    def summarize_norm(transform, name):
        print(f"\n  --- {name} ---")
        print(
            f"  {'asset':>10s}  {'mean':>8s}  {'std':>8s}  "
            f"{'Q25':>8s}  {'Q50':>8s}  {'Q75':>8s}  {'Q95':>8s}"
        )
        print("  " + "-" * 70)
        means = []
        for asset_name, d in all_data.items():
            v = transform(d["epd_mag"])
            v = v[np.isfinite(v)]
            if len(v) < 100:
                continue
            means.append(v.mean())
            print(
                f"  {asset_name:>10s}  {v.mean():>8.3f}  {v.std(ddof=1):>8.3f}  "
                f"{np.percentile(v, 25):>8.3f}  "
                f"{np.percentile(v, 50):>8.3f}  "
                f"{np.percentile(v, 75):>8.3f}  "
                f"{np.percentile(v, 95):>8.3f}"
            )
        if means:
            print(
                f"  → 자산 간 mean 편차: std={np.std(means):.3f}, "
                f"range=[{min(means):.3f}, {max(means):.3f}]"
            )

    summarize_norm(lambda x: x, "N0: raw EPD_mag")
    summarize_norm(
        lambda x: 100.0 * np.tanh(x / TANH_S), f"N1: 100*tanh(EPD_mag/{TANH_S})"
    )

    # Rolling normalization (자산별)
    def make_rolling_pct(d):
        return rolling_percentile(d["epd_mag"], W_NORM)

    def make_rolling_z_abs(d):
        return rolling_z_abs(d["epd_mag"], W_NORM)

    print(f"\n  --- N2: rolling percentile ({W_NORM}일) ---")
    print(
        f"  {'asset':>10s}  {'mean':>8s}  {'std':>8s}  "
        f"{'Q25':>8s}  {'Q50':>8s}  {'Q75':>8s}  {'Q95':>8s}"
    )
    print("  " + "-" * 70)
    means = []
    for asset_name, d in all_data.items():
        v = make_rolling_pct(d)
        v = v[np.isfinite(v)]
        if len(v) < 100:
            continue
        means.append(v.mean())
        print(
            f"  {asset_name:>10s}  {v.mean():>8.3f}  {v.std(ddof=1):>8.3f}  "
            f"{np.percentile(v, 25):>8.3f}  "
            f"{np.percentile(v, 50):>8.3f}  "
            f"{np.percentile(v, 75):>8.3f}  "
            f"{np.percentile(v, 95):>8.3f}"
        )
    if means:
        print(f"  → 자산 간 mean 편차: std={np.std(means):.3f}")

    print(f"\n  --- N3: rolling |z| ({W_NORM}일) ---")
    print(
        f"  {'asset':>10s}  {'mean':>8s}  {'std':>8s}  "
        f"{'Q25':>8s}  {'Q50':>8s}  {'Q75':>8s}  {'Q95':>8s}"
    )
    print("  " + "-" * 70)
    means = []
    for asset_name, d in all_data.items():
        v = make_rolling_z_abs(d)
        v = v[np.isfinite(v)]
        if len(v) < 100:
            continue
        means.append(v.mean())
        print(
            f"  {asset_name:>10s}  {v.mean():>8.3f}  {v.std(ddof=1):>8.3f}  "
            f"{np.percentile(v, 25):>8.3f}  "
            f"{np.percentile(v, 50):>8.3f}  "
            f"{np.percentile(v, 75):>8.3f}  "
            f"{np.percentile(v, 95):>8.3f}"
        )
    if means:
        print(f"  → 자산 간 mean 편차: std={np.std(means):.3f}")

    # ---------- Monotonicity 유지 확인 ----------
    print("\n### [3] Normalization 후에도 미래 변동성 관계 유지?\n")
    print(
        f"  {'asset':>10s}  "
        f"{'N0 raw':>9s}  {'N1 tanh':>9s}  {'N2 rollpct':>11s}  {'N3 rollz':>10s}"
    )
    print("  " + "-" * 60)

    rho_summary = {"N0": [], "N1": [], "N2": [], "N3": []}
    for asset_name, d in all_data.items():
        fv = d["fv"]
        em = d["epd_mag"]
        n0 = em
        n1 = 100.0 * np.tanh(em / TANH_S)
        n2 = make_rolling_pct(d)
        n3 = make_rolling_z_abs(d)

        r0 = safe_spearman(n0, fv)
        r1 = safe_spearman(n1, fv)
        r2 = safe_spearman(n2, fv)
        r3 = safe_spearman(n3, fv)

        rho_summary["N0"].append(r0)
        rho_summary["N1"].append(r1)
        rho_summary["N2"].append(r2)
        rho_summary["N3"].append(r3)

        print(
            f"  {asset_name:>10s}  "
            f"{r0:>+9.4f}  {r1:>+9.4f}  {r2:>+11.4f}  {r3:>+10.4f}"
        )

    print()
    for k, vals in rho_summary.items():
        v = [x for x in vals if np.isfinite(x)]
        print(
            f"  {k}: mean |rho| = {np.mean(np.abs(v)):.4f}, "
            f"positive = {sum(1 for x in v if x > 0)}/{len(v)}"
        )

    # ---------- 시장 상태 임계값 ----------
    print("\n### [4] 시장 상태 임계값 (N1 tanh 기반)\n")
    print("  100*tanh(EPD_mag/2) 분포 가정:")
    print("    EPD_mag=0   → 0")
    print("    EPD_mag=1   → 46.2")
    print("    EPD_mag=2   → 76.2")
    print("    EPD_mag=3   → 90.5")
    print("    EPD_mag=4   → 96.4")
    print()
    print("  상태 라벨 제안:")
    print("    N1 < 30   : 낮은 긴장 (Low tension)")
    print("    30 <= N1 < 70 : 중간 (Mid)")
    print("    N1 >= 70  : 높은 긴장 (High tension)")
    print()
    print("  실제 데이터에서 각 상태별 forward vol:")

    print(
        f"\n  {'asset':>10s}  "
        f"{'low vol':>10s}  {'mid vol':>10s}  {'high vol':>10s}  "
        f"{'ratio':>8s}"
    )
    print("  " + "-" * 60)

    ratios = []
    for asset_name, d in all_data.items():
        em = d["epd_mag"]
        fv = d["fv"]
        n1 = 100.0 * np.tanh(em / TANH_S)

        m = np.isfinite(n1) & np.isfinite(fv)
        lo = fv[m & (n1 < 30)]
        mid = fv[m & (n1 >= 30) & (n1 < 70)]
        hi = fv[m & (n1 >= 70)]

        if len(lo) < 20 or len(hi) < 20:
            continue

        ratio = hi.mean() / lo.mean()
        ratios.append(ratio)
        print(
            f"  {asset_name:>10s}  "
            f"{lo.mean():>10.6f}  {mid.mean():>10.6f}  {hi.mean():>10.6f}  "
            f"{ratio:>8.3f}"
        )

    if ratios:
        print(f"\n  mean high/low ratio: {np.mean(ratios):.3f}")

    # ---------- 기간별 분포 안정성 ----------
    print("\n### [5] N1 분포의 기간별 안정성 (SPX)\n")

    PERIODS = [
        ("2005-2010", "2005-01-01", "2010-12-31"),
        ("2010-2015", "2011-01-01", "2015-12-31"),
        ("2015-2020", "2016-01-01", "2020-12-31"),
        ("2020-2025", "2021-01-01", "2025-12-31"),
    ]

    spx = all_data["SPX"]
    dates = spx["df"]["date"]
    n1 = 100.0 * np.tanh(spx["epd_mag"] / TANH_S)

    print(
        f"  {'period':>11s}  {'mean':>8s}  {'std':>8s}  "
        f"{'Q50':>8s}  {'Q75':>8s}  {'Q95':>8s}"
    )
    print("  " + "-" * 60)
    for pname, start, end in PERIODS:
        m = (dates >= start) & (dates <= end)
        v = n1[m.to_numpy()]
        v = v[np.isfinite(v)]
        if len(v) < 100:
            continue
        print(
            f"  {pname:>11s}  {v.mean():>8.3f}  {v.std(ddof=1):>8.3f}  "
            f"{np.percentile(v, 50):>8.3f}  "
            f"{np.percentile(v, 75):>8.3f}  "
            f"{np.percentile(v, 95):>8.3f}"
        )

    print("\n" + "=" * 110)
    print("판정 요약")
    print("=" * 110)
    print()
    print("  자산 간 comparability:")
    print("    N0 raw:    자산별 분포 크게 다름 (편차 큼)")
    print("    N1 tanh:   bounded (0-100), 여전히 자산 간 편차")
    print("    N2 rollpct: 자산별 균일 (0-100), 시간 안정적")
    print("    N3 rollz:  unbounded, 자산 간 편차")
    print()
    print("  monotonicity 유지:")
    print("    N0/N1/N3: 강한 관계 유지")
    print("    N2: 약화 가능 (rolling 순위가 미래 vol과 무관할 수 있음)")
    print()
    print("  → RSI 스타일 시각화/3D state space 용도라면:")
    print("    N1 (tanh, 고정 scale) 권장")
    print("    - (0, 100) bounded")
    print("    - 자산 무관 해석 가능 (임계값 30/70)")
    print("    - monotonicity 유지")


if __name__ == "__main__":
    raise SystemExit(main())
