"""Z_r vs Z_e 수치 비교 — 분포, 꼬리, 상관.

목적:
  - 두 z-score의 실제 분포 확인
  - EPD_raw = Z_r - Z_e의 지배 성분 확인
  - 왜 Stage 4에서 Z_r이 dominant였는지 진단
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


def describe(x, name):
    """분포 요약."""
    x = x[np.isfinite(x)]
    if x.size < 100:
        return None
    qs = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    skew = float(((x - mean) ** 3).mean() / (std**3)) if std > 0 else np.nan
    kurt = float(((x - mean) ** 4).mean() / (std**4) - 3) if std > 0 else np.nan
    return {
        "name": name,
        "n": x.size,
        "mean": mean,
        "std": std,
        "min": float(x.min()),
        "max": float(x.max()),
        "q01": qs[0],
        "q05": qs[1],
        "q25": qs[2],
        "median": qs[3],
        "q75": qs[4],
        "q95": qs[5],
        "q99": qs[6],
        "skew": skew,
        "kurt": kurt,
        "frac|z|>1": float(np.mean(np.abs(x) > 1)),
        "frac|z|>2": float(np.mean(np.abs(x) > 2)),
        "frac|z|>3": float(np.mean(np.abs(x) > 3)),
    }


def print_stat(s, indent=2):
    pre = " " * indent
    print(f"{pre}--- {s['name']} (n={s['n']}) ---")
    print(
        f"{pre}  mean={s['mean']:+.4f}  std={s['std']:.4f}  "
        f"skew={s['skew']:+.3f}  kurt={s['kurt']:+.3f}"
    )
    print(f"{pre}  range: [{s['min']:+.3f}, {s['max']:+.3f}]")
    print(
        f"{pre}  Q01={s['q01']:+.3f}  Q05={s['q05']:+.3f}  Q25={s['q25']:+.3f}  "
        f"median={s['median']:+.3f}  Q75={s['q75']:+.3f}  Q95={s['q95']:+.3f}  Q99={s['q99']:+.3f}"
    )
    print(
        f"{pre}  frac|z|>1={s['frac|z|>1']:.1%}  "
        f"frac|z|>2={s['frac|z|>2']:.1%}  "
        f"frac|z|>3={s['frac|z|>3']:.1%}"
    )


def main():
    import yfinance as yf

    for tkr, name in TICKERS:
        print("=" * 100)
        print(f"### {name} ({tkr})")
        print("=" * 100)

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
                print(f"  skip\n")
                continue
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            if close.size < 500:
                continue
        except Exception as e:
            print(f"  err: {e}\n")
            continue

        out = compute_epd(close)
        z_r = out["return_z"]
        z_e = out["entropy_z"]
        epd_raw = out["epd_raw"]

        # --- Z_r, Z_e, EPD_raw 각각 요약 ---
        print("\n[1] 각 성분 분포\n")
        s_r = describe(z_r, "Z_r")
        s_e = describe(z_e, "Z_e")
        s_epd = describe(epd_raw, "EPD_raw (Z_r − Z_e)")
        for s in [s_r, s_e, s_epd]:
            print_stat(s)

        # --- 상관 ---
        m = np.isfinite(z_r) & np.isfinite(z_e)
        if m.sum() > 100:
            x, y = z_r[m], z_e[m]
            c_pearson = float(np.corrcoef(x, y)[0, 1])
            xr = np.argsort(np.argsort(x))
            yr = np.argsort(np.argsort(y))
            c_spear = float(np.corrcoef(xr, yr)[0, 1])
        else:
            c_pearson = c_spear = np.nan

        print(f"\n[2] 상관\n")
        print(f"  corr(Z_r, Z_e)  Pearson={c_pearson:+.4f}  Spearman={c_spear:+.4f}")

        # --- EPD_raw 분해: Z_r 기여 vs Z_e 기여 ---
        print(f"\n[3] EPD_raw = Z_r − Z_e 의 지배 성분 분석\n")
        m = np.isfinite(z_r) & np.isfinite(z_e) & np.isfinite(epd_raw)
        x, y, e = z_r[m], z_e[m], epd_raw[m]

        # 분산 분해 (Z_r, Z_e가 독립이면 각각 1, 총 2)
        var_zr = float(x.var(ddof=1))
        var_ze = float(y.var(ddof=1))
        var_epd = float(e.var(ddof=1))
        cov = float(np.cov(x, y, ddof=1)[0, 1])

        print(f"  var(Z_r)   = {var_zr:.4f}")
        print(f"  var(Z_e)   = {var_ze:.4f}")
        print(f"  cov(Z_r,Z_e) = {cov:+.4f}")
        print(f"  var(EPD_raw) = {var_epd:.4f}")
        print(
            f"  →  var(Z_r) + var(Z_e) − 2·cov = "
            f"{var_zr + var_ze - 2*cov:.4f}  (이론값)"
        )
        print(f"  →  var(Z_r)/var(EPD_raw) = {var_zr/var_epd:.3%}  (Z_r 기여)")
        print(f"  →  var(Z_e)/var(EPD_raw) = {var_ze/var_epd:.3%}  (Z_e 기여)")

        # --- 극단 발생 빈도 ---
        print(f"\n[4] 극단값 빈도 (|z|>2, 3 기준)\n")
        n_r_2 = int(np.sum(np.abs(x) > 2))
        n_e_2 = int(np.sum(np.abs(y) > 2))
        n_r_3 = int(np.sum(np.abs(x) > 3))
        n_e_3 = int(np.sum(np.abs(y) > 3))
        print(
            f"  |Z_r|>2: {n_r_2:5d} ({n_r_2/len(x):.2%})   "
            f"|Z_e|>2: {n_e_2:5d} ({n_e_2/len(y):.2%})"
        )
        print(
            f"  |Z_r|>3: {n_r_3:5d} ({n_r_3/len(x):.2%})   "
            f"|Z_e|>3: {n_e_3:5d} ({n_e_3/len(y):.2%})"
        )

        # --- 왜 Stage 4에서 Z_r dominant였나 ---
        print(f"\n[5] Stage 4 (4분면) 재해석\n")
        # Z_r, Z_e 각각 |z|>1인 상태에서 forward vol 계산
        from numpy import nan

        def forward_vol(r, h=5):
            fv = np.full_like(r, nan)
            for t in range(len(r) - h):
                win = r[t + 1 : t + 1 + h]
                win = win[np.isfinite(win)]
                if win.size >= 3:
                    fv[t] = win.std(ddof=1)
            return fv

        r = np.full_like(close, nan)
        r[1:] = np.diff(np.log(close))
        fv = forward_vol(r, 5)

        # Z_r 그룹 (부호별 forward vol)
        m_all = np.isfinite(z_r) & np.isfinite(fv)
        for sign, label in [(+1, "Z_r > +1"), (-1, "Z_r < -1")]:
            mm = m_all & (np.sign(z_r) == sign) & (np.abs(z_r) > 1)
            if mm.sum() > 20:
                print(f"  {label:12s}: n={mm.sum():5d}, fwd_vol={fv[mm].mean():.6f}")

        m_all = np.isfinite(z_e) & np.isfinite(fv)
        for sign, label in [(+1, "Z_e > +1"), (-1, "Z_e < -1")]:
            mm = m_all & (np.sign(z_e) == sign) & (np.abs(z_e) > 1)
            if mm.sum() > 20:
                print(f"  {label:12s}: n={mm.sum():5d}, fwd_vol={fv[mm].mean():.6f}")

        print()


if __name__ == "__main__":
    raise SystemExit(main())
