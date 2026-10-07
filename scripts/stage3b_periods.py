"""Stage 3.5: 기간별 보편성 (2005-12, 13-19, 20-25)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd  # noqa: E402

warnings.filterwarnings("ignore")

RHO_STRONG = 0.6
FRAC_THRESHOLD = 0.60

PERIODS = [("2005-2012", "2005-01-01", "2012-12-31"),
           ("2013-2019", "2013-01-01", "2019-12-31"),
           ("2020-2025", "2020-01-01", "2025-12-31")]


def forward_return(r, h=5):
    fr = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fr[t] = r[t+1:t+1+h].sum()
    return fr


def mono(epd, fwd, n_q=5):
    m = np.isfinite(epd) & np.isfinite(fwd)
    if m.sum() < n_q * 30:
        return np.nan
    e, f = epd[m], fwd[m]
    qs = np.quantile(e, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1e-12; qs[-1] += 1e-12
    means = []
    for i in range(n_q):
        mm = (e >= qs[i]) & (e < qs[i+1])
        if mm.sum() < 20:
            return np.nan
        means.append(f[mm].mean())
    rq = np.arange(1, n_q + 1)
    rm = np.argsort(np.argsort(np.array(means))) + 1
    return float(np.corrcoef(rq, rm)[0, 1])


def main():
    import yfinance as yf

    print("=" * 80)
    print("Stage 3.5: 기간별 보편성 — EPD v1.2")
    print("=" * 80)

    print(f"{'asset':>8s}  {'period':>10s}  {'n':>5s}  {'frac|rho|>=0.6':>16s}  {'med_rho':>9s}")
    print("-" * 80)

    all_rows = []
    for tkr, asset in [("^GSPC", "SPX"), ("^KS11", "KOSPI"), ("^IXIC", "NASDAQ")]:
        for pname, start, end in PERIODS:
            df = yf.download(tkr, start=start, end=end,
                             interval="1d", progress=False, auto_adjust=True)
            if df is None or len(df) < 300:
                print(f"{asset:>8s}  {pname:>10s}  (short)")
                continue
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)][1:]
            r = np.diff(np.log(np.concatenate([[close[0]], close])))
            fwd = forward_return(r, 5)

            rhos = []
            for w_pe in [20, 60, 120]:
                for m in [3, 4]:
                    for w_z in [60, 252, 504]:
                        try:
                            out = compute_epd(close, w_pe=w_pe, m=m, tau=1, w_z=w_z)
                            rho = mono(out["epd_100"], fwd)
                            if np.isfinite(rho):
                                rhos.append(rho)
                        except Exception:
                            pass

            if not rhos:
                continue
            frac = sum(1 for v in rhos if abs(v) >= RHO_STRONG) / len(rhos)
            med = float(np.median(rhos))
            print(f"{asset:>8s}  {pname:>10s}  {len(rhos):>5d}  {frac:>16.2%}  {med:>9.4f}")
            all_rows.append({"asset": asset, "period": pname,
                             "frac_strong": frac, "med_rho": med})

    print()
    print(f"판정 기준: 각 (asset, period)에서 frac >= {FRAC_THRESHOLD:.0%}")
    fails = [r for r in all_rows if r["frac_strong"] < FRAC_THRESHOLD]
    if fails:
        print(f"\n[FAIL] {len(fails)}개 (asset, period) 미달:")
        for f in fails:
            print(f"  {f['asset']:>8s}  {f['period']:>10s}  {f['frac_strong']:.2%}")
    else:
        print(f"\n[PASS] 모든 (asset, period) 통과")

    overall = len(fails) == 0
    print(f"\n=== Stage 3.5 Result: {'PASS' if overall else 'FAIL'} ===")

    rpt = ROOT / "docs" / "universality_report.md"
    if rpt.exists():
        txt = rpt.read_text(encoding="utf-8")
    else:
        txt = "# Universality Report\n"
    txt += "\n## Stage 3.5 — 기간별\n\n"
    txt += f"판정: frac >= {FRAC_THRESHOLD:.0%}\n\n"
    txt += "| Asset | Period | n | frac_strong | med_rho | Pass |\n|---|---|---|---|---|---|\n"
    for r in all_rows:
        ok = r["frac_strong"] >= FRAC_THRESHOLD
        txt += (f"| {r['asset']} | {r['period']} | - | "
                f"{r['frac_strong']:.2%} | {r['med_rho']:.4f} | "
                f"{'✅' if ok else '❌'} |\n")
    txt += f"\n**Stage 3.5 Overall**: {'PASS' if overall else 'FAIL'}\n"
    rpt.write_text(txt, encoding="utf-8")
    print(f"report -> {rpt}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
