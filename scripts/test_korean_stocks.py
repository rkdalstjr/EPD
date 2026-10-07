"""한국 개별 종목 + 지수 + 미국 종목 비교 (v1.1, v1.2, v1.3)."""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import rolling_pe, rolling_zscore  # noqa: E402

warnings.filterwarnings("ignore")

W_PE, M, TAU, W_Z = 60, 3, 1, 252
TANH_SCALE = 2.0
H = 5


def compute_variant(close, kind):
    close = np.asarray(close, dtype=float)
    n = close.size
    log_P = np.log(close)
    r = np.full(n, np.nan)
    r[1:] = np.diff(log_P)

    pe = rolling_pe(r, w_pe=W_PE, m=M, tau=TAU)
    dpe = np.full(n, np.nan)
    dpe[1:] = pe[1:] - pe[:-1]

    z_r = rolling_zscore(r, W_Z)
    z_p = rolling_zscore(log_P, W_Z)
    z_e = rolling_zscore(dpe, W_Z)

    if kind == "r":
        base = z_r
    elif kind == "p":
        base = z_p
    elif kind == "rp":
        base = (z_r + z_p) / np.sqrt(2.0)

    raw = base - z_e
    return 50.0 * (1.0 + np.tanh(raw / TANH_SCALE))


def forward_return(r, h=H):
    fr = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fr[t] = r[t + 1 : t + 1 + h].sum()
    return fr


def spearman_ic(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    xr = np.argsort(np.argsort(x[m]))
    yr = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(xr, yr)[0, 1])


def quintile_rho(epd, fwd, n_q=5):
    m = np.isfinite(epd) & np.isfinite(fwd)
    if m.sum() < n_q * 30:
        return np.nan, None
    e, f = epd[m], fwd[m]
    qs = np.quantile(e, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    means = []
    for i in range(n_q):
        mm = (e >= qs[i]) & (e < qs[i + 1])
        if mm.sum() < 20:
            return np.nan, None
        means.append(f[mm].mean())
    means = np.array(means)
    rq = np.arange(1, n_q + 1)
    rm = np.argsort(np.argsort(means)) + 1
    return float(np.corrcoef(rq, rm)[0, 1]), means


def main():
    import yfinance as yf

    tickers = [
        # 한국 개별 종목
        ("005930.KS", "삼성전자"),
        ("000660.KS", "SK하이닉스"),
        ("005380.KS", "현대차"),
        ("035420.KS", "NAVER"),
        ("051910.KS", "LG화학"),
        ("005490.KS", "POSCO홀딩스"),
        ("068270.KS", "셀트리온"),
        ("373220.KS", "LG에너지솔루션"),
        # 한국 지수
        ("^KS11", "KOSPI지수"),
        ("^KQ11", "KOSDAQ지수"),
        # 미국 비교
        ("^GSPC", "SPX"),
        ("^IXIC", "NASDAQ"),
    ]

    print("=" * 110)
    print(f"EPD variant 비교 — 한국 개별 종목 + 지수 (H={H}, w_pe={W_PE}, w_z={W_Z})")
    print("=" * 110)
    print(
        f"{'asset':>14s}  {'n':>5s}  | "
        f"{'v1.1 r':>22s}  | {'v1.2 p':>22s}  | {'v1.3 rp':>22s}"
    )
    print(
        f"{'':>14s}  {'':>5s}  | "
        f"{'IC':>7s} {'rho':>6s} {'|rho|':>6s}  | "
        f"{'IC':>7s} {'rho':>6s} {'|rho|':>6s}  | "
        f"{'IC':>7s} {'rho':>6s} {'|rho|':>6s}"
    )
    print("-" * 110)

    rows = []
    for tkr, name in tickers:
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
                print(f"{name:>14s}  (skip, n={0 if df is None else len(df)})")
                continue
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            if close.size < 500:
                continue
            r = np.full_like(close, np.nan)
            r[1:] = np.diff(np.log(close))
            fwd = forward_return(r, H)
        except Exception as e:
            print(f"{name:>14s}  (err: {e})")
            continue

        row = {"name": name, "n": len(close)}
        for kind in ["r", "p", "rp"]:
            epd = compute_variant(close, kind)
            ic = spearman_ic(epd, fwd)
            rho, _ = quintile_rho(epd, fwd)
            row[f"{kind}_ic"] = ic
            row[f"{kind}_rho"] = rho
        rows.append(row)

        def fmt(kind):
            ic = row[f"{kind}_ic"]
            rho = row[f"{kind}_rho"]
            return (
                f"{ic:>7.4f} {rho:>6.2f} {abs(rho):>6.2f}"
                if np.isfinite(ic) and np.isfinite(rho)
                else " " * 22
            )

        print(
            f"{name:>14s}  {len(close):>5d}  | {fmt('r')}  | {fmt('p')}  | {fmt('rp')}"
        )

    # 요약
    print("\n" + "=" * 110)
    print("자산군별 평균 |rho|")
    print("=" * 110)

    groups = {
        "한국 개별 종목": [
            "삼성전자",
            "SK하이닉스",
            "현대차",
            "NAVER",
            "LG화학",
            "POSCO홀딩스",
            "셀트리온",
            "LG에너지솔루션",
        ],
        "한국 지수": ["KOSPI지수", "KOSDAQ지수"],
        "미국 지수": ["SPX", "NASDAQ"],
    }
    print(
        f"{'group':>16s}  {'n':>3s}  | {'v1.1 |rho|':>12s}  | "
        f"{'v1.2 |rho|':>12s}  | {'v1.3 |rho|':>12s}"
    )
    for gname, names in groups.items():
        sub = [r for r in rows if r["name"] in names]
        if not sub:
            continue
        means = {}
        for kind in ["r", "p", "rp"]:
            vals = [abs(r[f"{kind}_rho"]) for r in sub if np.isfinite(r[f"{kind}_rho"])]
            means[kind] = np.mean(vals) if vals else np.nan
        print(
            f"{gname:>16s}  {len(sub):>3d}  | "
            f"{means['r']:>12.3f}  | {means['p']:>12.3f}  | {means['rp']:>12.3f}"
        )

    print("\n기준: |rho| >= 0.6 = 강한 단조 관계")


if __name__ == "__main__":
    raise SystemExit(main())
