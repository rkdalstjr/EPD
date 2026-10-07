"""한국 종목 부호 뒤섞임 원인 진단.

1) quintile means 직접 확인 (부호가 실제로 반대인지, 노이즈인지)
2) 기간별 rho (시대 의존성 확인)
3) 상승/하락 국면별 rho (regime 의존성 확인)
"""

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


def compute_v1_1(close):
    close = np.asarray(close, dtype=float)
    n = close.size
    log_P = np.log(close)
    r = np.full(n, np.nan)
    r[1:] = np.diff(log_P)
    pe = rolling_pe(r, w_pe=W_PE, m=M, tau=TAU)
    dpe = np.full(n, np.nan)
    dpe[1:] = pe[1:] - pe[:-1]
    z_r = rolling_zscore(r, W_Z)
    z_e = rolling_zscore(dpe, W_Z)
    raw = z_r - z_e
    return 50.0 * (1.0 + np.tanh(raw / TANH_SCALE)), r


def forward_return(r, h=H):
    fr = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fr[t] = r[t + 1 : t + 1 + h].sum()
    return fr


def quintile_means(epd, fwd, n_q=5):
    m = np.isfinite(epd) & np.isfinite(fwd)
    if m.sum() < n_q * 30:
        return None
    e, f = epd[m], fwd[m]
    qs = np.quantile(e, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    means = []
    for i in range(n_q):
        mm = (e >= qs[i]) & (e < qs[i + 1])
        if mm.sum() < 20:
            return None
        means.append(f[mm].mean())
    return np.array(means)


def rho_from_means(means):
    rq = np.arange(1, len(means) + 1)
    rm = np.argsort(np.argsort(means)) + 1
    return float(np.corrcoef(rq, rm)[0, 1])


def main():
    import yfinance as yf

    tickers = [
        ("005930.KS", "삼성전자"),
        ("000660.KS", "SK하이닉스"),
        ("005380.KS", "현대차"),
        ("035420.KS", "NAVER"),
        ("051910.KS", "LG화학"),
        ("005490.KS", "POSCO홀딩스"),
        ("068270.KS", "셀트리온"),
        ("373220.KS", "LG에너지솔루션"),
    ]

    periods = [
        ("2005-2012", "2005-01-01", "2012-12-31"),
        ("2013-2019", "2013-01-01", "2019-12-31"),
        ("2020-2025", "2020-01-01", "2025-12-31"),
    ]

    print("=" * 110)
    print("한국 종목 진단 — 부호 뒤섞임 원인")
    print("=" * 110)

    print("\n### A. 전체 기간 quintile means (Q1..Q5)\n")
    print(
        f"{'name':>14s}  {'Q1':>9s} {'Q2':>9s} {'Q3':>9s} {'Q4':>9s} {'Q5':>9s}  {'rho':>6s}"
    )
    print("-" * 90)

    all_data = {}
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
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            epd, r = compute_v1_1(close)
            fwd = forward_return(r, H)
            all_data[name] = (close, epd, r, fwd)
            q = quintile_means(epd, fwd)
            if q is None:
                print(f"{name:>14s}  (insufficient)")
                continue
            rho = rho_from_means(q)
            print(
                f"{name:>14s}  " + " ".join(f"{v:>+9.5f}" for v in q) + f"  {rho:>6.2f}"
            )
        except Exception as e:
            print(f"{name:>14s}  err: {e}")

    print("\n### B. 기간별 rho\n")
    print(
        f"{'name':>14s}  | "
        f"{'2005-2012':>12s}  | {'2013-2019':>12s}  | {'2020-2025':>12s}"
    )
    print("-" * 70)
    for tkr, name in tickers:
        row = []
        for pname, start, end in periods:
            try:
                df = yf.download(
                    tkr,
                    start=start,
                    end=end,
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
                close = df["Close"].squeeze().to_numpy(dtype=float)
                close = close[np.isfinite(close)]
                if close.size < 400:
                    row.append("     n/a")
                    continue
                epd, r = compute_v1_1(close)
                fwd = forward_return(r, H)
                q = quintile_means(epd, fwd)
                if q is None:
                    row.append("     n/a")
                else:
                    row.append(f"{rho_from_means(q):>+12.2f}")
            except Exception:
                row.append("     err")
        print(f"{name:>14s}  | " + "  | ".join(row))

    print("\n### C. 상승/하락 국면별 rho\n")
    print("국면 정의: 지난 60일 누적 수익 부호")
    print(f"{'name':>14s}  | {'상승기 rho':>12s}  | {'하락기 rho':>12s}")
    print("-" * 50)
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
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            epd, r = compute_v1_1(close)
            fwd = forward_return(r, H)
            log_P = np.log(close)
            # 지난 60일 수익
            cum60 = np.full(len(close), np.nan)
            for t in range(60, len(close)):
                cum60[t] = log_P[t] - log_P[t - 60]

            up_mask = cum60 > 0.02
            dn_mask = cum60 < -0.02

            def q_rho(mask):
                e2 = np.where(mask, epd, np.nan)
                q = quintile_means(e2, fwd)
                return rho_from_means(q) if q is not None else np.nan

            up_rho = q_rho(up_mask)
            dn_rho = q_rho(dn_mask)
            print(f"{name:>14s}  | {up_rho:>+12.2f}  | {dn_rho:>+12.2f}")
        except Exception:
            print(f"{name:>14s}  | (err)")

    print("\n해석 가이드:")
    print("  - A: quintile means가 단조 감소/증가면 신호 있음. 뒤섞이면 노이즈.")
    print("  - B: 기간별 rho 부호가 다르면 시대 의존적.")
    print(
        "  - C: 상승기/하락기 rho 부호가 다르면 regime 의존적 → 이게 부호 뒤섞임의 원인."
    )


if __name__ == "__main__":
    raise SystemExit(main())
