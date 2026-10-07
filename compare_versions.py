"""v1.1 (Z_r) vs v1.2 (Z_p) vs v1.3 (combined) 전체 기간 비교."""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd, rolling_pe, rolling_zscore  # noqa: E402

warnings.filterwarnings("ignore")

H = 5  # forward horizon
TANH_SCALE = 2.0
W_PE, M, TAU, W_Z = 60, 3, 1, 252


def forward_return(r, h=H):
    fr = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fr[t] = r[t + 1 : t + 1 + h].sum()
    return fr


def causal_z(x, w):
    return rolling_zscore(x, w_z=w)


def compute_variant(close, kind):
    """kind: 'r' (v1.1), 'p' (v1.2), 'rp' (v1.3)."""
    close = np.asarray(close, dtype=float)
    n = close.size
    log_P = np.log(close)
    r = np.full(n, np.nan)
    r[1:] = np.diff(log_P)

    pe = rolling_pe(r, w_pe=W_PE, m=M, tau=TAU)
    dpe = np.full(n, np.nan)
    dpe[1:] = pe[1:] - pe[:-1]

    z_r = causal_z(r, W_Z)
    z_p = causal_z(log_P, W_Z)
    z_e = causal_z(dpe, W_Z)

    if kind == "r":
        base = z_r
    elif kind == "p":
        base = z_p
    elif kind == "rp":
        base = (z_r + z_p) / np.sqrt(2.0)
    else:
        raise ValueError(kind)

    raw = base - z_e
    epd_100 = 50.0 * (1.0 + np.tanh(raw / TANH_SCALE))
    return epd_100


def spearman_ic(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    xr = np.argsort(np.argsort(x[m]))
    yr = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(xr, yr)[0, 1])


def quintile_returns(epd, fwd, n_q=5):
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


def long_short_sharpe(epd, fwd, n_q=5):
    """Q1 - Q5 portfolio. 매일 리밸런싱. 연율화."""
    m = np.isfinite(epd) & np.isfinite(fwd)
    if m.sum() < n_q * 30:
        return np.nan
    e, f = epd[m], fwd[m]
    qs = np.quantile(e, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12

    # 각 시점에서 (Q1 속하면 +1, Q5 속하면 -1) → 그날의 fwd return
    # (5일 fwd를 매일 overlapping으로 사용; Sharpe는 sqrt(252/5) 스케일)
    q1_mask = e <= qs[1]
    q5_mask = e >= qs[-2]

    q1_rets = f[q1_mask]
    q5_rets = f[q5_mask]

    if q1_rets.size < 50 or q5_rets.size < 50:
        return np.nan

    ls = q1_rets.mean() - q5_rets.mean()  # 5일 누적 기준
    ls_daily = ls / H
    # 대략적 std (개별 5일 fwd의 std / sqrt(H))
    std_5d = np.std(np.concatenate([q1_rets, q5_rets]))
    std_daily = std_5d / np.sqrt(H)
    if std_daily <= 0:
        return np.nan
    return float(ls_daily / std_daily * np.sqrt(252))


def evaluate(close, fwd, label):
    out = {}
    for kind in ["r", "p", "rp"]:
        epd = compute_variant(close, kind)
        ic = spearman_ic(epd, fwd)
        q = quintile_returns(epd, fwd)
        sh = long_short_sharpe(epd, fwd)
        rho = np.nan
        if q is not None:
            rq = np.arange(1, 6)
            rm = np.argsort(np.argsort(q)) + 1
            rho = float(np.corrcoef(rq, rm)[0, 1])
        out[kind] = {"ic": ic, "rho": rho, "sharpe": sh, "quintiles": q}
    return out


def main():
    import yfinance as yf

    print("=" * 90)
    print("EPD v1.1 (Z_r) vs v1.2 (Z_p) vs v1.3 (combined) — 전체 기간")
    print(f"H={H}, w_pe={W_PE}, m={M}, w_z={W_Z}, tanh_scale={TANH_SCALE}")
    print("=" * 90)

    assets = [("^GSPC", "SPX"), ("^KS11", "KOSPI"), ("^IXIC", "NASDAQ")]
    data = {}
    for tkr, name in assets:
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
        r = np.full_like(close, np.nan)
        r[1:] = np.diff(np.log(close))
        fwd = forward_return(r, H)
        data[name] = (close, fwd, r)
        print(f"[data] {name}: n={len(close)}")

    # 자산별 결과
    for name, (close, fwd, r) in data.items():
        print(f"\n--- {name} ---")
        res = evaluate(close, fwd, name)
        print(
            f"{'variant':>8s}  {'IC':>8s}  {'rho':>7s}  {'Sharpe(LS)':>11s}  quintiles Q1..Q5"
        )
        for kind in ["r", "p", "rp"]:
            v = res[kind]
            qstr = ""
            if v["quintiles"] is not None:
                qstr = "  ".join(f"{x:+.4f}" for x in v["quintiles"])
            print(
                f"{kind:>8s}  {v['ic']:>8.4f}  {v['rho']:>7.3f}  "
                f"{v['sharpe']:>11.3f}  {qstr}"
            )

    # 전 자산 통합: 각 variant의 평균 IC, 평균 Sharpe
    print("\n" + "=" * 90)
    print("자산 통합 요약")
    print("=" * 90)
    print(f"{'variant':>8s}  {'mean|IC|':>9s}  {'mean rho':>10s}  {'mean Sharpe':>13s}")
    for kind in ["r", "p", "rp"]:
        ics, rhos, shs = [], [], []
        for name, (close, fwd, r) in data.items():
            res = evaluate(close, fwd, name)[kind]
            if np.isfinite(res["ic"]):
                ics.append(abs(res["ic"]))
            if np.isfinite(res["rho"]):
                rhos.append(res["rho"])
            if np.isfinite(res["sharpe"]):
                shs.append(res["sharpe"])
        print(
            f"{kind:>8s}  {np.mean(ics):>9.4f}  {np.mean(rhos):>10.3f}  "
            f"{np.mean(shs):>13.3f}"
        )

    print("\n판정: mean Sharpe 최고 variant가 v1.3 후보")
    print("※ 실제 채택은 자산별 안정성도 함께 검토")


if __name__ == "__main__":
    raise SystemExit(main())
