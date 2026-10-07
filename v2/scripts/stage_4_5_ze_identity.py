"""Stage 4.5 — Z_e의 정체 검증.

Stage 4에서 발견:
  - Z_r은 forward vol 예측 (특히 하락 방향)
  - Z_e는 forward vol과 무관
  - 그런데 var(EPD_raw)의 50%를 Z_e가 담당

질문:
  Z_e는 무엇을 재는가? 어떤 forward 통계와 연결되는가?

Forward targets (5일, 20일):
  1. forward_vol          — 미래 변동성 (이미 무관 확인)
  2. forward_abs_ret      — 미래 |수익률| (크기)
  3. forward_autocorr     — 미래 자기상관
  4. forward_PE_change    — 미래 엔트로피 변화
  5. forward_skew         — 미래 왜도
  6. forward_regime_persist — regime 지속 여부 (부호 지속)
  7. forward_tail         — 미래 |return| 최대값

비교: Z_r vs Z_e vs EPD_raw
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd, rolling_pe

warnings.filterwarnings("ignore")

TICKERS = [
    ("^GSPC", "SPX", "US_index"),
    ("^IXIC", "NASDAQ", "US_index"),
    ("^KS11", "KOSPI", "KR_index"),
    ("^KQ11", "KOSDAQ", "KR_index"),
    ("005930.KS", "삼성전자", "KR_stock"),
    ("000660.KS", "SK하이닉스", "KR_stock"),
    ("035420.KS", "NAVER", "KR_stock"),
    ("051910.KS", "LG화학", "KR_stock"),
]

HORIZONS = [5, 20]


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    xr = np.argsort(np.argsort(x[m]))
    yr = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(xr, yr)[0, 1])


def forward_vol(r, h):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.std(ddof=1)
    return fv


def forward_abs_ret(r, h):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fv[t] = np.nansum(np.abs(r[t + 1 : t + 1 + h]))
    return fv


def forward_autocorr(r, h):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size < 3 or w.std() == 0:
            continue
        x, y = w[:-1], w[1:]
        if x.std() == 0 or y.std() == 0:
            continue
        fv[t] = np.corrcoef(x, y)[0, 1]
    return fv


def forward_pe_change(r, h, w_pe=60):
    """미래 h일 후의 PE − 현재 PE."""
    n = len(r)
    pe_now = rolling_pe(r, w_pe=w_pe, m=3, tau=1)
    fv = np.full(n, np.nan)
    for t in range(n - h):
        win = r[t + 1 : t + 1 + h + w_pe]
        if np.isfinite(win).sum() < w_pe + h:
            continue
        pe_future = rolling_pe(r, w_pe=w_pe, m=3, tau=1)
        # 단순화: 미래 시점 t+h의 PE
        if t + h < n and np.isfinite(pe_future[t + h]) and np.isfinite(pe_now[t]):
            fv[t] = pe_future[t + h] - pe_now[t]
    return fv


def forward_skew(r, h):
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = r[t + 1 : t + 1 + h]
        w = w[np.isfinite(w)]
        if w.size < 4 or w.std() == 0:
            continue
        fv[t] = ((w - w.mean()) ** 3).mean() / (w.std(ddof=1) ** 3)
    return fv


def forward_regime_persist(r, h):
    """현재 부호가 미래 h일 동안 얼마나 지속되는가 (0~1)."""
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        if not np.isfinite(r[t]) or r[t] == 0:
            continue
        cur_sign = np.sign(r[t])
        win = r[t + 1 : t + 1 + h]
        win = win[np.isfinite(win)]
        if win.size < 3:
            continue
        same = (np.sign(win) == cur_sign).mean()
        fv[t] = same
    return fv


def forward_tail(r, h):
    """미래 h일 중 |r|의 최대값."""
    fv = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        w = np.abs(r[t + 1 : t + 1 + h])
        w = w[np.isfinite(w)]
        if w.size >= 3:
            fv[t] = w.max()
    return fv


TARGETS = {
    "fwd_vol": forward_vol,
    "fwd_abs_ret": forward_abs_ret,
    "fwd_autocorr": forward_autocorr,
    "fwd_skew": forward_skew,
    "fwd_regime_persist": forward_regime_persist,
    "fwd_tail": forward_tail,
}


def main():
    import yfinance as yf

    print("=" * 130)
    print("Stage 4.5 — Z_e 정체 검증")
    print("Z_r, Z_e, EPD_raw 각각과 forward 통계의 Spearman 상관")
    print("=" * 130)

    # 결과 저장
    results = {}  # (asset, h, target) -> {"z_r": rho, "z_e": rho, "epd": rho}

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
            if df is None or len(df) < 500:
                continue
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            if close.size < 500:
                continue
        except Exception as e:
            print(f"{name} err: {e}")
            continue

        out = compute_epd(close)
        z_r = out["return_z"]
        z_e = out["entropy_z"]
        epd_raw = out["epd_raw"]
        r = np.full_like(close, np.nan)
        r[1:] = np.diff(np.log(close))

        print(f"\n### {name} ({grp})\n")

        for h in HORIZONS:
            print(f"  --- H={h}일 ---")
            print(
                f"  {'target':>22s}  {'corr(Z_r)':>10s}  "
                f"{'corr(Z_e)':>10s}  {'corr(EPD)':>10s}  "
                f"{'winner':>10s}"
            )
            print("  " + "-" * 78)

            for tname, fn in TARGETS.items():
                target = fn(r, h)
                rho_r = spearman(z_r, target)
                rho_e = spearman(z_e, target)
                rho_p = spearman(epd_raw, target)

                # winner: |rho|가 가장 큰 것
                candidates = [
                    (abs(rho_r), "Z_r"),
                    (abs(rho_e), "Z_e"),
                    (abs(rho_p), "EPD"),
                ]
                winner = (
                    max(candidates, key=lambda x: x[0])[1]
                    if all(np.isfinite(x[0]) for x in candidates)
                    else "n/a"
                )

                results[(name, h, tname)] = {"z_r": rho_r, "z_e": rho_e, "epd": rho_p}

                print(
                    f"  {tname:>22s}  {rho_r:>+10.4f}  "
                    f"{rho_e:>+10.4f}  {rho_p:>+10.4f}  "
                    f"{winner:>10s}"
                )
            print()

    # ============================================================
    # 요약: target별로 어느 성분이 지배적인가
    # ============================================================
    print()
    print("=" * 130)
    print("요약 — 각 target에서 Z_r / Z_e / EPD 중 승자")
    print("=" * 130)

    print(
        f"\n{'target':>22s}  {'H':>3s}  "
        f"{'n_assets':>9s}  {'mean|rho_R|':>12s}  {'mean|rho_E|':>12s}  "
        f"{'mean|rho_P|':>12s}  {'Z_r win':>8s}  {'Z_e win':>8s}"
    )
    print("-" * 130)

    for tname in TARGETS.keys():
        for h in HORIZONS:
            rows = [
                (k[0], v) for k, v in results.items() if k[1] == h and k[2] == tname
            ]
            if not rows:
                continue
            r_vals = [abs(v["z_r"]) for _, v in rows if np.isfinite(v["z_r"])]
            e_vals = [abs(v["z_e"]) for _, v in rows if np.isfinite(v["z_e"])]
            p_vals = [abs(v["epd"]) for _, v in rows if np.isfinite(v["epd"])]

            n_r_win = sum(
                1
                for _, v in rows
                if np.isfinite(v["z_r"])
                and np.isfinite(v["z_e"])
                and abs(v["z_r"]) > abs(v["z_e"])
            )
            n_e_win = sum(
                1
                for _, v in rows
                if np.isfinite(v["z_r"])
                and np.isfinite(v["z_e"])
                and abs(v["z_e"]) > abs(v["z_r"])
            )

            print(
                f"{tname:>22s}  {h:>3d}  {len(rows):>9d}  "
                f"{np.mean(r_vals) if r_vals else np.nan:>12.4f}  "
                f"{np.mean(e_vals) if e_vals else np.nan:>12.4f}  "
                f"{np.mean(p_vals) if p_vals else np.nan:>12.4f}  "
                f"{n_r_win:>8d}  {n_e_win:>8d}"
            )

    # ============================================================
    # Z_e가 유독 강한 target 찾기
    # ============================================================
    print()
    print("=" * 130)
    print("Z_e의 정체 — Z_e가 지배적인 target (|rho_E| > 0.05)")
    print("=" * 130)

    for tname in TARGETS.keys():
        for h in HORIZONS:
            rows = [
                (k[0], v) for k, v in results.items() if k[1] == h and k[2] == tname
            ]
            e_vals = [abs(v["z_e"]) for _, v in rows if np.isfinite(v["z_e"])]
            if not e_vals:
                continue
            mean_e = np.mean(e_vals)
            if mean_e > 0.05:
                print(f"  {tname:>22s} H={h:>3d}: " f"mean|rho_E| = {mean_e:.4f}")

    print()
    print("해석:")
    print("  - Z_e가 특정 target에서 강함 → 그게 Z_e의 정체")
    print("  - 모든 target에서 약함 → Z_e는 '노이즈' 또는 다른 무언가")


if __name__ == "__main__":
    raise SystemExit(main())
