"""Stage 3: Universality — 36조합 보편성 검증.

판정:
  - 36조합 각각에 대해 5분위 forward return의 Spearman rank corr (quintile means vs 1..5)
  - 부호가 일치하는 조합 비율 ≥ 90% → PASS
  - 로버스트 영역의 "중심값 선택"은 하지 않음 (그건 최적화)
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))

from epd_core import compute_epd, rolling_slope  # noqa: E402

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------
# 데이터 로딩 (yfinance 시도 → 실패 시 합성)
# ---------------------------------------------------------------
def load_real_data():
    """여러 자산 로드. 실패하면 None 반환."""
    try:
        import yfinance as yf
    except ImportError:
        print("[data] yfinance 없음 — 합성 데이터로 폴백")
        return None

    tickers = {
        "SPX": "^GSPC",
        "KOSPI": "^KS11",
        "NASDAQ": "^IXIC",
    }
    out = {}
    for name, tkr in tickers.items():
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
                print(f"[data] {name} 실패 (n={0 if df is None else len(df)})")
                continue
            close = df["Close"].squeeze().to_numpy(dtype=float)
            close = close[np.isfinite(close)]
            ret = np.diff(np.log(close))
            out[name] = {"close": close[1:], "returns": ret}
            print(f"[data] {name}: n={len(ret)}")
        except Exception as e:
            print(f"[data] {name} 에러: {e}")
    return out if out else None


def synth_scenarios(n=3000, seed=42):
    """합성 시나리오 여러 개 (yfinance 실패 시 사용)."""
    rng = np.random.default_rng(seed)
    sc = {}

    # 1) 추세 지속 (AR(1) 양수)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.05 + 0.3 * r[t - 1] + rng.standard_normal() * 0.5
    sc["AR_pos"] = {"returns": r, "close": np.exp(np.cumsum(r))}

    # 2) 평균 회귀 (AR(1) 음수)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = -0.2 * r[t - 1] + rng.standard_normal() * 0.5
    sc["AR_neg"] = {"returns": r, "close": np.exp(np.cumsum(r))}

    # 3) GARCH
    r = np.zeros(n)
    s2 = np.zeros(n)
    s2[0] = 1.0
    eps = rng.standard_normal(n)
    for t in range(1, n):
        s2[t] = 0.05 + 0.1 * r[t - 1] ** 2 + 0.85 * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * eps[t]
    sc["GARCH"] = {"returns": r, "close": np.exp(np.cumsum(r))}

    # 4) Regime switch
    states = np.zeros(n, dtype=int)
    state = 0
    for t in range(n):
        if rng.random() > 0.98:
            state = 1 - state
        states[t] = state
    r = rng.standard_normal(n) * np.where(states == 0, 1.0, 3.0)
    sc["Regime"] = {"returns": r, "close": np.exp(np.cumsum(r))}

    # 5) Drift (상승)
    r = 0.05 + rng.standard_normal(n) * 1.0
    sc["Drift_up"] = {"returns": r, "close": np.exp(np.cumsum(r))}

    return sc


# ---------------------------------------------------------------
# 5분위 forward return 단조성
# ---------------------------------------------------------------
def forward_return(r, h=5):
    fr = np.full_like(r, np.nan)
    for t in range(len(r) - h):
        fr[t] = r[t + 1 : t + 1 + h].sum()
    return fr


def quintile_monotonicity(epd, fwd, n_q=5):
    """EPD 분위별 fwd return 평균의 Spearman rank corr (quintile 1..5 vs means)."""
    mask = np.isfinite(epd) & np.isfinite(fwd)
    if mask.sum() < n_q * 30:
        return np.nan
    e = epd[mask]
    f = fwd[mask]
    # 분위 경계
    qs = np.quantile(e, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    means = []
    for i in range(n_q):
        m = (e >= qs[i]) & (e < qs[i + 1])
        if m.sum() < 20:
            return np.nan
        means.append(f[m].mean())
    means = np.array(means)
    # Spearman rank corr of (1..n_q, means)
    rank_q = np.arange(1, n_q + 1)
    rank_m = np.argsort(np.argsort(means)) + 1
    return float(np.corrcoef(rank_q, rank_m)[0, 1])


# ---------------------------------------------------------------
# 36조합 × 자산별 monotonicity
# ---------------------------------------------------------------
def run_grid(returns_dict, price_variants=None):
    """price_variants: dict(name -> 함수(r) -> 입력시계열). None이면 return만."""
    if price_variants is None:
        price_variants = {"return": lambda r: r, "slope": None}

    w_pe_grid = [20, 60, 120]
    m_grid = [3, 4]
    w_z_grid = [60, 252, 504]

    rows = []
    for asset_name, data in returns_dict.items():
        r = data["returns"]
        close = data.get("close")
        for w_pe, m, w_z, (pv_name, pv_fn) in product(
            w_pe_grid, m_grid, w_z_grid, price_variants.items()
        ):

            # price variant: 입력 시계열을 만든 뒤 PE/z에 사용
            if pv_name == "return":
                r_input = r
            elif pv_name == "slope":
                if close is None:
                    continue
                slope = rolling_slope(close, w=20)
                r_input = np.where(np.isfinite(slope), slope, 0.0)
            else:
                continue

            try:
                out = compute_epd(r_input, w_pe=w_pe, m=m, tau=1, w_z=w_z)
            except Exception:
                continue

            epd = out["epd_100"]
            fwd = forward_return(r, h=5)  # fwd는 원래 r 기준
            rho = quintile_monotonicity(epd, fwd, n_q=5)

            rows.append(
                {
                    "asset": asset_name,
                    "w_pe": w_pe,
                    "m": m,
                    "w_z": w_z,
                    "price": pv_name,
                    "rho": rho,
                    "sign": int(np.sign(rho)) if np.isfinite(rho) else 0,
                }
            )

    return rows


# ---------------------------------------------------------------
# 메인
# ---------------------------------------------------------------
def main():
    print("=" * 70)
    print("Stage 3: Universality Check")
    print("=" * 70)

    data = load_real_data()
    source = "real"
    if data is None:
        data = synth_scenarios()
        source = "synthetic"
    print(f"[data] source={source}, assets={list(data.keys())}")

    rows = run_grid(data, price_variants={"return": None, "slope": None})

    # 유효 조합만
    valid = [r for r in rows if np.isfinite(r["rho"])]
    if not valid:
        print("\n[FAIL] 유효한 조합이 없음")
        return 1

    # 자산별 방향 통계
    print(
        f"\n{'asset':10s}  {'n_valid':>8s}  {'frac_pos':>9s}  {'frac_neg':>9s}  {'median_rho':>11s}"
    )
    print("-" * 60)
    all_same_dir_asset = {}
    for asset in sorted(set(r["asset"] for r in valid)):
        sub = [r for r in valid if r["asset"] == asset]
        signs = [r["sign"] for r in sub]
        npos = sum(1 for s in signs if s > 0)
        nneg = sum(1 for s in signs if s < 0)
        rho_med = float(np.median([r["rho"] for r in sub]))
        frac_pos = npos / len(sub)
        frac_neg = nneg / len(sub)
        print(
            f"{asset:10s}  {len(sub):>8d}  {frac_pos:>9.2%}  {frac_neg:>9.2%}  {rho_med:>11.4f}"
        )
        all_same_dir_asset[asset] = max(frac_pos, frac_neg)

    # 전체 판정: 각 자산에서 "동일 방향 비율 ≥ 90%" 를 만족해야 함
    print("\n--- 자산별 보편성 판정 (동일 방향 비율 ≥ 90%) ---")
    pass_assets = []
    fail_assets = []
    for asset, frac in all_same_dir_asset.items():
        ok = frac >= 0.90
        flag = "PASS" if ok else "FAIL"
        print(f"  {asset:10s}  {frac:.2%}  -> {flag}")
        (pass_assets if ok else fail_assets).append(asset)

    overall = len(fail_assets) == 0
    print(f"\n=== Stage 3 Result: {'PASS' if overall else 'FAIL'} ===")

    # 리포트
    rpt = ROOT / "docs" / "universality_report.md"
    lines = [
        "# Universality Report — Stage 3",
        "",
        f"Source: {source}",
        f"Assets: {list(data.keys())}",
        "Threshold: 동일 방향 비율 ≥ 90% (자산별)",
        "",
        "## 자산별 결과",
        "",
        "| Asset | n_valid | frac_pos | frac_neg | median_rho | Pass |",
        "|---|---|---|---|---|---|",
    ]
    for asset in sorted(all_same_dir_asset):
        sub = [r for r in valid if r["asset"] == asset]
        signs = [r["sign"] for r in sub]
        npos = sum(1 for s in signs if s > 0)
        nneg = sum(1 for s in signs if s < 0)
        rho_med = float(np.median([r["rho"] for r in sub]))
        frac_pos = npos / len(sub)
        frac_neg = nneg / len(sub)
        ok = max(frac_pos, frac_neg) >= 0.90
        lines.append(
            f"| {asset} | {len(sub)} | {frac_pos:.2%} | {frac_neg:.2%} | "
            f"{rho_med:.4f} | {'✅' if ok else '❌'} |"
        )

    lines.append("")
    lines.append(f"**Overall**: {'PASS' if overall else 'FAIL'}")
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {rpt}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
