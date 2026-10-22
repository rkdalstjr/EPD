"""Regime-conditional EPD 실험 (A, D, C1, C2).

수정 사항:
  1) 이중 통제: Spearman rank가 아닌 (a) raw-value Pearson partial,
     (b) RMS 5분위 bin 카테고리 통제로 비선형 제대로 처리
  2) Regime 분류: |z_rms| 아닌 raw RMS의 causal rolling quantile
  3) 후보 4개 동일 자산/날짜 비교: A, D, C1, C2
  4) 시간 순서 검증: 4기간 분할
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import compute_epd

warnings.filterwarnings("ignore")

ASSETS = {
    "SPX": "^GSPC", "NASDAQ": "^IXIC",
    "KOSPI": "^KS11", "KOSDAQ": "^KQ11",
    "Samsung": "005930.KS", "SK_Hynix": "000660.KS",
    "NAVER": "035420.KS", "LG_Chem": "051910.KS",
}
H = 5
W_RMS = 3
W_Z = 252
REGIME_WIN = 252
REGIME_Q = 0.67     # high-vol = top 33%
PERIODS = [
    ("2005-2010", "2005-01-01", "2010-12-31"),
    ("2010-2015", "2011-01-01", "2015-12-31"),
    ("2015-2020", "2016-01-01", "2020-12-31"),
    ("2020-2025", "2021-01-01", "2025-12-31"),
]
BOOT_BLOCK = 20
BOOT_N = 500


# ============================================================
#  Utilities
# ============================================================
def rolling_rms(r, w=W_RMS):
    n = len(r); out = np.full(n, np.nan)
    for t in range(w - 1, n):
        win = r[t-w+1:t+1]
        if np.isfinite(win).all():
            out[t] = float(np.sqrt(np.mean(win ** 2)))
    return out

def fwd_vol(r, h=H):
    n = len(r); out = np.full(n, np.nan)
    for t in range(n - h):
        win = r[t+1:t+1+h]
        win = win[np.isfinite(win)]
        if win.size >= 3: out[t] = win.std(ddof=1)
    return out

def causal_regime_high_vol(rms, window=REGIME_WIN, q=REGIME_Q):
    """At t, True if rms[t] >= rolling quantile of rms[t-window:t].
    과거 데이터만 사용 (causal)."""
    n = len(rms)
    out = np.zeros(n, dtype=bool)
    for t in range(window, n):
        if not np.isfinite(rms[t]): continue
        past = rms[t-window:t]
        past = past[np.isfinite(past)]
        if past.size < window * 0.5: continue
        cutoff = np.quantile(past, q)
        out[t] = rms[t] >= cutoff
    return out

def spear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 200: return np.nan
    return float(spearmanr(x[m], y[m]).statistic)

def partial_spear_1control(x, y, c):
    """rank-based partial (1 control)."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if m.sum() < 200: return np.nan
    xr = pd.Series(x[m]).rank().to_numpy()
    yr = pd.Series(y[m]).rank().to_numpy()
    cr = pd.Series(c[m]).rank().to_numpy()
    X = np.column_stack([np.ones(len(cr)), cr])
    rx = xr - X @ np.linalg.lstsq(X, xr, rcond=None)[0]
    ry = yr - X @ np.linalg.lstsq(X, yr, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0,1])

def partial_pearson_2control(x, y, c1, c2):
    """raw-value Pearson partial, 2 controls (RMS, RMS²)."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c1) & np.isfinite(c2)
    if m.sum() < 200: return np.nan
    X = np.column_stack([np.ones(m.sum()), c1[m], c2[m]])
    rx = x[m] - X @ np.linalg.lstsq(X, x[m], rcond=None)[0]
    ry = y[m] - X @ np.linalg.lstsq(X, y[m], rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0,1])

def partial_binned(x, y, control, n_bins=5):
    """rank partial controlling for control via quantile bins (non-linear)."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    if m.sum() < 200: return np.nan
    xr = pd.Series(x[m]).rank().to_numpy()
    yr = pd.Series(y[m]).rank().to_numpy()
    c = control[m]
    try:
        bins = pd.qcut(c, n_bins, labels=False, duplicates="drop")
    except Exception:
        return np.nan
    nb = int(bins.max()) + 1
    if nb < 2: return np.nan
    X = np.ones((len(xr), nb))
    for i in range(1, nb):
        X[:, i] = (bins == i).astype(float)
    rx = xr - X @ np.linalg.lstsq(X, xr, rcond=None)[0]
    ry = yr - X @ np.linalg.lstsq(X, yr, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0,1])

def q5_q1_ratio(x, y, nq=5):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 200: return np.nan
    a, b = x[m], y[m]
    qs = np.quantile(a, np.linspace(0, 1, nq+1)); qs[0] -= 1e-12; qs[-1] += 1e-12
    means = []
    for i in range(nq):
        mm = (a >= qs[i]) & (a < qs[i+1])
        if mm.sum() < 20: return np.nan
        means.append(b[mm].mean())
    return float(means[-1] / means[0]) if means[0] > 0 else np.nan

def block_bootstrap_ci(x, y, fn=spear, block=BOOT_BLOCK, B=BOOT_N, seed=42):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 500: return np.nan, np.nan
    x, y = x[m], y[m]; n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    vals = []
    for _ in range(B):
        starts = rng.integers(0, n - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        v = fn(x[idx], y[idx])
        if np.isfinite(v): vals.append(v)
    if not vals: return np.nan, np.nan
    arr = np.array(vals)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

def load(tkr):
    import yfinance as yf
    df = yf.download(tkr, start="2005-01-01", end="2025-12-31",
                     progress=False, auto_adjust=True)
    if df is None or len(df) < 1000: return None, None
    c = df["Close"].squeeze().to_numpy(float)
    v = np.isfinite(c)
    return c[v], df.index[v]


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 115)
    print("Regime-conditional EPD — A / D / C1 / C2 비교")
    print("=" * 115)

    per_asset = []
    per_period = []

    for name, tkr in ASSETS.items():
        print(f"\n[{name}]")
        close, dates = load(tkr)
        if close is None: print("  skip"); continue

        r = np.full_like(close, np.nan); r[1:] = np.diff(np.log(close))
        fv = fwd_vol(r, H)
        rms = rolling_rms(r, W_RMS)
        regime_high = causal_regime_high_vol(rms)

        # Candidates
        out_a = compute_epd(close, z_r_mode="daily")
        out_d = compute_epd(close, z_r_mode="3day_rms")
        epd_a = out_a["epd_mag"]; epd_d = out_d["epd_mag"]
        epd_c2 = np.where(regime_high, epd_d, epd_a)
        z_r_a = out_a["z_r"]; z_r_d = out_d["z_r"]

        # 진단
        m0 = np.isfinite(epd_d) & np.isfinite(rms)
        if m0.sum() > 200:
            corr_epd_rms = float(np.corrcoef(epd_d[m0], rms[m0])[0,1])
            corr_epd_abszr = float(np.corrcoef(epd_d[m0], np.abs(z_r_d[m0]))[0,1])
            corr_epd_absze = float(np.corrcoef(epd_d[m0], np.abs(out_d["z_e"][m0]))[0,1])
        else:
            corr_epd_rms = corr_epd_abszr = corr_epd_absze = np.nan
        n_high = int(regime_high.sum())

        # ============================================================
        # ① 이중 통제 진단 (Spearman vs Pearson vs Binned)
        # ============================================================
        p_spear_1 = partial_spear_1control(epd_d, fv, rms)
        p_spear_2 = partial_spear_1control(epd_d, fv, rms)  # rank(X)=rank(X²)
        p_pearson_2 = partial_pearson_2control(epd_d, fv, rms, rms ** 2)
        p_binned = partial_binned(epd_d, fv, rms, n_bins=5)

        # ============================================================
        # ② 상관 (fwd_vol), 각 후보 full/subset
        # ============================================================
        def corr_on(x, subset_mask=None):
            base = np.isfinite(x) & np.isfinite(fv)
            if subset_mask is not None:
                base = base & subset_mask
            xs, ys = x[base], fv[base]
            if len(xs) < 200: return np.nan
            return spear(xs, ys)

        rho_a_full = corr_on(epd_a)
        rho_d_full = corr_on(epd_d)
        rho_c2_full = corr_on(epd_c2)
        rho_a_hi = corr_on(epd_a, regime_high)
        rho_d_hi = corr_on(epd_d, regime_high)
        rho_a_lo = corr_on(epd_a, ~regime_high)
        rho_d_lo = corr_on(epd_d, ~regime_high)

        # ============================================================
        # ③ Q5/Q1
        # ============================================================
        q_a = q5_q1_ratio(epd_a, fv)
        q_d = q5_q1_ratio(epd_d, fv)
        q_c2 = q5_q1_ratio(epd_c2, fv)

        # ============================================================
        # ④ Partial (RMS 통제, Pearson 기반 — 진짜 비선형)
        # ============================================================
        part_a = partial_pearson_2control(epd_a, fv, rms, rms**2)
        part_d = partial_pearson_2control(epd_d, fv, rms, rms**2)
        part_c2 = partial_pearson_2control(epd_c2, fv, rms, rms**2)

        # ============================================================
        # ⑤ Bootstrap CI (핵심 비교)
        # ============================================================
        ci_a = block_bootstrap_ci(epd_a, fv)
        ci_d = block_bootstrap_ci(epd_d, fv)
        ci_c2 = block_bootstrap_ci(epd_c2, fv)

        # ---- 출력 ----
        print(f"  [진단]")
        print(f"    corr(EPD_D, RMS)     = {corr_epd_rms:+.4f}")
        print(f"    corr(EPD_D, |Zr|)    = {corr_epd_abszr:+.4f}")
        print(f"    corr(EPD_D, |Ze|)    = {corr_epd_absze:+.4f}")
        print(f"    high-vol 비율        = {n_high/len(close):.1%}")

        print(f"  [① 이중 통제 진단]")
        print(f"    Spearman(단일 RMS)       = {p_spear_1:+.4f}")
        print(f"    Spearman(RMS, RMS²)      = {p_spear_2:+.4f}  ← rank collinear 확인")
        print(f"    Pearson(RMS, RMS²)       = {p_pearson_2:+.4f}  ← 진짜 raw")
        print(f"    Binned(RMS 5분위)         = {p_binned:+.4f}  ← 비선형 제대로")

        print(f"  [② 상관]")
        print(f"    A full = {rho_a_full:+.4f} | D full = {rho_d_full:+.4f} | C2 full = {rho_c2_full:+.4f}")
        print(f"    A hi   = {rho_a_hi:+.4f} | D hi   = {rho_d_hi:+.4f}")
        print(f"    A lo   = {rho_a_lo:+.4f} | D lo   = {rho_d_lo:+.4f}")

        print(f"  [③ Q5/Q1]")
        print(f"    A = {q_a:.3f} | D = {q_d:.3f} | C2 = {q_c2:.3f}")

        print(f"  [④ Partial (RMS, RMS² 통제)]")
        print(f"    A = {part_a:+.4f} | D = {part_d:+.4f} | C2 = {part_c2:+.4f}")

        print(f"  [⑤ 95% CI]")
        print(f"    A: [{ci_a[0]:+.3f}, {ci_a[1]:+.3f}]")
        print(f"    D: [{ci_d[0]:+.3f}, {ci_d[1]:+.3f}]")
        print(f"    C2:[{ci_c2[0]:+.3f}, {ci_c2[1]:+.3f}]")

        per_asset.append({
            "asset": name,
            "p_spear": p_spear_1, "p_pearson": p_pearson_2, "p_binned": p_binned,
            "rho_a": rho_a_full, "rho_d": rho_d_full, "rho_c2": rho_c2_full,
            "rho_a_hi": rho_a_hi, "rho_d_hi": rho_d_hi,
            "rho_a_lo": rho_a_lo, "rho_d_lo": rho_d_lo,
            "q_a": q_a, "q_d": q_d, "q_c2": q_c2,
            "part_a": part_a, "part_d": part_d, "part_c2": part_c2,
        })

        # 시간 순서
        for pname, s, e in PERIODS:
            mask = (dates >= s) & (dates <= e)
            if mask.sum() < 400: continue
            rho_a_p = spear(epd_a[mask], fv[mask])
            rho_d_p = spear(epd_d[mask], fv[mask])
            rho_c2_p = spear(epd_c2[mask], fv[mask])
            per_period.append({
                "asset": name, "period": pname,
                "A": rho_a_p, "D": rho_d_p, "C2": rho_c2_p,
            })

    # ============================================================
    # 요약
    # ============================================================
    print("\n" + "=" * 115)
    print("전체 요약")
    print("=" * 115)

    df_a = pd.DataFrame(per_asset)

    print("\n[진단 — Spearman이 Pearson/Binned와 다른가?]")
    print(f"  Mean |Spearman (RMS 1-control)|:  {df_a['p_spear'].abs().mean():.4f}")
    print(f"  Mean |Pearson (RMS,RMS²)|:        {df_a['p_pearson'].abs().mean():.4f}")
    print(f"  Mean |Binned (5-quantile)|:       {df_a['p_binned'].abs().mean():.4f}")
    print(f"  → Pearson/Binned < Spearman이면 rank-collinear 문제 없음")

    print("\n[상관 요약 (mean)]")
    print(f"  A full:  {df_a['rho_a'].mean():+.4f}   D full: {df_a['rho_d'].mean():+.4f}   C2 full: {df_a['rho_c2'].mean():+.4f}")
    print(f"  A hi:    {df_a['rho_a_hi'].mean():+.4f}   D hi:   {df_a['rho_d_hi'].mean():+.4f}")
    print(f"  A lo:    {df_a['rho_a_lo'].mean():+.4f}   D lo:   {df_a['rho_d_lo'].mean():+.4f}")

    print("\n[Q5/Q1 요약]")
    print(f"  A: {df_a['q_a'].mean():.3f}   D: {df_a['q_d'].mean():.3f}   C2: {df_a['q_c2'].mean():.3f}")

    print("\n[Partial (RMS,RMS² 통제) 요약]")
    print(f"  A: {df_a['part_a'].mean():+.4f}   D: {df_a['part_d'].mean():+.4f}   C2: {df_a['part_c2'].mean():+.4f}")
    print(f"  D 양수: {(df_a['part_d']>0).sum()}/8, C2 양수: {(df_a['part_c2']>0).sum()}/8")

    print("\n[시간 순서 검증]")
    df_p = pd.DataFrame(per_period)
    if not df_p.empty:
        piv = df_p.pivot_table(index="period", values=["A","D","C2"], aggfunc="mean")
        print(piv.round(3).to_string())

    # 최종 판정
    print("\n" + "=" * 115)
    print("최종 판정")
    print("=" * 115)

    spearman_vs_binned_gap = df_a['p_spear'].abs().mean() - df_a['p_binned'].abs().mean()
    c2_vs_a_gap = df_a['part_c2'].mean() - df_a['part_a'].mean()
    d_hi_vs_a_hi = df_a['rho_d_hi'].mean() - df_a['rho_a_hi'].mean()

    print(f"  Spearman - Binned gap:  {spearman_vs_binned_gap:+.4f}")
    print(f"    (양수면 rank-collinear 문제로 과대평가된 것)")
    print(f"  D(hi) - A(hi):          {d_hi_vs_a_hi:+.4f}")
    print(f"  C2 - A (partial):       {c2_vs_a_gap:+.4f}")
    print()
    print("  판정:")
    if c2_vs_a_gap > 0.01 and (df_a['part_c2'] > 0).sum() >= 6:
        print("  ✅ C2 (regime-conditional) 우월 — v3.3 후보 확정 근거")
    elif c2_vs_a_gap > 0.005:
        print("  ⚠️ C2 개선 미미 — 추가 검증 필요")
    else:
        print("  ❌ C2 개선 없음 — regime-conditional 설계 불필요")
    print()

    if abs(spearman_vs_binned_gap) > 0.02:
        print("  ⚠️ Spearman vs Binned 격차 큼 → 이전 ② 결과 재해석 필요")
    else:
        print("  ✅ Spearman-Pearson-Binned 일치 — 이전 결론 방어 가능")


if __name__ == "__main__":
    raise SystemExit(main())