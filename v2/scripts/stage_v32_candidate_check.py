"""v3.2 후보 D 정밀 검증 (①~⑤) — 수정판."""
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


def rms3(r):
    out = np.full_like(r, np.nan)
    for t in range(2, len(r)):
        w = r[t-2:t+1]
        if np.isfinite(w).all():
            out[t] = float(np.sqrt(np.mean(w ** 2)))
    return out

def fwd_vol(r, h=H):
    n = len(r); out = np.full(n, np.nan)
    for t in range(n - h):
        w = r[t+1:t+1+h]; w = w[np.isfinite(w)]
        if w.size >= 3: out[t] = w.std(ddof=1)
    return out

def zscore_causal(x, w=252):
    out = np.full_like(x, np.nan)
    for t in range(w, len(x)):
        win = x[t-w:t]; win = win[np.isfinite(win)]
        if win.size < w//2 or not np.isfinite(x[t]): continue
        mu, sd = win.mean(), win.std(ddof=1)
        if sd > 0: out[t] = (x[t] - mu) / sd
    return out

def spear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 200: return np.nan
    return float(spearmanr(x[m], y[m]).statistic)

def partial_spear(x, y, c):
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

def partial2_spear(x, y, c1, c2):
    m = (np.isfinite(x) & np.isfinite(y) & np.isfinite(c1) & np.isfinite(c2))
    if m.sum() < 200: return np.nan
    xr = pd.Series(x[m]).rank().to_numpy()
    yr = pd.Series(y[m]).rank().to_numpy()
    c1r = pd.Series(c1[m]).rank().to_numpy()
    c2r = pd.Series(c2[m]).rank().to_numpy()
    X = np.column_stack([np.ones(len(xr)), c1r, c2r])
    rx = xr - X @ np.linalg.lstsq(X, xr, rcond=None)[0]
    ry = yr - X @ np.linalg.lstsq(X, yr, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0,1])

def load(tkr):
    import yfinance as yf
    df = yf.download(tkr, start="2005-01-01", end="2025-12-31",
                     progress=False, auto_adjust=True)
    if df is None or len(df) < 1000: return None, None
    c = df["Close"].squeeze().to_numpy(float)
    v = np.isfinite(c)
    return c[v], df.index[v]


def main():
    print("=" * 110)
    print("v3.2 후보 D 정밀 검증 (수정판)")
    print("=" * 110)

    rows_1, rows_2, rows_3, rows_4, rows_5 = [], [], [], [], []
    rows_diag = []   # ★ 진단: epd_mag와 |z_r|의 관계

    for name, tkr in ASSETS.items():
        print(f"\n[{name}]")
        close, dates = load(tkr)
        if close is None:
            print("  skip"); continue

        r = np.full_like(close, np.nan); r[1:] = np.diff(np.log(close))
        fv = fwd_vol(r, H)

        rms = rms3(r)
        z_rms = zscore_causal(rms, 252)
        abs_rms = np.abs(z_rms)

        out_d = compute_epd(close, z_r_mode="3day_rms")
        epd_d = out_d["epd_mag"]
        z_r_d = out_d["z_r"]
        z_e_d = out_d["z_e"]
        abs_zr = np.abs(z_r_d)
        abs_ze = np.abs(z_e_d)

        # ★★★ 진단: epd_mag와 |z_r| 상관 (같으면 z_e 무용) ★★★
        m_diag = np.isfinite(epd_d) & np.isfinite(abs_zr)
        corr_epd_abszr = float(np.corrcoef(epd_d[m_diag], abs_zr[m_diag])[0,1]) if m_diag.sum() > 200 else np.nan
        corr_epd_absze = float(np.corrcoef(epd_d[m_diag & np.isfinite(abs_ze)],
                                            abs_ze[m_diag & np.isfinite(abs_ze)])[0,1]) if m_diag.sum() > 200 else np.nan
        corr_abszr_absze = float(np.corrcoef(abs_zr[m_diag & np.isfinite(abs_ze)],
                                              abs_ze[m_diag & np.isfinite(abs_ze)])[0,1]) if m_diag.sum() > 200 else np.nan
        mean_absze = float(abs_ze[np.isfinite(abs_ze)].mean())
        mean_abszr = float(abs_zr[np.isfinite(abs_zr)].mean())

        rows_diag.append({
            "asset": name,
            "corr(EPD_mag, |z_r|)": corr_epd_abszr,
            "corr(EPD_mag, |z_e|)": corr_epd_absze,
            "corr(|z_r|, |z_e|)": corr_abszr_absze,
            "mean|z_r|": mean_abszr,
            "mean|z_e|": mean_absze,
        })
        print(f"  [진단] corr(EPD_mag, |Zr|)={corr_epd_abszr:+.3f}  "
              f"corr(EPD_mag, |Ze|)={corr_epd_absze:+.3f}  "
              f"corr(|Zr|, |Ze|)={corr_abszr_absze:+.3f}")
        print(f"         mean|Zr|={mean_abszr:.3f}  mean|Ze|={mean_absze:.3f}")

        # ---- ① RMS 단독 vs EPD_D ----
        rho_rms   = spear(abs_rms, fv)
        rho_epd_d = spear(epd_d, fv)
        p_epd_given_rms = partial_spear(epd_d, fv, abs_rms)

        rows_1.append({
            "asset": name,
            "corr_absRMS_fv": rho_rms,
            "corr_EPD_D_fv": rho_epd_d,
            "delta_EPD_over_RMS": rho_epd_d - rho_rms,
            "partial_EPD_D_given_RMS": p_epd_given_rms,
        })
        print(f"  ① absRMS→fv: {rho_rms:+.4f}  |  EPD_D→fv: {rho_epd_d:+.4f}  "
              f"|  EPD_D|RMS: {p_epd_given_rms:+.4f}")

        # ---- ② 비선형 통제 ----
        p2 = partial2_spear(epd_d, fv, abs_rms, abs_rms ** 2)
        rows_2.append({"asset": name,
                        "partial_EPD_D_given_RMS_and_RMS2": p2})
        print(f"  ② EPD_D | RMS, RMS²: {p2:+.4f}")

        # ---- ③ IS vs OOS (수정) ----
        for label, s, e in [
            ("IS_2005-2019",  "2005-01-01", "2019-12-31"),
            ("OOS_2020-2025", "2020-01-01", "2025-12-31"),
        ]:
            mask = (dates >= s) & (dates <= e)
            if mask.sum() < 400: continue
            fv_m = fv[mask]; epd_dm = epd_d[mask]; rms_m = abs_rms[mask]
            pd_ = partial_spear(epd_dm, fv_m, rms_m)
            pr_ = spear(rms_m, fv_m)
            rows_3.append({"asset": name, "period": label,
                            "corr_absRMS": pr_, "partial_EPD_D": pd_})

        # ---- ④ 저변동 vs 고변동 ----
        m = np.isfinite(abs_rms) & np.isfinite(fv) & np.isfinite(epd_d)
        if m.sum() > 500:
            qs = np.quantile(abs_rms[m], [0, 0.33, 0.66, 1.0])
            lo_mask = m & (abs_rms <= qs[1])
            hi_mask = m & (abs_rms >= qs[2])
            lo_corr = spear(epd_d[lo_mask], fv[lo_mask])
            hi_corr = spear(epd_d[hi_mask], fv[hi_mask])
            rows_4.append({
                "asset": name,
                "corr_EPD_D_lowRMS": lo_corr,
                "corr_EPD_D_highRMS": hi_corr,
                "delta_hi_minus_lo": hi_corr - lo_corr,
            })

        # ---- ⑤ 취약 구간 (수정) ----
        for label, s, e in [
            ("KOSPI_2015-2020", "2015-01-01", "2020-12-31"),
            ("KR_2020-2025",    "2020-01-01", "2025-12-31"),
        ]:
            mask = (dates >= s) & (dates <= e)
            if mask.sum() < 300: continue
            fv_m = fv[mask]; epd_dm = epd_d[mask]; rms_m = abs_rms[mask]
            p_d = partial_spear(epd_dm, fv_m, rms_m)
            ze_m = np.abs(z_e_d[mask])
            zr_m = np.abs(z_r_d[mask])
            corr_ze = spear(ze_m, fv_m)
            corr_zr = spear(zr_m, fv_m)
            rows_5.append({"asset": name, "window": label,
                            "partial_EPD_D": p_d,
                            "corr_absZr": corr_zr,
                            "corr_absZe": corr_ze})

    # ============================================================
    print("\n" + "=" * 110)
    print("★★ 진단: epd_mag와 |Zr|의 관계 (z_e가 실제로 기여하나?)")
    print("=" * 110)
    df_diag = pd.DataFrame(rows_diag)
    print(df_diag.round(3).to_string(index=False))
    print(f"\n평균 corr(EPD_mag, |Zr|): {df_diag['corr(EPD_mag, |z_r|)'].mean():.3f}")
    print(f"평균 corr(EPD_mag, |Ze|): {df_diag['corr(EPD_mag, |z_e|)'].mean():.3f}")
    print(f"평균 corr(|Zr|, |Ze|):    {df_diag['corr(|z_r|, |z_e|)'].mean():.3f}")
    print()
    if df_diag["corr(EPD_mag, |z_r|)"].mean() > 0.9:
        print("⚠️ corr(EPD_mag, |Zr|) > 0.9 → z_e가 거의 작동하지 않음 (EPD ≈ |Zr|)")
    elif df_diag["corr(EPD_mag, |z_r|)"].mean() > 0.7:
        print("⚠️ corr(EPD_mag, |Zr|) > 0.7 → z_e 기여 미미")
    else:
        print("✅ z_e가 독립적 기여 (corr < 0.7)")

    print("\n" + "=" * 110)
    print("① RMS 단독 vs EPD_D")
    print("=" * 110)
    df1 = pd.DataFrame(rows_1)
    print(df1.round(4).to_string(index=False))
    print(f"\n평균 |RMS→fv| {df1['corr_absRMS_fv'].abs().mean():.4f}  |  "
          f"|EPD_D→fv| {df1['corr_EPD_D_fv'].abs().mean():.4f}  |  "
          f"partial_EPD|RMS {df1['partial_EPD_D_given_RMS'].mean():+.4f}")

    print("\n" + "=" * 110)
    print("② 이중 통제 (RMS, RMS²)")
    print("=" * 110)
    df2 = pd.DataFrame(rows_2)
    print(df2.round(4).to_string(index=False))
    print(f"\n평균: {df2['partial_EPD_D_given_RMS_and_RMS2'].mean():+.4f}  "
          f"양수: {(df2['partial_EPD_D_given_RMS_and_RMS2'] > 0).sum()}/{len(df2)}")

    print("\n" + "=" * 110)
    print("③ IS vs OOS")
    print("=" * 110)
    df3 = pd.DataFrame(rows_3)
    print(df3.pivot_table(index="asset", columns="period",
                            values=["corr_absRMS", "partial_EPD_D"],
                            aggfunc="first").round(4).to_string())

    print("\n" + "=" * 110)
    print("④ 저/고변동 구간")
    print("=" * 110)
    df4 = pd.DataFrame(rows_4)
    print(df4.round(4).to_string(index=False))
    print(f"\n저변동 평균: {df4['corr_EPD_D_lowRMS'].mean():+.4f}")
    print(f"고변동 평균: {df4['corr_EPD_D_highRMS'].mean():+.4f}")

    print("\n" + "=" * 110)
    print("⑤ 취약 구간")
    print("=" * 110)
    df5 = pd.DataFrame(rows_5)
    print(df5.round(4).to_string(index=False))

    # ---- 최종 판정 ----
    print("\n" + "=" * 110)
    print("최종 판정")
    print("=" * 110)

    corr_epd_zr = df_diag["corr(EPD_mag, |z_r|)"].mean()
    mean_p2 = df2["partial_EPD_D_given_RMS_and_RMS2"].mean()
    pos_p2 = (df2["partial_EPD_D_given_RMS_and_RMS2"] > 0).sum()

    print(f"진단: corr(EPD_mag, |Zr|) = {corr_epd_zr:.3f}")
    print(f"② 이중 통제 후 partial = {mean_p2:+.4f}, 양수 {pos_p2}/8")
    print()

    if corr_epd_zr > 0.9:
        print("❌ EPD ≈ |Zr| → z_e 무용. D의 개선은 RMS 효과만.")
        print("   → v3.2 후보 D 폐기, v3.1 유지")
    elif mean_p2 > 0.02 and pos_p2 >= 6:
        print("✅ RMS 통제 후에도 EPD_D 정보 유지 — 엔트로피 정보 개선 실증")
        print("   → v3.2 확정 근거 축적")
    elif mean_p2 > 0.005:
        print("⚠️ RMS 통제 후 EPD_D 정보 미약 — 조건부 유효")
    else:
        print("❌ RMS 통제 후 EPD_D 정보 소멸 — 변동성 프록시")


if __name__ == "__main__":
    raise SystemExit(main())