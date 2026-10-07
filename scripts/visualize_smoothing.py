"""EPD 스무딩 옵션 비교 — 원본 vs EMA(5) vs EMA(14) 등."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import rolling_pe, rolling_zscore  # noqa: E402

W_PE, M, TAU, W_Z = 60, 3, 1, 252
TANH_SCALE = 2.0


def ema(x, span):
    """지수이동평균 (NaN 무시)."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    alpha = 2 / (span + 1)
    prev = np.nan
    for t in range(len(x)):
        if np.isnan(x[t]):
            out[t] = prev
        else:
            prev = x[t] if np.isnan(prev) else alpha * x[t] + (1 - alpha) * prev
            out[t] = prev
    return out


def compute_variants(close):
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

    # 원본
    epd_raw_v0 = raw

    # A: raw를 EMA(5)
    epd_raw_v1 = ema(raw, 5)

    # A': raw를 EMA(14)
    epd_raw_v2 = ema(raw, 14)

    # B: z_r, z_e 각각 EMA(5)
    epd_raw_v3 = ema(z_r, 5) - ema(z_e, 5)

    # 원시 입력 스무딩 (C)
    def roll_mean(x, w):
        out = np.full_like(x, np.nan)
        for t in range(w, len(x)):
            win = x[t - w : t]
            win = win[np.isfinite(win)]
            if win.size >= w // 2:
                out[t] = win.mean()
        return out

    r_s = roll_mean(r, 5)
    dpe_s = roll_mean(dpe, 5)
    z_r_s = rolling_zscore(r_s, W_Z)
    z_e_s = rolling_zscore(dpe_s, W_Z)
    epd_raw_v4 = z_r_s - z_e_s

    def to100(raw):
        return 50.0 * (1.0 + np.tanh(raw / TANH_SCALE))

    return {
        "v0_원본": to100(epd_raw_v0),
        "v1_EMA5(raw)": to100(epd_raw_v1),
        "v2_EMA14(raw)": to100(epd_raw_v2),
        "v3_EMA5(z_r,z_e)": to100(epd_raw_v3),
        "v4_roll5(입력)": to100(epd_raw_v4),
    }


def make_chart(df, title, outfile):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    o = df["Open"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    c = df["Close"].squeeze()
    dates = df.index
    close_np = c.to_numpy(dtype=float)

    variants = compute_variants(close_np)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.5],
        subplot_titles=(title, "EPD 스무딩 옵션 비교"),
    )

    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=o,
            high=h,
            low=l,
            close=c,
            name="Price",
            increasing_line_color="#26a69a",
            increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350",
            decreasing_fillcolor="#ef5350",
        ),
        row=1,
        col=1,
    )

    colors = {
        "v0_원본": ("#78909c", 0.6),
        "v1_EMA5(raw)": ("#ffeb3b", 1.5),
        "v2_EMA14(raw)": ("#4fc3f7", 1.8),
        "v3_EMA5(z_r,z_e)": ("#ff7043", 1.5),
        "v4_roll5(입력)": ("#ab47bc", 1.5),
    }
    for name, y in variants.items():
        col, w = colors[name]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=col, width=w),
            ),
            row=2,
            col=1,
        )

    for y, color in [(70, "#ef5350"), (50, "#787b86"), (30, "#26a69a")]:
        fig.add_hline(y=y, line=dict(color=color, dash="dot", width=1), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(color="#d1d4dc", size=11),
        height=900,
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=30, t=60, b=40),
    )
    fig.update_xaxes(gridcolor="#1e222d", rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(gridcolor="#1e222d")
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    fig.write_html(str(outfile), include_plotlyjs="cdn")
    print(f"  saved {outfile}")


def main():
    import yfinance as yf

    outdir = ROOT / "docs" / "charts"
    outdir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("005930.KS", "삼성전자", "2020-01-01", "2025-12-31"),
        ("^GSPC", "S&P 500", "2020-01-01", "2025-12-31"),
        ("^KQ11", "KOSDAQ", "2020-01-01", "2025-12-31"),
    ]
    for tkr, title, start, end in targets:
        print(f"[{title}]")
        df = yf.download(
            tkr, start=start, end=end, interval="1d", progress=False, auto_adjust=True
        )
        if df is None or len(df) < 300:
            print("  skip")
            continue
        safe = tkr.replace("^", "").replace(".", "_")
        make_chart(df, title, outdir / f"{safe}_smoothing.html")

    print(f"\n열기: {outdir}")


if __name__ == "__main__":
    raise SystemExit(main())
