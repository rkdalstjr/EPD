"""EPD 독자 UI — 두 가지 스타일.

A) Divergence Dashboard: 가격 / Z_r vs Z_e 나비 / EPD
B) Structural Tension: 가격 + |EPD| 배경 히트맵 + EPD 라인
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import rolling_pe, rolling_zscore  # noqa: E402

W_PE, M, TAU, W_Z = 60, 3, 1, 252
TANH_SCALE = 2.0
EMA_SPAN = 14


def ema(x, span):
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


def compute_all(close):
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
    ema14 = ema(raw, EMA_SPAN)
    epd_raw_100 = 50.0 * (1.0 + np.tanh(raw / TANH_SCALE))
    epd_ema_100 = 50.0 * (1.0 + np.tanh(ema14 / TANH_SCALE))

    return {
        "r": r,
        "pe": pe,
        "dpe": dpe,
        "z_r": z_r,
        "z_e": z_e,
        "raw": raw,
        "ema14": ema14,
        "epd_raw_100": epd_raw_100,
        "epd_ema_100": epd_ema_100,
        "epd_abs": np.abs(raw),
    }


# ============================================================
# 스타일 A: Divergence Dashboard
# ============================================================
def chart_divergence(df, title, outfile):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    o = df["Open"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    c = df["Close"].squeeze()
    dates = df.index
    sig = compute_all(c.to_numpy(dtype=float))

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.45, 0.25, 0.30],
        subplot_titles=(
            title,
            "Z_r (수익률)  vs  Z_e (엔트로피 변화)  [나비형]",
            "EPD_100 = 50(1 + tanh((Z_r - Z_e)/2))",
        ),
    )

    # --- 1. 가격 ---
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

    # --- 2. Butterfly: Z_r 위쪽, -Z_e 아래쪽 ---
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sig["z_r"],
            mode="lines",
            name="Z_r",
            line=dict(color="#4fc3f7", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(79,195,247,0.25)",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=-sig["z_e"],
            mode="lines",
            name="-Z_e",
            line=dict(color="#ff7043", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,112,67,0.25)",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color="#555", width=0.8), row=2, col=1)
    for y in [2, -2]:
        fig.add_hline(y=y, line=dict(color="#555", dash="dot", width=0.5), row=2, col=1)

    # --- 3. EPD_100: 원본 얇게 + EMA14 굵게 ---
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sig["epd_raw_100"],
            mode="lines",
            name="EPD_raw",
            line=dict(color="#5a5a5a", width=0.7),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sig["epd_ema_100"],
            mode="lines",
            name="EPD_EMA14",
            line=dict(color="#ffeb3b", width=2.0),
        ),
        row=3,
        col=1,
    )
    for y, col in [(70, "#ef5350"), (50, "#787b86"), (30, "#26a69a")]:
        fig.add_hline(y=y, line=dict(color=col, dash="dash", width=1), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#d1d4dc", size=11),
        height=1000,
        showlegend=True,
        legend=dict(orientation="h", y=-0.04, x=0),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=30, t=60, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#1a1e26", rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(gridcolor="#1a1e26")
    fig.update_yaxes(range=[0, 100], row=3, col=1, tickvals=[0, 30, 50, 70, 100])
    fig.update_yaxes(range=[-4, 4], row=2, col=1)

    fig.write_html(str(outfile), include_plotlyjs="cdn")
    print(f"  saved {outfile}")


# ============================================================
# 스타일 B: Structural Tension (배경 히트맵)
# ============================================================
def chart_tension(df, title, outfile):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    o = df["Open"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    c = df["Close"].squeeze()
    dates = df.index
    sig = compute_all(c.to_numpy(dtype=float))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.68, 0.32],
        subplot_titles=(title, "EPD_100"),
    )

    # --- 배경 히트맵: |EPD_raw| 강도를 색으로 ---
    # x축을 날짜 그대로, y축을 가격 범위로
    y_min = float(l.min()) * 0.98
    y_max = float(h.max()) * 1.02
    intensity = np.abs(sig["raw"])

    # 색상 매핑 (0~2.5 → 투명한 파랑 ~ 진한 빨강)
    colorscale = [
        [0.0, "rgba(38,166,154,0.0)"],  # 낮은 긴장 → 투명
        [0.4, "rgba(38,166,154,0.12)"],  # 초록 약간
        [0.6, "rgba(255,193,7,0.20)"],  # 노랑
        [0.8, "rgba(255,112,67,0.35)"],  # 주황
        [1.0, "rgba(239,83,80,0.55)"],  # 빨강
    ]

    fig.add_trace(
        go.Heatmap(
            x=dates,
            y=np.linspace(y_min, y_max, 2),
            z=np.vstack([np.clip(intensity, 0, 2.5), np.clip(intensity, 0, 2.5)]),
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="|EPD_raw|",
                x=1.02,
                len=0.5,
                y=0.75,
                tickvals=[0, 0.5, 1, 1.5, 2, 2.5],
            ),
            hovertemplate="|EPD_raw|=%{z:.2f}<extra></extra>",
            zmin=0,
            zmax=2.5,
            name="tension",
        ),
        row=1,
        col=1,
    )

    # --- 가격 (히트맵 위에 오버레이) ---
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

    # --- EPD_100 (하단) ---
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sig["epd_raw_100"],
            mode="lines",
            name="EPD_raw",
            line=dict(color="#5a5a5a", width=0.7),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sig["epd_ema_100"],
            mode="lines",
            name="EPD_EMA14",
            line=dict(color="#ffeb3b", width=2.0),
        ),
        row=2,
        col=1,
    )
    for y, col in [(70, "#ef5350"), (50, "#787b86"), (30, "#26a69a")]:
        fig.add_hline(y=y, line=dict(color=col, dash="dash", width=1), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#d1d4dc", size=11),
        height=900,
        showlegend=True,
        legend=dict(orientation="h", y=-0.04, x=0),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=90, t=60, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#1a1e26", rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(gridcolor="#1a1e26")
    fig.update_yaxes(range=[0, 100], row=2, col=1, tickvals=[0, 30, 50, 70, 100])

    fig.write_html(str(outfile), include_plotlyjs="cdn")
    print(f"  saved {outfile}")


def main():
    import yfinance as yf

    outdir = ROOT / "docs" / "charts"
    outdir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("005930.KS", "삼성전자 (005930.KS)", "2020-01-01", "2025-12-31"),
        ("^GSPC", "S&P 500", "2020-01-01", "2025-12-31"),
        ("^KQ11", "KOSDAQ", "2020-01-01", "2025-12-31"),
    ]

    for tkr, title, start, end in targets:
        print(f"[{title}]")
        try:
            df = yf.download(
                tkr,
                start=start,
                end=end,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df is None or len(df) < 300:
                print("  skip")
                continue
            safe = tkr.replace("^", "").replace(".", "_")
            chart_divergence(df, title, outdir / f"{safe}_A_divergence.html")
            chart_tension(df, title, outdir / f"{safe}_B_tension.html")
        except Exception as e:
            print(f"  err: {e}")

    print(f"\n열기:")
    print(f"  start docs\\charts\\005930_KS_A_divergence.html")
    print(f"  start docs\\charts\\005930_KS_B_tension.html")


if __name__ == "__main__":
    raise SystemExit(main())
