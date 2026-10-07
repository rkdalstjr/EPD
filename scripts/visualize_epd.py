"""EPD 시각화 — TradingView 스타일 (plotly).

원래 EPD 정의 유지:
  EPD_raw = Z_r - Z_e
  EPD_100 = 50 * (1 + tanh(EPD_raw / 2))  in (0, 100)

출력: docs/charts/*.html (브라우저로 열기)
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))

from epd_core import rolling_pe, rolling_zscore  # noqa: E402

# ---- 고정 파라미터 (Stage 0 원래 정의) ----
W_PE, M, TAU, W_Z = 60, 3, 1, 252
TANH_SCALE = 2.0


def compute_epd(close):
    """원래 EPD: EPD_100, |EPD_raw| 둘 다 반환."""
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
    epd_100 = 50.0 * (1.0 + np.tanh(raw / TANH_SCALE))
    return epd_100, np.abs(raw)


def make_chart(df, title, outfile):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # MultiIndex 안전 처리
    o = df["Open"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    c = df["Close"].squeeze()
    dates = df.index

    close_np = c.to_numpy(dtype=float)
    epd_100, epd_abs = compute_epd(close_np)

    # 2행 subplot: 가격 / EPD
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.65, 0.35],
        subplot_titles=(title, "EPD (Entropy-Price Divergence)"),
    )

    # ---- 캔들스틱 (TradingView 색상) ----
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

    # ---- EPD_100 (0~100) ----
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=epd_100,
            mode="lines",
            name="EPD_100",
            line=dict(color="#ffeb3b", width=1.5),
        ),
        row=2,
        col=1,
    )

    # ---- 임계선 (70/50/30) ----
    for y, color, dash in [
        (70, "#ef5350", "dash"),
        (50, "#787b86", "dot"),
        (30, "#26a69a", "dash"),
    ]:
        fig.add_hline(y=y, line=dict(color=color, dash=dash, width=1), row=2, col=1)

    # ---- 스타일 (TradingView 다크) ----
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(color="#d1d4dc", size=11, family="Arial"),
        height=850,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=30, t=60, b=40),
        hovermode="x unified",
    )

    fig.update_xaxes(
        gridcolor="#1e222d",
        showgrid=True,
        rangebreaks=[dict(bounds=["sat", "mon"])],
    )
    fig.update_yaxes(gridcolor="#1e222d", showgrid=True)
    fig.update_yaxes(range=[0, 100], row=2, col=1, tickvals=[0, 30, 50, 70, 100])

    fig.write_html(str(outfile), include_plotlyjs="cdn")
    print(f"  saved {outfile}")


def main():
    import yfinance as yf

    outdir = ROOT / "docs" / "charts"
    outdir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("005930.KS", "삼성전자 (005930.KS)", "2020-01-01", "2025-12-31"),
        ("^GSPC", "S&P 500 (^GSPC)", "2020-01-01", "2025-12-31"),
        ("^KQ11", "KOSDAQ (^KQ11)", "2020-01-01", "2025-12-31"),
        ("035420.KS", "NAVER (035420.KS)", "2020-01-01", "2025-12-31"),
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
                print("  skip (insufficient)")
                continue
            safe = tkr.replace("^", "").replace(".", "_")
            outfile = outdir / f"{safe}_epd.html"
            make_chart(df, title, outfile)
        except Exception as e:
            print(f"  err: {e}")

    print(f"\n브라우저로 여세요: {outdir}")


if __name__ == "__main__":
    raise SystemExit(main())
