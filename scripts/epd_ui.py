"""EPD 나비형 UI — 최종 버전.

레이아웃:
  [1] 가격 (캔들스틱)              — 배경에 |Z_r - Z_e| 히트맵
  [2] 나비 패널                    — Z_r (위, 파랑) vs -Z_e (아래, 주황)
  [3] EPD_100                      — raw + EMA14, 30/50/70 밴드

사용:
  python scripts/epd_ui.py              # 기본 (4자산, 2020-2025)
  python scripts/epd_ui.py --ticker 005930.KS --start 2023-01-01
"""

from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))
from epd_core import rolling_pe, rolling_zscore  # noqa: E402

# ---- 고정 파라미터 ----
W_PE, M, TAU, W_Z = 60, 3, 1, 252
TANH_SCALE = 2.0
EMA_SPAN = 14

# ---- 색상 팔레트 (다크) ----
C_BG = "#0d1117"
C_GRID = "#1a1e26"
C_TEXT = "#d1d4dc"
C_UP = "#26a69a"
C_DOWN = "#ef5350"
C_ZR = "#4fc3f7"  # Z_r 파랑
C_ZE = "#ff7043"  # -Z_e 주황
C_EPD_RAW = "#5a5a5a"
C_EPD_EMA = "#ffeb3b"  # EPD 스무딩 노랑
C_THR_HIGH = "#ef5350"
C_THR_MID = "#787b86"
C_THR_LOW = "#26a69a"


def ema(x, span):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    alpha = 2.0 / (span + 1.0)
    prev = np.nan
    for t in range(len(x)):
        xt = x[t]
        if not np.isfinite(xt):
            out[t] = prev
        else:
            prev = xt if not np.isfinite(prev) else alpha * xt + (1 - alpha) * prev
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
        "log_P": log_P,
        "pe": pe,
        "dpe": dpe,
        "z_r": z_r,
        "z_e": z_e,
        "raw": raw,
        "ema14": ema14,
        "epd_raw_100": epd_raw_100,
        "epd_ema_100": epd_ema_100,
    }


def make_chart(df, title, outfile):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    o = df["Open"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    c = df["Close"].squeeze()
    dates = df.index
    sig = compute_all(c.to_numpy(dtype=float))

    # --- |raw| 히트맵 강도 (0~2.5 클립) ---
    intensity = np.clip(np.abs(sig["raw"]), 0, 2.5)
    y_min = float(np.nanmin(l)) * 0.97
    y_max = float(np.nanmax(h)) * 1.03

    # --- 레이아웃 ---
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.028,
        row_heights=[0.42, 0.28, 0.30],
        subplot_titles=(
            f"{title}",
            "Z_r (수익률, 위)  vs  −Z_e (엔트로피 변화, 아래)",
            "EPD_100  =  50 · (1 + tanh((Z_r − Z_e)/2))",
        ),
    )

    # ============================================================
    # [1] 가격 + 히트맵 배경
    # ============================================================
    colorscale_tension = [
        [0.00, "rgba(38,166,154,0.00)"],
        [0.30, "rgba(38,166,154,0.08)"],
        [0.50, "rgba(255,235,59,0.14)"],
        [0.70, "rgba(255,112,67,0.28)"],
        [1.00, "rgba(239,83,80,0.50)"],
    ]
    fig.add_trace(
        go.Heatmap(
            x=dates,
            y=np.array([y_min, y_max]),
            z=np.vstack([intensity, intensity]),
            colorscale=colorscale_tension,
            showscale=True,
            colorbar=dict(
                title=dict(text="|EPD<sub>raw</sub>|", font=dict(size=11)),
                x=1.01,
                len=0.35,
                y=0.83,
                tickfont=dict(size=10),
                tickvals=[0, 0.5, 1.0, 1.5, 2.0, 2.5],
            ),
            hovertemplate="|EPD|=%{z:.2f}<extra></extra>",
            zmin=0,
            zmax=2.5,
            name="|EPD|",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=o,
            high=h,
            low=l,
            close=c,
            name="Price",
            increasing_line_color=C_UP,
            increasing_fillcolor=C_UP,
            decreasing_line_color=C_DOWN,
            decreasing_fillcolor=C_DOWN,
            line=dict(width=0.8),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # ============================================================
    # [2] 나비 패널: Z_r (위), -Z_e (아래)
    # ============================================================
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sig["z_r"],
            mode="lines",
            name="Z_r (수익률)",
            line=dict(color=C_ZR, width=1.3),
            fill="tozeroy",
            fillcolor="rgba(79,195,247,0.22)",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=-sig["z_e"],
            mode="lines",
            name="−Z_e (엔트로피 변화)",
            line=dict(color=C_ZE, width=1.3),
            fill="tozeroy",
            fillcolor="rgba(255,112,67,0.22)",
        ),
        row=2,
        col=1,
    )

    # 0선과 ±2σ 참조선
    fig.add_hline(y=0, line=dict(color="#555", width=0.8), row=2, col=1)
    for y in [2, -2]:
        fig.add_hline(y=y, line=dict(color="#444", dash="dot", width=0.6), row=2, col=1)

    # ============================================================
    # [3] EPD_100
    # ============================================================
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sig["epd_raw_100"],
            mode="lines",
            name="EPD_raw",
            line=dict(color=C_EPD_RAW, width=0.7),
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
            line=dict(color=C_EPD_EMA, width=2.0),
        ),
        row=3,
        col=1,
    )

    for y, col, lbl in [
        (70, C_THR_HIGH, "과열"),
        (50, C_THR_MID, "중립"),
        (30, C_THR_LOW, "구조형성"),
    ]:
        fig.add_hline(y=y, line=dict(color=col, dash="dash", width=1), row=3, col=1)

    # ============================================================
    # 레이아웃 스타일
    # ============================================================
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C_BG,
        plot_bgcolor=C_BG,
        font=dict(color=C_TEXT, size=11, family="Segoe UI, Arial"),
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=90, t=80, b=40),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(20,24,32,0.95)", font=dict(color=C_TEXT, size=11)
        ),
    )
    fig.update_xaxes(
        gridcolor=C_GRID,
        showgrid=True,
        rangebreaks=[dict(bounds=["sat", "mon"])],
        zeroline=False,
    )
    fig.update_yaxes(gridcolor=C_GRID, showgrid=True, zeroline=False)
    fig.update_yaxes(range=[-4, 4], row=2, col=1, tickvals=[-4, -2, 0, 2, 4])
    fig.update_yaxes(range=[0, 100], row=3, col=1, tickvals=[0, 30, 50, 70, 100])

    # 서브플롯 타이틀 스타일
    for ann in fig.layout.annotations[:3]:
        ann.font.size = 12
        ann.font.color = C_TEXT
        ann.font.family = "Segoe UI, Arial"

    fig.write_html(
        str(outfile),
        include_plotlyjs="cdn",
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    print(f"  saved {outfile}")


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ticker", type=str, default=None, help="단일 종목 (예: 005930.KS)"
    )
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default="2025-12-31")
    args = ap.parse_args()

    import yfinance as yf

    outdir = ROOT / "docs" / "charts"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.ticker:
        targets = [(args.ticker, args.ticker, args.start, args.end)]
    else:
        targets = [
            ("005930.KS", "삼성전자 (005930.KS)", args.start, args.end),
            ("^GSPC", "S&P 500 (^GSPC)", args.start, args.end),
            ("^KQ11", "KOSDAQ (^KQ11)", args.start, args.end),
            ("^KS11", "KOSPI (^KS11)", args.start, args.end),
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
            out = outdir / f"{safe}_epd.html"
            make_chart(df, title, out)
        except Exception as e:
            print(f"  err: {e}")

    print(f"\n브라우저로 여세요:")
    for tkr, _, _, _ in targets:
        safe = tkr.replace("^", "").replace(".", "_")
        print(f"  start docs\\charts\\{safe}_epd.html")


if __name__ == "__main__":
    raise SystemExit(main())
