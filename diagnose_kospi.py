"""KOSPI 데이터 이상 진단."""

import numpy as np
import yfinance as yf

print("=" * 70)
for tkr in ["^KS11", "^GSPC", "^IXIC"]:
    print(f"\n=== {tkr} ===")
    df = yf.download(
        tkr,
        start="2005-01-01",
        end="2025-12-31",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    print(f"  rows={len(df)}")
    print(f"  columns dtype: {df.columns.dtype}")
    print(f"  columns: {list(df.columns)[:5]}")
    print(f"  index: {df.index[0]} ~ {df.index[-1]}")

    close = df["Close"].squeeze().to_numpy(dtype=float)
    print(f"  close shape: {close.shape}")
    print(f"  finite close: {np.isfinite(close).sum()} / {len(close)}")

    r = np.diff(np.log(close[1:]))  # 주의: 여기 실제 스크립트와 동일하게
    print(f"  returns: mean={r.mean():.6f}  std={r.std():.4f}")
    print(f"  extreme |r|>0.3 비율: {(np.abs(r) > 0.3).mean():.4%}")

    # 앞/뒤 값
    print(f"  close[0:5]: {close[:5]}")
    print(f"  close[-5:]: {close[-5:]}")
