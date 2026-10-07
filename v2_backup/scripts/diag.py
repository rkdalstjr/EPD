import sys, inspect
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "epd_core" / "src"))

import epd_core
from epd_core import compute_epd
from epd_core.epd import compute_epd as fn

print("=== 1. import 경로 ===")
print("epd_core.__file__ :", epd_core.__file__)
print("epd_core.__version__:", epd_core.__version__)
print("compute_epd 파일  :", inspect.getsourcefile(fn))

print()
print("=== 2. compute_epd 소스 앞 15줄 ===")
for i, line in enumerate(inspect.getsource(fn).splitlines()[:15], 1):
    print(f"{i:2d} | {line}")

print()
print("=== 3. 알려진 입력으로 결과 확인 ===")
import numpy as np
# 상수 추세 데이터: 매일 +0.1% 증가
n = 400
close = 100.0 * np.exp(0.001 * np.arange(n))
out = fn(close)
print("epd_100 마지막 5개:", out["epd_100"][-5:])
print("epd_raw 마지막 5개:", out["epd_raw"][-5:])
print("params:", out["params"])
