"""D 테스트 진단: 후반부에서 r, PE, EPD_raw가 어떻게 움직이는가."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parent / "packages" / "epd_core" / "src")
)
from epd_core import compute_epd  # noqa: E402


def gen_divergence(n=10_000, seed=13):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    half = n // 2
    r[:half] = rng.standard_normal(half)
    t = np.arange(half, n)
    r[half:] = (
        2.0 + 0.5 * np.sin(2 * np.pi * t / 200) + rng.standard_normal(n - half) * 0.02
    )
    return r


r = gen_divergence()
out = compute_epd(r)
half = len(r) // 2

print("=" * 90)
print(
    f"{'name':12s} | {'first_half mean':>16s} {'std':>8s} | {'second_half mean':>17s} {'std':>8s}"
)
print("=" * 90)
for name in ["return_z", "entropy_z", "epd_raw", "epd_100", "pe", "dpe"]:
    v = out[name]
    f = v[:half][np.isfinite(v[:half])]
    s = v[half:][np.isfinite(v[half:])]
    print(
        f"{name:12s} | {f.mean():16.4f} {f.std():8.4f} | "
        f"{s.mean():17.4f} {s.std():8.4f}"
    )

print()
print("--- 후반부 초기 500 샘플 스킵 후 (안정 상태) ---")
skip = half + 500
for name in ["return_z", "entropy_z", "epd_raw", "epd_100", "pe", "dpe"]:
    v = out[name][skip:]
    v = v[np.isfinite(v)]
    print(
        f"{name:12s} | mean={v.mean():7.4f}  std={v.std():6.4f}  "
        f"p5={np.percentile(v,5):7.3f}  p95={np.percentile(v,95):7.3f}"
    )

print()
print("--- 원시 가격/수익 통계 ---")
print(f"  r first half: mean={r[:half].mean():.4f}  std={r[:half].std():.4f}")
print(f"  r second half: mean={r[half:].mean():.4f}  std={r[half:].std():.4f}")
pe_f = out["pe"][:half]
pe_f = pe_f[np.isfinite(pe_f)]
pe_s = out["pe"][half:]
pe_s = pe_s[np.isfinite(pe_s)]
print(f"  PE first:  mean={pe_f.mean():.4f}  std={pe_f.std():.4f}")
print(f"  PE second: mean={pe_s.mean():.4f}  std={pe_s.std():.4f}")
