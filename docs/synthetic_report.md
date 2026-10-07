# Synthetic Report — Stage 1 (v1.2, log-price)

Fixed params: w_pe=60, m=3, tau=1, w_z=252, tanh_scale=2

## A_white_noise
- ic_ret_5d: 0.008221033521013818
- ic_vol_5d: -0.008767712272743897
- frac_blocks_2sigma: 0.3
- pass: True

## B_garch
- ic_ret_5d: 0.027395819157575996
- ic_vol_5d: 0.03920376299143379
- pass: True

## C_regime
- n_transitions: 177
- hit_rate_within_5d: 0.9717514124293786
- pass: True

## D_divergence
- frac_extreme_event: 0.45
- frac_extreme_bg: 0.2131009867730422
- pass: True

**Overall**: PASS