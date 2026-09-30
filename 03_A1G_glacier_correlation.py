# A1-G: Spearman correlation of the +2 K seasonal-shift responses with basin attributes (Table 1).
# Run after 02_A1R_perturbation_robustness.py; reads results/A1R_sensitivity.csv (or the Drive copy).
import os, sys, pandas as pd
from scipy.stats import spearmanr
OUT = sys.argv[1] if len(sys.argv) > 1 else ("/content/drive/MyDrive/jwcc_v2" if os.path.exists("/content/drive") else "../results")
ATTR = pd.DataFrame({  # Table 1 of the paper
    "gauge": [16279, 16290, 16300, 16936, 17202, 17211, 17288, 16176, 16202],
    "glacier_pct": [0.50, 3.21, 0.00, 1.96, 2.80, 0.04, 5.33, 0.69, 0.00],
    "elev_mean_m": [2693, 2795, 2046, 2849, 2666, 2333, 3110, 2797, 2373],
    "snowfall_pct": [62.1, 66.1, 51.5, 43.8, 62.3, 48.0, 64.2, 39.7, 41.1]}).set_index("gauge")
SR = pd.read_csv(f"{OUT}/A1R_sensitivity.csv")
t = SR[(SR.scenario == "T+2") & (SR.source == "AR_xgb") & (SR.variant == "empirical")].set_index("gauge")
T = ATTR.join(t[["basin", "seas", "DD_cal", "DD_uncal"]])
T["ratio_cal"] = T.seas / T.DD_cal; T["gap_cal"] = T.DD_cal - T.seas
for y in ["seas", "DD_cal", "DD_uncal", "ratio_cal", "gap_cal"]:
    print(y.ljust(10), "  ".join(f"{x}: rho={spearmanr(T[x], T[y])[0]:+.2f} (p={spearmanr(T[x], T[y])[1]:.3f})"
                                  for x in ATTR.columns))
