# =====================================================================
# JWCC v2 — B2: HBV + muzlik moduli benchmarki
# Asosiy katak va A1R katagidan KEYIN, xuddi shu sessiyada ishga tushiring.
#
# HBV (qor + tuproq namligi + 2 rezervuarli javob), 6 balandlik zonasi,
# harorat lapse rate 6.5 K/km. Muzlik o'rtacha balandlikdan yuqori zonalarga teng taqsimlanadi; qor
# erigandan keyin muzlik yuzasida muz CFMAX·CFGL tezlikda eriydi (maydon o'zgarmas).
# Ikki variant: HBV_glac (1-jadvaldagi muzlik ulushi) va HBV_noglac (muzlik = 0).
# Har biri 5 ta urug' bilan differential evolution orqali kalibrlanadi
# (1959–1985, NSE mm/dekadada), 1986–1990 da validatsiya qilinadi.
# Har 1-oktabrda muzlik ustidagi qolgan qor firnga o'tadi (qor minoralarining oldini oladi).
# Perturbatsiya: faqat T+dT va P·fP — DD benchmark bilan bir xil.
# =====================================================================
assert "load_basin" in globals(), "Avval asosiy katakni (jwcc_v2_A1_A2.py) ishga tushiring"
import os, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr
try:
    import numba as nb
except ImportError:
    os.system("pip -q install numba"); import numba as nb
T2 = time.time()

N_SEEDS, N_BANDS, LAPSE = 5, 6, 0.0065
# asl maqola, 1-jadval: maydon km², o'rtacha, min, maks balandlik m, muzlik %
ATTR = {16279: (5669, 2693, 945, 4441, 0.50), 16290: (2518, 2795, 896, 4372, 3.21),
        16300: (863, 2046, 755, 3600, 0.00), 16936: (52030, 2849, 861, 5119, 1.96),
        17202: (683, 2666, 915, 4744, 2.80), 17211: (906, 2333, 746, 4115, 0.04),
        17288: (10297, 3110, 1048, 5455, 5.33), 16176: (368, 2797, 1524, 4314, 0.69),
        16202: (351, 2373, 1068, 3424, 0.00)}
PNAMES = ["TT", "CFMAX", "SFCF", "CFGL", "FC", "LP", "BETA", "PERC", "UZL",
          "K0", "K1", "K2", "PCORR", "ETF"]
BOUNDS = [(-2.0, 2.0), (1.5, 8.0), (0.6, 1.5), (1.0, 2.5), (50, 600), (0.3, 1.0), (1.0, 6.0),
          (0.0, 30.0), (0.0, 100.0), (0.1, 0.9), (0.02, 0.6), (0.002, 0.2), (0.5, 2.5), (0.05, 0.4)]

@nb.njit(cache=True)
def hbv(T, P, nd, rs, bz, bw, bg, par, lapse):
    TT, CFMAX, SFCF, CFGL, FC, LP, BETA, PERC, UZL, K0, K1, K2, PCORR, ETF = par
    CFR, CWH, TTI = 0.05, 0.1, 2.0
    nbd = bz.shape[0]; n = T.shape[0]
    sp = np.zeros((nbd, 2)); wc = np.zeros((nbd, 2)); sm = np.full(nbd, 0.5 * FC)
    suz = 0.0; slz = 20.0
    Q = np.zeros(n); ICE = np.zeros(n)
    for t in range(n):
        d = nd[t]; rech = 0.0; gq = 0.0; icet = 0.0
        if rs[t]:                                   # 1-oktabr: muzlikda qor → firn, boshqa joyda qor minorasi cheklanadi
            for i in range(nbd):
                sp[i, 1] = 0.0; wc[i, 1] = 0.0
                sp[i, 0] = min(sp[i, 0], 500.0)
        for i in range(nbd):
            Ti = T[t] - lapse * bz[i]; Pi = P[t] * PCORR
            if Ti <= TT - TTI / 2: fs = 1.0
            elif Ti >= TT + TTI / 2: fs = 0.0
            else: fs = (TT + TTI / 2 - Ti) / TTI
            for j in range(2):
                a = bw[i] * (bg[i] if j == 1 else 1.0 - bg[i])
                if a <= 0.0: continue
                sp[i, j] += Pi * fs * SFCF
                if Ti > TT:
                    m = min(CFMAX * d * (Ti - TT), sp[i, j]); sp[i, j] -= m; wc[i, j] += m
                else:
                    r = min(0.05 * CFMAX * d * (TT - Ti), wc[i, j]); sp[i, j] += r; wc[i, j] -= r
                wc[i, j] += Pi * (1.0 - fs)
                out = max(wc[i, j] - CWH * sp[i, j], 0.0); wc[i, j] -= out
                if j == 1:
                    ice = CFGL * CFMAX * d * max(Ti - TT, 0.0) if sp[i, j] < 1.0 else 0.0
                    gq += a * (out + ice); icet += a * ice
                else:
                    ratio = min(sm[i] / FC, 1.0)
                    rr = out * ratio ** BETA
                    sm[i] += out - rr
                    ea = min(ETF * d * max(Ti, 0.0) * min(sm[i] / (FC * LP), 1.0), sm[i])
                    sm[i] -= ea
                    if sm[i] > FC:
                        rr += sm[i] - FC; sm[i] = FC
                    rech += a * rr
        suz += rech + gq
        perc = min(PERC * d / 10.0, suz); suz -= perc; slz += perc
        q = min(K0 * max(suz - UZL, 0.0) + K1 * suz, suz); suz -= q
        q2 = K2 * slz; slz -= q2
        Q[t] = q + q2; ICE[t] = icet
    return Q, ICE

def make_bands(zmean, zmin, zmax, glac_pct):
    e = np.linspace(zmin, zmax, N_BANDS + 1); z = (e[:-1] + e[1:]) / 2
    w = np.exp(-0.5 * ((z - zmean) / ((zmax - zmin) / 4)) ** 2); w /= w.sum()
    up = z >= zmean                                  # muzlik o'rtacha balandlikdan yuqoridagi zonalarda
    g = np.where(up, min(glac_pct / 100 / w[up].sum(), 1.0), 0.0)
    return z - (w * z).sum(), w, g          # ERA5 havza o'rtachasi = zonalar o'rtachasi

def kge(y, yh):
    m = np.isfinite(y) & np.isfinite(yh); y, yh = y[m], yh[m]
    r = np.corrcoef(y, yh)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (yh.std() / y.std() - 1) ** 2 + (yh.mean() / y.mean() - 1) ** 2)

# ------------------------------------------------------------------ asosiy sikl
rows, skill = [], []
for gid, name in GAUGES.items():
    tb = time.time()
    if not os.path.exists(f"{DATA_DIR}/basin_{gid}.csv"): continue
    d = load_basin(gid)
    area, zm, zmin, zmax, gl = ATTR[gid]
    nd = np.where(d.date.dt.day < 21, 10, d.date.dt.days_in_month - 20).astype(float)
    rs = ((d.date.dt.month == 10) & (d.date.dt.day == 1)).values
    qmm = d.q.values * 86400 * nd / (area * 1e3)                     # m³/s → mm/dekada
    pmm = 1000.0 if d.groupby(d.date.dt.year).prcp.sum().median() < 20 else 1.0
    T = d.t2m.values.astype(float); P = d.prcp.values.astype(float) * pmm
    ph = d.phase.values
    cal = ((d.date >= CAL_START) & (d.date <= CAL_END)).values.copy(); cal[:SPINUP] = False
    ev = ((d.date >= EV_START) & (d.date <= EV_END)).values
    calm = cal & np.isfinite(qmm); evm = ev & np.isfinite(qmm)
    summ = (ph >= SUMMER[0]) & (ph <= SUMMER[1])
    for var, glp in (("HBV_glac", gl), ("HBV_noglac", 0.0)):
        bz, bw, bg = make_bands(zm, zmin, zmax, glp)
        yo = qmm[calm]; den = ((yo - yo.mean()) ** 2).sum()
        obj = lambda x: ((yo - hbv(T, P, nd, rs, bz, bw, bg, x, LAPSE)[0][calm]) ** 2).sum() / den
        for seed in range(N_SEEDS):
            res = differential_evolution(obj, BOUNDS, seed=SEED + seed, maxiter=150, popsize=12,
                                         tol=1e-7, polish=False, init="latinhypercube")
            x = res.x
            qb, ice = hbv(T, P, nd, rs, bz, bw, bg, x, LAPSE)
            skill.append(dict(gauge=gid, basin=name, variant=var, seed=seed,
                              NSE_cal=1 - res.fun, NSE_val=nse(qmm[evm], qb[evm]), KGE_val=kge(qmm[evm], qb[evm]),
                              ice_share_summer=100 * ice[evm & summ].sum() / qb[evm & summ].sum(),
                              **dict(zip(PNAMES, x))))
            for dT, fP, lab in SCENARIOS:
                qs, _ = hbv(T + dT, P * fP, nd, rs, bz, bw, bg, x, LAPSE)
                rows.append(dict(gauge=gid, basin=name, variant=var, seed=seed, scenario=lab,
                                 vol=vol_resp(qs, qb, evm), seas=seas_shift(qs, qb, ph, evm)))
    sk = pd.DataFrame(skill); s1 = sk[(sk.gauge == gid)]
    print(f"{name:10s} tayyor  NSE val glac/noglac = "
          f"{s1[s1.variant=='HBV_glac'].NSE_val.median():.2f}/{s1[s1.variant=='HBV_noglac'].NSE_val.median():.2f}"
          f"  muz ulushi (yoz) = {s1[s1.variant=='HBV_glac'].ice_share_summer.median():.1f}%  ({time.time()-tb:.0f} s)")

H = pd.DataFrame(rows); SK = pd.DataFrame(skill)
H.to_csv(f"{OUT}/B2_hbv_responses.csv", index=False); SK.to_csv(f"{OUT}/B2_hbv_skill.csv", index=False)

# ------------------------------------------------------------------ taqqoslash
Hm = H.groupby(["gauge", "basin", "scenario", "variant"]).agg(
        seas=("seas", "median"), seas_lo=("seas", "min"), seas_hi=("seas", "max"),
        vol=("vol", "median")).reset_index()
Hp = Hm.pivot_table(index=["gauge", "scenario"], columns="variant", values=["seas", "vol"])
Hp.columns = [f"{a}_{b}" for a, b in Hp.columns]

R0 = pd.read_csv(f"{OUT}/A1A2_responses.csv"); R0 = R0[R0.pert_mode == "consistent"]
A = R0.pivot_table(index=["gauge", "scenario"], columns="source", values=["seas", "vol"])
A = A[[c for c in A.columns if c[1] in ("AR_xgb", "OS_ens", "DD_cal", "DD_uncal")]]
A.columns = [f"{a}_{b}" for a, b in A.columns]
C = A.join(Hp)
if os.path.exists(f"{OUT}/A1R_sensitivity.csv"):
    SR = pd.read_csv(f"{OUT}/A1R_sensitivity.csv")
    e = SR[(SR.variant == "empirical") & (SR.source == "AR_xgb")].set_index(["gauge", "scenario"]).seas
    C = C.join(e.rename("seas_AR_xgb_emp"))
C = C.reset_index()
C["basin"] = C.gauge.map(GAUGES); C["glacier_%"] = C.gauge.map(lambda g: ATTR[g][4])
C.to_csv(f"{OUT}/B2_comparison.csv", index=False)

# ------------------------------------------------------------------ rasm
t2 = C[C.scenario == "T+2"].sort_values("glacier_%")
cols = [("seas_AR_xgb", "AR_xgb (model)"), ("seas_DD_cal", "DD_cal"),
        ("seas_HBV_noglac", "HBV muzliksiz"), ("seas_HBV_glac", "HBV muzlikli")]
fig, ax = plt.subplots(figsize=(11, 4.5)); wdt = 0.2; xx = np.arange(len(t2))
for k, (c, lab) in enumerate(cols):
    ax.bar(xx + (k - 1.5) * wdt, t2[c], wdt, label=lab)
lo = Hm[(Hm.scenario == "T+2") & (Hm.variant == "HBV_glac")].set_index("gauge").reindex(t2.gauge)
ax.errorbar(xx + 1.5 * wdt, t2["seas_HBV_glac"], yerr=[t2["seas_HBV_glac"].values - lo.seas_lo.values,
            lo.seas_hi.values - t2["seas_HBV_glac"].values], fmt="none", ecolor="k", lw=1)
ax.set_xticks(xx); ax.set_xticklabels([f"{b}\n{g:.1f}%" for b, g in zip(t2.basin, t2["glacier_%"])], fontsize=8)
ax.axhline(0, color="grey", lw=.6); ax.set_ylabel("T+2 K mavsumiy siljish, %")
ax.set_title("Basseynlar muzlik ulushi bo'yicha tartiblangan (xato chiziqlari: 5 urug' oralig'i)", fontsize=10)
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/figB2_hbv.png", dpi=300); plt.close(fig)

# ------------------------------------------------------------------ hisobot
pd.set_option("display.width", 220, "display.max_columns", 30)
print("\n=== Kalibratsiya sifati (5 urug' mediani) ===")
print(SK.groupby(["basin", "variant"])[["NSE_cal", "NSE_val", "KGE_val", "ice_share_summer"]]
      .median().unstack("variant").round(2).to_string())
show = ["basin", "glacier_%", "seas_AR_xgb", "seas_AR_xgb_emp", "seas_DD_cal", "seas_DD_uncal",
        "seas_HBV_noglac", "seas_HBV_glac"]
show = [c for c in show if c in t2.columns]
print("\n=== T+2 K: mavsumiy siljish % (muzlik bo'yicha tartiblangan) ===")
print(t2[show].round(2).to_string(index=False))
print("\n=== Median va model/benchmark nisbati (T+2, basseynlar mediani) ===")
for ref in ("seas_DD_cal", "seas_DD_uncal", "seas_HBV_noglac", "seas_HBV_glac"):
    r = (t2.seas_AR_xgb / t2[ref])
    gl_lo, gl_hi = t2["glacier_%"] < 1, t2["glacier_%"] >= 1
    print(f"{ref:17s} median={t2[ref].median():6.2f}%  nisbat: hammasi={r.median():.2f}  "
          f"muzlik<1%={r[gl_lo].median():.2f}  muzlik≥1%={r[gl_hi].median():.2f}  "
          f"ishora={int((np.sign(t2.seas_AR_xgb)==np.sign(t2[ref])).sum())}/9")
print("\n=== Muzlik ulushi bilan Spearman ρ (T+2) ===")
for c in ["seas_AR_xgb", "seas_DD_cal", "seas_HBV_noglac", "seas_HBV_glac"]:
    r, p = spearmanr(t2["glacier_%"], t2[c]); print(f"{c:17s} ρ={r:+.2f}  p={p:.3f}")
print("\n=== Barcha ssenariylar: median % (mavsumiy siljish | hajm) ===")
agg = C.groupby("scenario")[[c for c in C.columns if c.startswith(("seas_", "vol_"))]].median()
print(agg.round(2).T.to_string())
print(f"\nNatijalar: {OUT}/B2_*.csv, figB2_hbv.png   |   Vaqt: {(time.time()-T2)/60:.1f} daqiqa")
