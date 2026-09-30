# =====================================================================
# JWCC v2 — A1-R: "consistent" perturbatsiyaning ishonchliligi
# jwcc_v2_A1_A2.py dan KEYIN, xuddi shu Colab sessiyasida ishga tushiring.
#
# 1) Qor moduli parametrlari: DDF ∈ {2,4,6}, T50 ∈ {0,1,2} → 9 variant
# 2) "empirical": erish vaqti siljishi degree-day modelsiz, ERA5 ning o'zidan —
#    yillik erish markazi (dekada) ~ bahorgi harorat regressiyasi qiyaligi.
#    Benchmark (DD) bilan aylanma mantiq bo'lmasligi uchun.
# Benchmark qiymatlari (DD_cal, DD_uncal) oldingi katak natijasidan olinadi.
# =====================================================================
assert "fit_models" in globals(), "Avval asosiy katakni (jwcc_v2_A1_A2.py) ishga tushiring"
import itertools, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
T1 = time.time()

SENS_GRID = list(itertools.product([2.0, 4.0, 6.0], [0.0, 1.0, 2.0]))
SCEN_T = [(1, 1.0, "T+1"), (2, 1.0, "T+2"), (3, 1.0, "T+3")]
SRC = ["OS_xgb", "OS_ens", "AR_xgb", "AR_ens", "NQ_xgb"]

R0 = pd.read_csv(f"{OUT}/A1A2_responses.csv")
REF = (R0[(R0.pert_mode == "consistent") & R0.source.isin(["DD_cal", "DD_uncal"])]
       .pivot_table(index=["gauge", "scenario"], columns="source", values="seas"))

def roll_delta(d, p):
    for v, how in (("t2m", "mean"), ("prcp", "sum"), ("melt", "sum")):
        c = f"{v}_roll3"
        if c in p.columns and v in p.columns:
            r0 = getattr(d[v].rolling(3, min_periods=1), how)()
            r1 = getattr(p[v].rolling(3, min_periods=1), how)()
            p[c] = d[c] + (r1 - r0)
    return p

def melt_sensitivity(d):
    """Erish markazining bahorgi haroratga qiyaligi, dekada/K (odatda manfiy)."""
    ph = d["phase"].values; p0 = ph.min()
    spring = (ph >= p0 + 3) & (ph <= p0 + 17)                 # ~ fevral–iyun
    pts = []
    for yr, g in d.groupby(d["date"].dt.year):
        mm = g["melt"].values
        if len(g) < 36 or mm.sum() <= 0: continue
        cen = (np.arange(len(mm)) * mm).sum() / mm.sum()
        pts.append((d.loc[g.index[spring[g.index]], "t2m"].mean(), cen))
    a = np.array(pts)
    return min(np.polyfit(a[:, 0], a[:, 1], 1)[0], 0.0)

def perturb_empirical(d, dT, fP, s_melt):
    p = d.copy(); t0 = d["t2m"].values; t1 = t0 + dT
    p["t2m"] = t1; p["prcp"] = d["prcp"] * fP
    f0 = 1 / (1 + np.exp((t0 - 1.0) / 1.5)); f1 = 1 / (1 + np.exp((t1 - 1.0) / 1.5))
    r = np.where(f0 > 1e-6, f1 / f0, 1.0)
    for v in ("snowf", "snow"):
        if v in p: p[v] = d[v] * fP * r
    idx = np.arange(len(d)); lag = s_melt * dT              # manfiy → erta erish
    for v in ("melt", "swe_dif"):
        if v in p: p[v] = fP * np.interp(idx - lag, idx, d[v].values)
    if "ddsum" in p:
        pos0, pos1 = np.maximum(t0, 0), np.maximum(t1, 0)
        p["ddsum"] = d["ddsum"].values + _scale(d["ddsum"], pos0) * (pos1 - pos0)
    if "pet" in p: p["pet"] = d["pet"] * (1 + 0.06 * dT)
    return roll_delta(d, p)

def sim_subset(M, p, qobs, ev_idx):
    o = one_step(M, p, qobs)
    return dict(OS_xgb=o["xgb"], OS_ens=o["ens"],
                AR_xgb=rollout(M, p, qobs, ev_idx, "xgb", False),
                AR_ens=rollout(M, p, qobs, ev_idx, "ens", False),
                NQ_xgb=M["nq"].predict(build_X(p, use_q=False).values))

rows, slopes = [], {}
_snow_orig = snow_module
try:
    for gid, name in GAUGES.items():
        tb = time.time()
        if not os.path.exists(f"{DATA_DIR}/basin_{gid}.csv"): continue
        d = load_basin(gid)
        qobs = d["q"].values.astype(float); ph = d["phase"].values
        ev = ((d["date"] >= EV_START) & (d["date"] <= EV_END)).values
        ev_idx = np.where(ev)[0]; evm = ev & np.isfinite(qobs)
        pmm = 1000.0 if d.groupby(d["date"].dt.year)["prcp"].sum().median() < 20 else 1.0
        M = fit_models(d)
        base = sim_subset(M, d, qobs, ev_idx)
        s = melt_sensitivity(d); slopes[name] = s
        variants = [(f"DD{a:g}_T{b:g}", ("dd", a, b)) for a, b in SENS_GRID] + [("empirical", ("emp", s))]
        for vname, spec in variants:
            for dT, fP, lab in SCEN_T:
                if spec[0] == "dd":
                    globals()["snow_module"] = (lambda T, P, _a=spec[1], _b=spec[2]:
                                                _snow_orig(T, P, DDF=_a, T50=_b))
                    p = perturb_consistent(d, dT, fP, pmm)
                else:
                    p = perturb_empirical(d, dT, fP, spec[1])
                sim = sim_subset(M, p, qobs, ev_idx)
                for src in SRC:
                    rows.append(dict(gauge=gid, basin=name, variant=vname, scenario=lab, source=src,
                                     seas=seas_shift(sim[src], base[src], ph, evm)))
        print(f"{name:10s} tayyor  erish qiyaligi={s:+.2f} dekada/K  ({time.time() - tb:.0f} s)")
finally:
    globals()["snow_module"] = _snow_orig

SR = pd.DataFrame(rows)
SR = SR.join(REF, on=["gauge", "scenario"])
SR["ratio_cal"] = SR.seas / SR.DD_cal
SR["ratio_uncal"] = SR.seas / SR.DD_uncal
SR["sign_ok"] = (np.sign(SR.seas) == np.sign(SR.DD_cal)).astype(int)
SR.to_csv(f"{OUT}/A1R_sensitivity.csv", index=False)

SUMR = (SR.groupby(["scenario", "variant", "source"])
        .agg(seas_med=("seas", "median"), ratio_cal=("ratio_cal", "median"),
             ratio_uncal=("ratio_uncal", "median"), sign_ok=("sign_ok", "sum")).reset_index())
SUMR.to_csv(f"{OUT}/A1R_summary.csv", index=False)

# ------------------------------------------------------------------ rasm
t2 = SUMR[SUMR.scenario == "T+2"]
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, src in enumerate(SRC):
    g = t2[t2.source == src]
    dd = g[g.variant != "empirical"].seas_med.values
    ax.scatter(np.full(len(dd), i), dd, color="grey", s=22, alpha=.7, label="DD variantlari" if i == 0 else None)
    em = g[g.variant == "empirical"].seas_med.values
    ax.scatter([i], em, color="tab:red", marker="D", s=50, label="empirik" if i == 0 else None)
ref_cal = R0[(R0.pert_mode == "consistent") & (R0.scenario == "T+2") & (R0.source == "DD_cal")].seas.median()
ref_unc = R0[(R0.pert_mode == "consistent") & (R0.scenario == "T+2") & (R0.source == "DD_uncal")].seas.median()
ax.axhline(ref_cal, color="k", ls="-", lw=1, label=f"DD_cal ({ref_cal:.1f}%)")
ax.axhline(ref_unc, color="k", ls="--", lw=1, label=f"DD_uncal ({ref_unc:.1f}%)")
ax.axhline(0, color="grey", lw=.6)
ax.set_xticks(range(len(SRC))); ax.set_xticklabels(SRC)
ax.set_ylabel("T+2 K mavsumiy siljish, median %"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/figA1R_sensitivity.png", dpi=300); plt.close(fig)

# ------------------------------------------------------------------ hisobot
pd.set_option("display.width", 200, "display.max_columns", 20)
print("\n=== T+2: mavsumiy siljish median % (qator: variant, ustun: rejim) ===")
print(t2.pivot(index="variant", columns="source", values="seas_med")[SRC].round(2).to_string())
print("\n=== T+2: DD_cal ga nisbat (basseynlar mediani) ===")
print(t2.pivot(index="variant", columns="source", values="ratio_cal")[SRC].round(3).to_string())
print("\n=== T+2: ishora mosligi (9 dan) ===")
print(t2.pivot(index="variant", columns="source", values="sign_ok")[SRC].to_string())
rng_ = t2[t2.variant != "empirical"].groupby("source").seas_med.agg(["min", "max"])
print("\n=== 9 ta DD variant bo'yicha oraliq (T+2, median %) ===")
print(rng_.loc[SRC].round(2).to_string())
print("\n=== Erish markazi qiyaligi (dekada/K) ===")
print(pd.Series(slopes).round(2).to_string())
print("\n=== Basseynlar bo'yicha AR_xgb, T+2 (DD4_T1 va empirik) — ishorasi mos kelmaydigan basseyn ===")
bb = SR[(SR.scenario == "T+2") & (SR.source == "AR_xgb") & SR.variant.isin(["DD4_T1", "empirical"])]
print(bb.pivot(index="basin", columns="variant", values="seas").join(
      bb.drop_duplicates("basin").set_index("basin")[["DD_cal", "DD_uncal"]]).round(2).to_string())
print(f"\nNatijalar: {OUT}   |   Vaqt: {(time.time() - T1) / 60:.1f} daqiqa")
