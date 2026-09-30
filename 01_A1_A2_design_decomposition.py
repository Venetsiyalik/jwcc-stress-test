# =====================================================================
# JWCC v2 — A1 (simulyatsiya rejimlari) + A2 (ansambl ajratish)
# Colab'da BITTA katakka qo'yib ishga tushiring. Taxminan 5–10 daqiqa.
#
# Ma'lumot:  MyDrive/leakfree/basins/basin_<gauge>.csv  (asl maqoladagi)
# Natija:    MyDrive/jwcc_v2/  (CSV jadvallar + PNG rasmlar)
#
# Simulyatsiya rejimlari (A1):
#   OS  — bir qadamli prognoz, lagged Q kuzatilgan (asl maqola dizayni)
#   AR  — to'liq avtoregressiv rollout: lagged Q = modelning o'z bashorati
#   ARy — har yili 1-dekadada kuzatilgan Q bilan qayta ishga tushiriladigan rollout
#   NQ  — lagged Q siz sof iqlim-simulyatsiya modeli
# Ansambl a'zolari (A2): xgb, clim (klimatologiya+AR1), ens (vaznli ansambl)
# Perturbatsiya rejimlari:
#   paper      — asl maqoladagi perturb() funksiyasi aynan
#   consistent — bazaviy qatorni saqlaydigan delta usuli + erish vaqti
#                degree-day qor moduli orqali siljiydi (fizik izchil inputlar)
# Javob = perturbatsiyali run − bazaviy run (bir xil rejimda), shuning uchun
# rollout xatolarining to'planishi ikkala tomonda qisqarib ketadi.
# =====================================================================
import os, sys, time, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
T0 = time.time()

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = "/content/drive/MyDrive"
else:
    ROOT = os.environ.get("JWCC_ROOT", ".")
DATA_DIR = os.environ.get("JWCC_DATA", f"{ROOT}/leakfree/basins")
OUT = os.environ.get("JWCC_OUT", f"{ROOT}/jwcc_v2")
os.makedirs(OUT, exist_ok=True)

try:
    from xgboost import XGBRegressor
except ImportError:
    os.system("pip -q install xgboost")
    from xgboost import XGBRegressor
import matplotlib
if not IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ config
SEED, N_LAGS, VAL_TAIL = 42, 6, 108
CAL_START, CAL_END = "1959-10-01", "1985-12-21"
EV_START, EV_END = "1986-01-01", "1990-12-21"
GAUGES = {17288: "Zeravshan", 16279: "Chatkal", 16290: "Pskem", 16300: "Ugam",
          16936: "Naryn", 17202: "Karatag", 17211: "Sangardak",
          16176: "Padshaata", 16202: "Chadak"}
FORCING = ["t2m", "prcp", "snow", "melt", "snowf", "pet", "swe_dif", "ddsum"]
ROLLING = ["t2m_roll3", "prcp_roll3", "melt_roll3"]
SPRING, SUMMER = (7, 15), (16, 24)
SCENARIOS = [(1, 1.0, "T+1"), (2, 1.0, "T+2"), (3, 1.0, "T+3"),
             (0, 1.1, "P+10"), (0, 0.9, "P-10"), (2, 0.9, "T+2/P-10")]
PERT_MODES = ["paper", "consistent"]
SWEEP = dict(DDF=[2.0, 4.0, 6.0], k=[0.05, 0.10, 0.20],
             aet_c=[0.20, 0.35, 0.50], T50=[0.0, 1.0, 2.0])           # 81 ta
CAL_GRID = dict(DDF=[1.5, 2.5, 3.5, 4.5, 5.5, 7.0], k=[0.03, 0.06, 0.10, 0.15, 0.22, 0.30],
                aet_c=[0.10, 0.20, 0.35, 0.50, 0.65], T50=[-1.0, 0.0, 1.0, 2.0, 2.5])  # 900 ta
SPINUP = 36

# ------------------------------------------------------------------ data
def load_basin(gid):
    d = pd.read_csv(f"{DATA_DIR}/basin_{gid}.csv")
    d.columns = [c.strip().lower() for c in d.columns]
    d["date"] = pd.to_datetime(d["date"])
    d["q"] = pd.to_numeric(d["q"], errors="coerce")
    d = d.sort_values("date").reset_index(drop=True)
    d["phase"] = d["phase"].astype(int)
    return d

def build_X(d, use_q=True):
    X = pd.DataFrame(index=d.index)
    if use_q:
        for L in range(1, N_LAGS + 1):
            X[f"q_lag{L}"] = d["q"].shift(L)
    X["sin"] = np.sin(2 * np.pi * d["phase"] / 36)
    X["cos"] = np.cos(2 * np.pi * d["phase"] / 36)
    for v in FORCING + ROLLING:
        if v in d.columns:
            X[v] = d[v]
            X[f"{v}_lag1"] = d[v].shift(1)
    return X

def new_xgb():
    return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        random_state=SEED, n_jobs=2)

# ------------------------------------------------------------------ perturbation
def perturb_paper(d, dT=0.0, fP=1.0):
    """Asl maqoladagi perturb() — aynan (taqqoslash uchun)."""
    p = d.copy()
    p["t2m"] = p["t2m"] + dT
    p["prcp"] = p["prcp"] * fP
    if "snowf" in p:
        f0 = 1 / (1 + np.exp((d["t2m"] - 1.0) / 1.5)); f1 = 1 / (1 + np.exp((p["t2m"] - 1.0) / 1.5))
        r = np.where(f0 > 1e-6, f1 / f0, 1.0)
        p["snowf"] = p["snowf"] * fP * r
        if "snow" in p: p["snow"] = p["snow"] * fP * r
    if "ddsum" in p: p["ddsum"] = np.maximum(p["t2m"], 0)
    if "pet" in p: p["pet"] = p["pet"] * (1 + 0.06 * dT)
    if "melt" in p: p["melt"] = p["melt"] * (1 + 0.12 * dT)
    if "swe_dif" in p: p["swe_dif"] = p["swe_dif"] * (1 + 0.12 * dT)
    for v in ["t2m", "prcp", "melt"]:
        c = f"{v}_roll3"
        if c in p.columns:
            p[c] = (p[v].rolling(3, min_periods=1).mean() if v == "t2m"
                    else p[v].rolling(3, min_periods=1).sum())
    return p

def snow_module(T, P, DDF=4.0, T50=1.0):
    sf = 1 / (1 + np.exp((T - T50) / 1.5)); Ps = P * sf
    swe = 0.0; melt = np.zeros(len(T)); dswe = np.zeros(len(T))
    for i in range(len(T)):
        swe += Ps[i]
        m = min(swe, max(DDF * 10 * T[i], 0.0)); swe -= m
        melt[i] = m; dswe[i] = Ps[i] - m
    return Ps, melt, dswe

def _scale(target, proxy):
    """target ≈ s·proxy (birliklarga bog'liq emas)."""
    a, b = np.nan_to_num(target.values), np.nan_to_num(proxy)
    den = (b * b).sum()
    return (a * b).sum() / den if den > 0 else 0.0

def perturb_consistent(d, dT=0.0, fP=1.0, pmm=1.0):
    """dT=0, fP=1 da bazaviy qatorni aynan qaytaradi; qor o'zgaruvchilari
       degree-day qor modulidagi farq (delta) bilan o'zgaradi — erish vaqti siljiydi."""
    p = d.copy()
    if dT == 0 and fP == 1.0:
        return p
    t0 = d["t2m"].values; t1 = t0 + dT
    P0 = d["prcp"].values * pmm; P1 = P0 * fP
    p["t2m"] = t1
    p["prcp"] = d["prcp"] * fP
    s0, m0, w0 = snow_module(t0, P0); s1, m1, w1 = snow_module(t1, P1)
    for v, a0, a1 in (("snowf", s0, s1), ("snow", s0, s1), ("melt", m0, m1), ("swe_dif", w0, w1)):
        if v in p:
            k = _scale(d[v], a0)
            new = d[v].values + k * (a1 - a0)
            p[v] = np.maximum(new, 0.0) if v != "swe_dif" else new
    if "ddsum" in p:
        pos0, pos1 = np.maximum(t0, 0), np.maximum(t1, 0)
        p["ddsum"] = d["ddsum"].values + _scale(d["ddsum"], pos0) * (pos1 - pos0)
    if "pet" in p: p["pet"] = d["pet"] * (1 + 0.06 * dT)
    for v, how in (("t2m", "mean"), ("prcp", "sum"), ("melt", "sum")):
        c = f"{v}_roll3"
        if c in p.columns and v in p.columns:
            r0 = getattr(d[v].rolling(3, min_periods=1), how)()
            r1 = getattr(p[v].rolling(3, min_periods=1), how)()
            p[c] = d[c] + (r1 - r0)
    return p

# ------------------------------------------------------------------ physical benchmark
def dd_balance(T, P, DDF=4.0, k=0.10, aet_c=0.35, T50=1.0):
    sf = 1 / (1 + np.exp((T - T50) / 1.5)); Ps, Pr = P * sf, P * (1 - sf)
    swe = S = 0.0; out = np.zeros(len(T))
    for i in range(len(T)):
        swe += Ps[i]
        melt = min(swe, max(DDF * 10 * T[i], 0.0)); swe -= melt
        pet = max(0.0, 0.20 * max(T[i], 0) * 10)
        inp = Pr[i] + melt
        S += max(0.0, inp - min(inp, aet_c * pet))
        out[i] = k * S; S -= out[i]
    return out

def nse(y, yh):
    m = np.isfinite(y) & np.isfinite(yh)
    y, yh = y[m], yh[m]
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2) if len(y) > 10 else np.nan

def calibrate_dd(d, cal, ev, pmm):
    T, P, q = d["t2m"].values, d["prcp"].values * pmm, d["q"].values
    cm = cal.copy(); cm[:SPINUP] = False; cm &= np.isfinite(q)
    best = (-np.inf, None, 1.0)
    for vals in itertools.product(*CAL_GRID.values()):
        pr = dict(zip(CAL_GRID.keys(), vals))
        s = dd_balance(T, P, **pr)
        a = (q[cm] * s[cm]).sum() / max((s[cm] ** 2).sum(), 1e-12)
        e = nse(q[cm], a * s[cm])
        if e > best[0]: best = (e, pr, a)
    e_cal, pr, a = best
    s = dd_balance(T, P, **pr)
    evm = ev & np.isfinite(q)
    return pr, e_cal, nse(q[evm], a * s[evm])

# ------------------------------------------------------------------ response metrics
def vol_resp(s, b, m):
    m = m & np.isfinite(s) & np.isfinite(b)
    return 100 * (s[m].mean() / b[m].mean() - 1)

def seas_shift(s, b, ph, m):
    m = m & np.isfinite(s) & np.isfinite(b)
    sp = m & (ph >= SPRING[0]) & (ph <= SPRING[1])
    su = m & (ph >= SUMMER[0]) & (ph <= SUMMER[1])
    rel = lambda k: 100 * (s[k].mean() / b[k].mean() - 1)
    return rel(sp) - rel(su)

# ------------------------------------------------------------------ models
def fit_models(d):
    X = build_X(d); y = d["q"].values
    cal = ((d["date"] >= CAL_START) & (d["date"] <= CAL_END)).values
    ok = cal & X.iloc[:, :N_LAGS].notna().all(1).values & np.isfinite(y)
    idx = np.where(ok)[0]; head, tail = idx[:-VAL_TAIL], idx[-VAL_TAIL:]

    def clim_fit(rows):
        c = pd.Series(y[rows]).groupby(d["phase"].values[rows]).mean()
        cm = c.reindex(np.unique(d["phase"].values)).interpolate().bfill().ffill()
        cv = cm.reindex(d["phase"].values).values
        a = y - cv
        a0, a1 = a[rows - 1], a[rows]
        g = np.isfinite(a0) & np.isfinite(a1)
        phi = (a0[g] * a1[g]).sum() / (a0[g] ** 2).sum()
        return cv, phi

    # ansambl vaznlari: faqat kalibratsiya oynasining oxirgi VAL_TAIL dekadasida
    m_h = new_xgb().fit(X.values[head], y[head])
    cv_h, phi_h = clim_fit(head)
    px = m_h.predict(X.values[tail])
    pc = cv_h[tail] + phi_h * (y[tail - 1] - cv_h[tail - 1])
    ix = 1 / np.sqrt(np.mean((y[tail] - px) ** 2)); ic = 1 / np.sqrt(np.mean((y[tail] - pc) ** 2))
    w = ix / (ix + ic)

    m_x = new_xgb().fit(X.values[idx], y[idx])
    cv, phi = clim_fit(idx)
    Xn = build_X(d, use_q=False)
    okn = cal & np.isfinite(y)
    m_n = new_xgb().fit(Xn.values[okn], y[okn])
    return dict(xgb=m_x, nq=m_n, cv=cv, phi=phi, w=w)

def one_step(M, p, qobs):
    X = build_X(p.assign(q=qobs)).values
    px = M["xgb"].predict(X)
    pc = M["cv"] + M["phi"] * (np.r_[np.nan, qobs[:-1]] - np.r_[np.nan, M["cv"][:-1]])
    return dict(xgb=px, clim=pc, ens=M["w"] * px + (1 - M["w"]) * pc)

def rollout(M, p, qobs, ev_idx, member, annual):
    X = build_X(p.assign(q=qobs)).values.astype(float)
    bst = M["xgb"].get_booster()
    cv, phi, w, ph = M["cv"], M["phi"], M["w"], p["phase"].values
    qfill = np.where(np.isfinite(qobs), qobs, cv)          # bo'shliqlar → klimatologiya
    qs = qfill.astype(float).copy(); out = np.full(len(qobs), np.nan)
    p0 = ph.min()                                          # yil boshidagi faza (0 yoki 1)
    for t in ev_idx:
        if annual and ph[t] == p0 and t != ev_idx[0]:
            qs[t - N_LAGS:t] = qfill[t - N_LAGS:t]
        pc = cv[t] + phi * (qs[t - 1] - cv[t - 1])
        if member == "clim":
            v = pc
        else:
            X[t, :N_LAGS] = qs[t - N_LAGS:t][::-1]
            px = float(bst.inplace_predict(X[t:t + 1])[0])
            v = px if member == "xgb" else w * px + (1 - w) * pc
        v = max(v, 0.0); qs[t] = v; out[t] = v
    return out

def simulate_all(M, p, qobs, ev_idx):
    r = {f"OS_{k}": v for k, v in one_step(M, p, qobs).items()}
    for mem in ("xgb", "clim", "ens"):
        r[f"AR_{mem}"] = rollout(M, p, qobs, ev_idx, mem, annual=False)
        r[f"ARy_{mem}"] = rollout(M, p, qobs, ev_idx, mem, annual=True)
    r["NQ_xgb"] = M["nq"].predict(build_X(p, use_q=False).values)
    return r

# ------------------------------------------------------------------ main loop
resp_rows, skill_rows = [], []
for gid, name in GAUGES.items():
    tb = time.time()
    fpath = f"{DATA_DIR}/basin_{gid}.csv"
    if not os.path.exists(fpath):
        print(f"!! {fpath} topilmadi — o'tkazib yuborildi"); continue
    d = load_basin(gid)
    qobs = d["q"].values.astype(float); ph = d["phase"].values
    cal = ((d["date"] >= CAL_START) & (d["date"] <= CAL_END)).values
    ev = ((d["date"] >= EV_START) & (d["date"] <= EV_END)).values
    ev_idx = np.where(ev)[0]
    evm = ev & np.isfinite(qobs)
    ann_p = d.groupby(d["date"].dt.year)["prcp"].sum().median()
    pmm = 1000.0 if ann_p < 20 else 1.0          # ERA5 metrda bo'lsa → mm

    M = fit_models(d)

    # fizik benchmarklar (inputlar ikkala rejimda bir xil — t2m, prcp)
    T, P = d["t2m"].values, d["prcp"].values * pmm
    pr_cal, nse_dd_cal, nse_dd_val = calibrate_dd(d, cal, ev, pmm)
    sweep = [dict(zip(SWEEP.keys(), v)) for v in itertools.product(*SWEEP.values())]
    phys = {}
    for dT, fP, lab in SCENARIOS:
        vu, su = [], []
        for pr in sweep:
            b = dd_balance(T, P, **pr); s = dd_balance(T + dT, P * fP, **pr)
            vu.append(vol_resp(s, b, evm)); su.append(seas_shift(s, b, ph, evm))
        b = dd_balance(T, P, **pr_cal); s = dd_balance(T + dT, P * fP, **pr_cal)
        phys[lab] = dict(DD_uncal=(np.median(vu), np.median(su)),
                         DD_cal=(vol_resp(s, b, evm), seas_shift(s, b, ph, evm)))

    for pmode in PERT_MODES:
        pert = (perturb_paper if pmode == "paper"
                else lambda dd, a, b: perturb_consistent(dd, a, b, pmm))
        base = simulate_all(M, d, qobs, ev_idx)
        if pmode == "paper":
            for src, s in base.items():
                skill_rows.append(dict(gauge=gid, basin=name, source=src, NSE_eval=nse(qobs[evm], s[evm])))
            skill_rows.append(dict(gauge=gid, basin=name, source="DD_cal", NSE_eval=nse_dd_val,
                                   NSE_cal=nse_dd_cal, w_xgb=M["w"], phi=M["phi"], **pr_cal))
        for dT, fP, lab in SCENARIOS:
            sim = simulate_all(M, pert(d, dT, fP), qobs, ev_idx)
            for src in sim:
                resp_rows.append(dict(gauge=gid, basin=name, pert_mode=pmode, scenario=lab,
                                      source=src, vol=vol_resp(sim[src], base[src], evm),
                                      seas=seas_shift(sim[src], base[src], ph, evm)))
            for src, (v, s_) in phys[lab].items():
                resp_rows.append(dict(gauge=gid, basin=name, pert_mode=pmode, scenario=lab,
                                      source=src, vol=v, seas=s_))
    print(f"{name:10s} tayyor  w_xgb={M['w']:.2f}  DD_cal NSE cal/val={nse_dd_cal:.2f}/{nse_dd_val:.2f}"
          f"  ({time.time() - tb:.0f} s)")

R = pd.DataFrame(resp_rows); S = pd.DataFrame(skill_rows)
R.to_csv(f"{OUT}/A1A2_responses.csv", index=False)
S.to_csv(f"{OUT}/A1A2_skill.csv", index=False)

# ------------------------------------------------------------------ summary
def summarise(R):
    rows = []
    for (pm, sc), g in R.groupby(["pert_mode", "scenario"]):
        ref = {k: g[g.source == k].set_index("gauge") for k in ("DD_uncal", "DD_cal")}
        for src, gs in g.groupby("source"):
            gs = gs.set_index("gauge"); row = dict(pert_mode=pm, scenario=sc, source=src,
                                                   n=len(gs), vol_med=gs.vol.median(), seas_med=gs.seas.median())
            for rk, rf in ref.items():
                j = gs.join(rf[["vol", "seas"]], rsuffix="_ref")
                key = "seas" if sc.startswith("T") else "vol"
                ratio = j[key] / j[f"{key}_ref"]
                row[f"ratio_{rk}"] = ratio.median()
                row[f"sign_{rk}"] = int((np.sign(j[key]) == np.sign(j[f"{key}_ref"])).sum())
            rows.append(row)
    return pd.DataFrame(rows)

SUM = summarise(R)
SUM.to_csv(f"{OUT}/A1A2_summary.csv", index=False)

# bevosita va tarqalgan ulush (ansambl, mavsumiy siljish)
dec = []
for (pm, sc), g in R[R.scenario.str.startswith("T")].groupby(["pert_mode", "scenario"]):
    p_ = g.pivot(index="gauge", columns="source", values="seas")
    dec.append(dict(pert_mode=pm, scenario=sc, direct_OS=p_["OS_ens"].median(),
                    total_AR=p_["AR_ens"].median(),
                    propagated=(p_["AR_ens"] - p_["OS_ens"]).median(),
                    total_ARy=p_["ARy_ens"].median(), NQ=p_["NQ_xgb"].median(),
                    DD_cal=p_["DD_cal"].median(), DD_uncal=p_["DD_uncal"].median()))
DEC = pd.DataFrame(dec); DEC.to_csv(f"{OUT}/A1_decomposition.csv", index=False)

# ------------------------------------------------------------------ figures
ORDER = ["OS_clim", "OS_xgb", "OS_ens", "AR_clim", "AR_xgb", "AR_ens",
         "ARy_xgb", "ARy_ens", "NQ_xgb", "DD_cal", "DD_uncal"]
def strip(ax, sub, col, title):
    for i, s in enumerate(ORDER):
        v = sub[sub.source == s][col].values
        c = "tab:red" if s.startswith("DD") else "tab:blue"
        ax.scatter(np.full(len(v), i) + np.random.default_rng(SEED).uniform(-.12, .12, len(v)),
                   v, s=18, color=c, alpha=.8)
        if len(v): ax.hlines(np.median(v), i - .3, i + .3, color="k", lw=2)
    ax.axhline(0, color="grey", lw=.8, ls="--")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=60, ha="right")
    ax.set_title(title, fontsize=10)

for pm in PERT_MODES:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    strip(ax[0], R[(R.pert_mode == pm) & (R.scenario == "T+2")], "seas",
          f"T+2 K: mavsumiy siljish, % (bahor − yoz) — {pm}")
    strip(ax[1], R[(R.pert_mode == pm) & (R.scenario == "P+10")], "vol",
          f"P+10%: yillik hajm javobi, % — {pm}")
    fig.tight_layout(); fig.savefig(f"{OUT}/figA1_{pm}.png", dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
sk = S[S.source.isin(["OS_ens", "OS_xgb", "AR_ens", "AR_xgb", "ARy_ens", "NQ_xgb", "DD_cal"])]
sk.pivot(index="basin", columns="source", values="NSE_eval").plot.bar(ax=ax, width=.85)
ax.set_ylabel("NSE (1986–1990)"); ax.set_ylim(-1, 1); ax.axhline(0, color="k", lw=.8)
ax.legend(fontsize=7, ncol=4); fig.tight_layout()
fig.savefig(f"{OUT}/figA1_skill.png", dpi=300); plt.close(fig)

# ------------------------------------------------------------------ report
pd.set_option("display.width", 200, "display.max_columns", 20)
print("\n=== Asl maqola bilan moslik tekshiruvi (paper rejimi, T+2) ===")
chk = SUM[(SUM.pert_mode == "paper") & (SUM.scenario == "T+2")].set_index("source")
if "OS_ens" in chk.index:
    print(f"OS_ens mavsumiy siljish median: {chk.loc['OS_ens','seas_med']:.2f}%  (maqolada −1.35%)")
    print(f"DD_uncal median:                {chk.loc['DD_uncal','seas_med']:.2f}%  (maqolada +26.37%)")
    print(f"OS_ens / DD_uncal nisbati:      {chk.loc['OS_ens','ratio_DD_uncal']:.3f}  (maqolada −0.080)")
print("\n=== A1: bevosita va tarqalgan javob (mavsumiy siljish, median %) ===")
print(DEC.round(2).to_string(index=False))
print("\n=== T+2 xulosa ===")
print(SUM[SUM.scenario == "T+2"][["pert_mode", "source", "seas_med", "ratio_DD_cal",
                                    "sign_DD_cal", "ratio_DD_uncal", "sign_DD_uncal"]]
      .round(3).to_string(index=False))
print("\n=== Baseline skill (NSE, median) ===")
print(S.groupby("source")["NSE_eval"].median().round(3).to_string())
print(f"\nNatijalar: {OUT}   |   Umumiy vaqt: {(time.time() - T0) / 60:.1f} daqiqa")
