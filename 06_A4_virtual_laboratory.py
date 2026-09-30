# =====================================================================
# JWCC v2 — A4: virtual laboratoriya
# Asosiy katak va A3 katagidan KEYIN, xuddi shu sessiyada ishga tushiring
# (A3 dagi Tab, Seq, rollout_gen kerak). B2 natijasi Drive'dan o'qiladi.
#
# "Haqiqat": har bir basseyn uchun eng yaxshi kalibrlangan HBV_glac, ERA5 T va P bilan.
#   Virtual kuzatuv = haqiqat × (1 + 10% shovqin). Iqlim javobi aniq ma'lum.
# ML inputlari faqat T va P dan olinadi → perturbatsiya haqiqat bilan to'liq izchil.
# Regressiya suyulishi testi: ML toza P bilan (sigma=0) va shovqinli P bilan
#   (P·exp(sigma·z − sigma²/2), sigma=0.5, o'rtachani saqlaydi) o'qitiladi;
#   haqiqat har doim toza P dan hisoblanadi.
# =====================================================================
assert "load_basin" in globals(), "Avval asosiy katakni (jwcc_v2_A1_A2.py) ishga tushiring"
assert "Seq" in globals() and "rollout_gen" in globals(), "Avval A3 katagini shu sessiyada ishga tushiring"
assert os.path.exists(f"{OUT}/B2_hbv_skill.csv"), "B2_hbv_skill.csv topilmadi — B2 katagini ishga tushiring"
import time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
try:
    import numba as nb
except ImportError:
    os.system("pip -q install numba"); import numba as nb
T4 = time.time()

SIGMAS = [0.0, 0.5]
SCEN_V = [(2, 1.0, "T+2"), (3, 1.0, "T+3"), (0, 1.1, "P+10"), (0, 0.9, "P-10")]
Q_NOISE = 0.10
LSTM_SEEDS = 2                      # A3 da 3 edi; vaqtni tejash uchun
N_BANDS, LAPSE = 6, 0.0065
ATTR = {16279: (5669, 2693, 945, 4441, 0.50), 16290: (2518, 2795, 896, 4372, 3.21),
        16300: (863, 2046, 755, 3600, 0.00), 16936: (52030, 2849, 861, 5119, 1.96),
        17202: (683, 2666, 915, 4744, 2.80), 17211: (906, 2333, 746, 4115, 0.04),
        17288: (10297, 3110, 1048, 5455, 5.33), 16176: (368, 2797, 1524, 4314, 0.69),
        16202: (351, 2373, 1068, 3424, 0.00)}
PNAMES = ["TT", "CFMAX", "SFCF", "CFGL", "FC", "LP", "BETA", "PERC", "UZL",
          "K0", "K1", "K2", "PCORR", "ETF"]

# ------------------------------------------------------------------ HBV (B2 bilan aynan bir xil)
@nb.njit(cache=True)
def hbv(T, P, nd, rs, bz, bw, bg, par, lapse):
    TT, CFMAX, SFCF, CFGL, FC, LP, BETA, PERC, UZL, K0, K1, K2, PCORR, ETF = par
    CWH, TTI = 0.1, 2.0
    nbd = bz.shape[0]; n = T.shape[0]
    sp = np.zeros((nbd, 2)); wc = np.zeros((nbd, 2)); sm = np.full(nbd, 0.5 * FC)
    suz = 0.0; slz = 20.0; Q = np.zeros(n); ICE = np.zeros(n)
    for t in range(n):
        d = nd[t]; rech = 0.0; gq = 0.0; icet = 0.0
        if rs[t]:
            for i in range(nbd):
                sp[i, 1] = 0.0; wc[i, 1] = 0.0; sp[i, 0] = min(sp[i, 0], 500.0)
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
                    rr = out * min(sm[i] / FC, 1.0) ** BETA
                    sm[i] += out - rr
                    ea = min(ETF * d * max(Ti, 0.0) * min(sm[i] / (FC * LP), 1.0), sm[i]); sm[i] -= ea
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
    up = z >= zmean
    g = np.where(up, min(glac_pct / 100 / w[up].sum(), 1.0), 0.0)
    return z - (w * z).sum(), w, g

# ------------------------------------------------------------------ virtual xususiyatlar
def _sh(a, L): return np.r_[np.full(L, np.nan), a[:-L]]
def _roll(a, how):
    s = pd.Series(a).rolling(3, min_periods=1); return (s.mean() if how == "mean" else s.sum()).values

def build_Xv(T, P, q, ph, use_q=True):
    cols = [_sh(q, L) for L in range(1, N_LAGS + 1)] if use_q else []
    cols += [np.sin(2 * np.pi * ph / 36), np.cos(2 * np.pi * ph / 36)]
    for v in (T, P, _roll(T, "mean"), _roll(P, "sum"), np.maximum(T, 0)):
        cols += [v, _sh(v, 1)]
    return np.column_stack(cols).astype(float)

HP = pd.read_csv(f"{OUT}/B2_hbv_skill.csv")
HP = HP[HP.variant == "HBV_glac"].sort_values("NSE_cal", ascending=False).drop_duplicates("gauge").set_index("gauge")

# ------------------------------------------------------------------ asosiy sikl
rows, skill = [], []
for gid, name in GAUGES.items():
    tb = time.time()
    if not os.path.exists(f"{DATA_DIR}/basin_{gid}.csv") or gid not in HP.index: continue
    d = load_basin(gid); n = len(d); ph = d.phase.values
    area, zm, zmin, zmax, gl = ATTR[gid]
    nd = np.where(d.date.dt.day < 21, 10, d.date.dt.days_in_month - 20).astype(float)
    rs = ((d.date.dt.month == 10) & (d.date.dt.day == 1)).values
    pmm = 1000.0 if d.groupby(d.date.dt.year).prcp.sum().median() < 20 else 1.0
    T = d.t2m.values.astype(float); P = d.prcp.values.astype(float) * pmm
    bz, bw, bg = make_bands(zm, zmin, zmax, gl)
    x = HP.loc[gid, PNAMES].values.astype(float)
    truth = lambda dT, fP: hbv(T + dT, P * fP, nd, rs, bz, bw, bg, x, LAPSE)[0]
    qt = truth(0, 1.0)
    rng = np.random.default_rng(SEED + gid)
    qv = np.maximum(qt * (1 + Q_NOISE * rng.standard_normal(n)), 0.0)       # virtual kuzatuv
    zP = rng.standard_normal(n)
    cal = ((d.date >= CAL_START) & (d.date <= CAL_END)).values.copy(); cal[:SPINUP] = False
    ev = ((d.date >= EV_START) & (d.date <= EV_END)).values
    ev_idx = np.where(ev)[0]; evm = ev.copy()
    rq = np.where(cal)[0][N_LAGS:]; rn = np.where(cal)[0]
    tr = {lab: (vol_resp(truth(dT, fP), qt, evm), seas_shift(truth(dT, fP), qt, ph, evm))
          for dT, fP, lab in SCEN_V}

    for sg in SIGMAS:
        Pin = P * np.exp(sg * zP - sg ** 2 / 2)
        Xq = build_Xv(T, Pin, qv, ph); Xn = build_Xv(T, Pin, qv, ph, use_q=False)
        M = {"XGB": (Tab(new_xgb(), Xq, qv, rq, impute=False), Tab(new_xgb(), Xn, qv, rn, impute=False)),
             "RF": (Tab(RandomForestRegressor(200, min_samples_leaf=3, n_jobs=-1, random_state=SEED), Xq, qv, rq),
                    Tab(RandomForestRegressor(200, min_samples_leaf=3, n_jobs=-1, random_state=SEED), Xn, qv, rn)),
             "Ridge": (Tab(RidgeCV(alphas=np.logspace(-2, 3, 12)), Xq, qv, rq),
                       Tab(RidgeCV(alphas=np.logspace(-2, 3, 12)), Xn, qv, rn)),
             "LSTM": (Seq(Xq, qv, rq, W_AR), Seq(Xn, qv, rn, W_NQ))}

        def run(dT, fP):
            Tp, Pp = T + dT, Pin * fP
            Xqp = build_Xv(Tp, Pp, qv, ph); Xnp = build_Xv(Tp, Pp, qv, ph, use_q=False)
            r = {}
            for a, (mq, mn) in M.items():
                r[f"{a}_OS"] = full(mq.batch(Xqp, ev_idx), ev_idx, n)
                r[f"{a}_AR"] = rollout_gen(mq, Xqp, qv, ev_idx)
                r[f"{a}_NQ"] = full(mn.batch(Xnp, ev_idx), ev_idx, n)
            return r

        base = run(0, 1.0)
        for k, s in base.items():
            skill.append(dict(gauge=gid, basin=name, sigma=sg, source=k, NSE_eval=nse(qv[evm], s[evm])))
        for dT, fP, lab in SCEN_V:
            sim = run(dT, fP)
            for k in sim:
                rows.append(dict(gauge=gid, basin=name, sigma=sg, scenario=lab, source=k,
                                 vol=vol_resp(sim[k], base[k], evm), seas=seas_shift(sim[k], base[k], ph, evm),
                                 vol_true=tr[lab][0], seas_true=tr[lab][1]))
    print(f"{name:10s} tayyor  haqiqat T+2 siljish={tr['T+2'][1]:5.1f}%  P+10 hajm={tr['P+10'][0]:5.1f}%"
          f"  ({time.time() - tb:.0f} s)")

V = pd.DataFrame(rows); SV = pd.DataFrame(skill)
V.to_csv(f"{OUT}/A4_virtual_responses.csv", index=False); SV.to_csv(f"{OUT}/A4_virtual_skill.csv", index=False)

# ------------------------------------------------------------------ xulosa
ORD = [f"{a}_{m}" for a in ("XGB", "RF", "Ridge", "LSTM") for m in ("OS", "AR", "NQ")]
def rt(scn, key):
    g = V[V.scenario == scn].copy(); g["r"] = g[key] / g[f"{key}_true"]
    return g.pivot_table(index="source", columns="sigma", values="r", aggfunc="median").reindex(ORD)
RT2, RT3, RP = rt("T+2", "seas"), rt("T+3", "seas"), rt("P+10", "vol")
SUMV = pd.concat({"T+2 siljish": RT2, "T+3 siljish": RT3, "P+10 hajm": RP}, axis=1)
SUMV.columns = [f"{a} | σ={b}" for a, b in SUMV.columns]
if os.path.exists(f"{OUT}/A3_summary.csv"):
    A3s = pd.read_csv(f"{OUT}/A3_summary.csv", index_col=0)
    SUMV["real T+2 (A3)"] = A3s.T2_seas_ratio; SUMV["real P+10 (A3)"] = A3s.P10_vol_ratio
SUMV.to_csv(f"{OUT}/A4_summary.csv")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.3)); xx = np.arange(len(ORD)); w = 0.27
for i, (tbl, lab) in enumerate(((RT2, "T+2: mavsumiy siljish / haqiqat"), (RP, "P+10: hajm / haqiqat"))):
    ax[i].bar(xx - w, tbl[0.0], w, label="virtual, toza P")
    ax[i].bar(xx, tbl[0.5], w, label="virtual, shovqinli P (σ=0.5)")
    real = SUMV.get("real T+2 (A3)" if i == 0 else "real P+10 (A3)")
    if real is not None: ax[i].bar(xx + w, real.reindex(ORD), w, label="real ma'lumot (HBV ga nisbat)")
    ax[i].axhline(1, color="k", lw=1); ax[i].set_xticks(xx)
    ax[i].set_xticklabels(ORD, rotation=60, ha="right", fontsize=8); ax[i].set_title(lab, fontsize=10)
    ax[i].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/figA4_virtual.png", dpi=300); plt.close(fig)

pd.set_option("display.width", 230, "display.max_columns", 30)
print("\n=== Virtual skill, NSE (median) ===")
print(SV.pivot_table(index="source", columns="sigma", values="NSE_eval", aggfunc="median").reindex(ORD).round(3).to_string())
print("\n=== Model javobi / haqiqiy javob (basseynlar mediani); 1.0 = to'liq tiklash ===")
print(SUMV.round(2).to_string())
print("\n=== Regressiya suyulishi: P+10 nisbati, shovqinli / toza ===")
print((RP[0.5] / RP[0.0]).round(2).to_string())
print(f"\nNatijalar: {OUT}/A4_*.csv, figA4_virtual.png   |   Vaqt: {(time.time() - T4) / 60:.1f} daqiqa")
