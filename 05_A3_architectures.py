# =====================================================================
# JWCC v2 — A3: arxitekturalar taqqoslovi
# Asosiy katak va B2 (HBV) katagidan KEYIN, xuddi shu sessiyada ishga tushiring.
# GPU tavsiya etiladi (Runtime → Change runtime type → T4), lekin CPU da ham ishlaydi.
#
# Arxitekturalar: XGB, RF, Ridge (daraxtli / ansambl / chiziqli), LSTM_AR va LSTM_NQ.
# Rejimlar: OS (bir qadam, kuzatilgan lagged Q), AR (rollout), NQ (lagged Q siz).
# Perturbatsiya: consistent (standart qor moduli), 6 ssenariy.
# Benchmark: HBV_glac (B2_comparison.csv dan).
# Savol: yog'inga past sezgirlik va katta isishda to'yinish daraxtlarga xosmi?
# =====================================================================
assert "perturb_consistent" in globals(), "Avval asosiy katakni (jwcc_v2_A1_A2.py) ishga tushiring"
assert os.path.exists(f"{OUT}/B2_comparison.csv"), "Avval B2 (HBV) katagini ishga tushiring"
import time, copy
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
try:
    import torch, torch.nn as nn
except ImportError:
    os.system("pip -q install torch"); import torch, torch.nn as nn
T3 = time.time()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("Qurilma:", DEV)

LSTM_SEEDS, W_AR, W_NQ, HID = 3, 12, 36, 32
EXTRAP_VARS = ["t2m", "ddsum", "t2m_roll3", "prcp", "melt", "snowf"]

# ------------------------------------------------------------------ model o'ramlari
class Tab:
    """Jadvalli model: NaN → kalibratsiya o'rtachasi (XGB uchun kerak emas, lekin zarar qilmaydi)."""
    def __init__(s, est, X, y, rows, impute=True):
        s.mu = np.nanmean(X[rows], 0) if impute else None
        s.est = est.fit(s._f(X[rows]), y[rows])
        if hasattr(s.est, "n_jobs"): s.est.n_jobs = 1
    def _f(s, A):
        if s.mu is None: return A
        A = A.copy(); m = ~np.isfinite(A); A[m] = np.take(s.mu, np.where(m)[1]); return A
    def batch(s, X, idx): return s.est.predict(s._f(X[idx]))
    def at(s, X, t): return float(s.est.predict(s._f(X[t:t + 1]))[0])

class Net(nn.Module):
    def __init__(s, nf):
        super().__init__(); s.l = nn.LSTM(nf, HID, batch_first=True); s.d = nn.Dropout(0.1); s.o = nn.Linear(HID, 1)
    def forward(s, x):
        h, _ = s.l(x); return s.o(s.d(h[:, -1])).squeeze(-1)

class Seq:
    def __init__(s, X, y, rows, W):
        s.W = W; s.mu = np.nanmean(X[rows], 0); s.sd = np.nanstd(X[rows], 0) + 1e-6
        s.ym, s.ys = y[rows].mean(), y[rows].std()
        rows = rows[rows >= W - 1]; head, tail = rows[:-VAL_TAIL], rows[-VAL_TAIL:]
        Xh, yh = s._win(X, head), torch.tensor((y[head] - s.ym) / s.ys, dtype=torch.float32, device=DEV)
        Xt, yt = s._win(X, tail), torch.tensor((y[tail] - s.ym) / s.ys, dtype=torch.float32, device=DEV)
        s.nets = []
        for k in range(LSTM_SEEDS):
            torch.manual_seed(SEED + k); np.random.seed(SEED + k)
            net = Net(X.shape[1]).to(DEV); opt = torch.optim.Adam(net.parameters(), 1e-3)
            best, bst, wait = np.inf, None, 0
            for ep in range(300):
                net.train(); perm = torch.randperm(len(yh), device=DEV)
                for b in range(0, len(yh), 64):
                    i = perm[b:b + 64]; opt.zero_grad()
                    loss = ((net(Xh[i]) - yh[i]) ** 2).mean(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad(): v = ((net(Xt) - yt) ** 2).mean().item()
                if v < best - 1e-5: best, bst, wait = v, copy.deepcopy(net.state_dict()), 0
                else:
                    wait += 1
                    if wait >= 25: break
            net.load_state_dict(bst); net.eval(); s.nets.append(net)
    def _win(s, X, idx):
        Z = np.nan_to_num((X - s.mu) / s.sd)
        A = np.stack([Z[i - s.W + 1:i + 1] for i in idx])
        return torch.tensor(A, dtype=torch.float32, device=DEV)
    def batch(s, X, idx):
        with torch.no_grad():
            xw = s._win(X, idx)
            p = torch.stack([n(xw) for n in s.nets]).mean(0).cpu().numpy()
        return p * s.ys + s.ym
    def at(s, X, t): return float(s.batch(X, [t])[0])

def rollout_gen(m, X0, qfill, ev_idx):
    X = X0.copy(); qs = qfill.copy(); out = np.full(len(qfill), np.nan)
    for t in ev_idx:
        X[t, :N_LAGS] = qs[t - N_LAGS:t][::-1]
        v = max(m.at(X, t), 0.0); qs[t] = v; out[t] = v
    return out

def full(pred, idx, n):
    o = np.full(n, np.nan); o[idx] = pred; return o

# ------------------------------------------------------------------ asosiy sikl
rows, skill, extra = [], [], []
for gid, name in GAUGES.items():
    tb = time.time()
    if not os.path.exists(f"{DATA_DIR}/basin_{gid}.csv"): continue
    d = load_basin(gid); n = len(d)
    y = d.q.values.astype(float); ph = d.phase.values
    qfill = pd.Series(y).interpolate().bfill().ffill().values
    cal = ((d.date >= CAL_START) & (d.date <= CAL_END)).values
    ev = ((d.date >= EV_START) & (d.date <= EV_END)).values
    ev_idx = np.where(ev)[0]; evm = ev & np.isfinite(y)
    pmm = 1000.0 if d.groupby(d.date.dt.year).prcp.sum().median() < 20 else 1.0
    Xq = build_X(d).values.astype(float); Xn = build_X(d, use_q=False).values.astype(float)
    rq = np.where(cal & np.isfinite(y) & np.isfinite(Xq[:, :N_LAGS]).all(1))[0]
    rn = np.where(cal & np.isfinite(y))[0]

    M = {"XGB": (Tab(new_xgb(), Xq, y, rq, impute=False), Tab(new_xgb(), Xn, y, rn, impute=False)),
         "RF": (Tab(RandomForestRegressor(200, min_samples_leaf=3, n_jobs=-1, random_state=SEED), Xq, y, rq),
                Tab(RandomForestRegressor(200, min_samples_leaf=3, n_jobs=-1, random_state=SEED), Xn, y, rn)),
         "Ridge": (Tab(RidgeCV(alphas=np.logspace(-2, 3, 12)), Xq, y, rq),
                   Tab(RidgeCV(alphas=np.logspace(-2, 3, 12)), Xn, y, rn)),
         "LSTM": (Seq(Xq, y, rq, W_AR), Seq(Xn, y, rn, W_NQ))}

    def run(p):
        Xqp = build_X(p.assign(q=qfill)).values.astype(float)
        Xnp = build_X(p, use_q=False).values.astype(float)
        r = {}
        for a, (mq, mn) in M.items():
            r[f"{a}_OS"] = full(mq.batch(Xqp, ev_idx), ev_idx, n)
            r[f"{a}_AR"] = rollout_gen(mq, Xqp, qfill, ev_idx)
            r[f"{a}_NQ"] = full(mn.batch(Xnp, ev_idx), ev_idx, n)
        return r

    base = run(d)
    for k, s in base.items():
        skill.append(dict(gauge=gid, basin=name, source=k, NSE_eval=nse(y[evm], s[evm])))
    calmax = {v: d.loc[cal, v].max() for v in EXTRAP_VARS if v in d.columns}
    for dT, fP, lab in SCENARIOS:
        p = perturb_consistent(d, dT, fP, pmm)
        sim = run(p)
        for k in sim:
            rows.append(dict(gauge=gid, basin=name, scenario=lab, source=k,
                             vol=vol_resp(sim[k], base[k], evm), seas=seas_shift(sim[k], base[k], ph, evm)))
        oor = np.zeros(ev.sum(), bool)
        for v, mx in calmax.items(): oor |= p.loc[ev, v].values > mx
        extra.append(dict(gauge=gid, basin=name, scenario=lab, out_of_range_pct=100 * oor.mean()))
    print(f"{name:10s} tayyor  ({time.time() - tb:.0f} s)")

A = pd.DataFrame(rows); S = pd.DataFrame(skill); E = pd.DataFrame(extra)
A.to_csv(f"{OUT}/A3_responses.csv", index=False); S.to_csv(f"{OUT}/A3_skill.csv", index=False)
E.to_csv(f"{OUT}/A3_out_of_range.csv", index=False)

# ------------------------------------------------------------------ HBV bilan taqqoslash
B = pd.read_csv(f"{OUT}/B2_comparison.csv").set_index(["gauge", "scenario"])[["seas_HBV_glac", "vol_HBV_glac"]]
A = A.join(B, on=["gauge", "scenario"])
A["arch"] = A.source.str.rsplit("_", n=1).str[0]; A["mode"] = A.source.str.rsplit("_", n=1).str[1]
ORD = [f"{a}_{m}" for a in ("XGB", "RF", "Ridge", "LSTM") for m in ("OS", "AR", "NQ")]

def tab(scn, key, ref):
    g = A[A.scenario == scn]
    out = g.groupby("source").agg(med=(key, "median"),
                                  ratio=(key, lambda s: (s / g.loc[s.index, ref]).median()),
                                  sign=(key, lambda s: int((np.sign(s) == np.sign(g.loc[s.index, ref])).sum())))
    return out.reindex(ORD)

T2t, P10 = tab("T+2", "seas", "seas_HBV_glac"), tab("P+10", "vol", "vol_HBV_glac")
sat = A[A.scenario.isin(["T+1", "T+2", "T+3"])].pivot_table(index="source", columns="scenario",
                                                             values="seas", aggfunc="median").reindex(ORD)
hb = B.reset_index().groupby("scenario").seas_HBV_glac.median()
sat.loc["HBV_glac"] = [hb["T+1"], hb["T+2"], hb["T+3"]]
sat["T3/T2"] = sat["T+3"] / sat["T+2"]
SUMM = T2t.add_prefix("T2_seas_").join(P10.add_prefix("P10_vol_")).join(sat[["T+1", "T+2", "T+3", "T3/T2"]])
SUMM.to_csv(f"{OUT}/A3_summary.csv")

# ------------------------------------------------------------------ rasm
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
cols = {"XGB": "tab:blue", "RF": "tab:green", "Ridge": "tab:orange", "LSTM": "tab:purple"}
for a, c in cols.items():
    for m, ls in (("AR", "-"), ("NQ", "--")):
        k = f"{a}_{m}"
        ax[0].plot([1, 2, 3], sat.loc[k, ["T+1", "T+2", "T+3"]], ls, color=c, marker="o", label=k)
ax[0].plot([1, 2, 3], sat.loc["HBV_glac", ["T+1", "T+2", "T+3"]], "k-", lw=2.5, marker="s", label="HBV_glac")
ax[0].set_xlabel("ΔT, K"); ax[0].set_ylabel("Mavsumiy siljish, median %"); ax[0].legend(fontsize=7, ncol=2)
ax[0].set_title("Isishga javob va to'yinish", fontsize=10)
for i, (tbl, lab) in enumerate(((T2t, "T+2: mavsumiy siljish / HBV"), (P10, "P+10: hajm / HBV"))):
    ax[i + 1].bar(range(len(ORD)), tbl.ratio, color=[cols[k.split("_")[0]] for k in ORD])
    ax[i + 1].axhline(1, color="k", lw=1); ax[i + 1].axhline(0, color="grey", lw=.6)
    ax[i + 1].set_xticks(range(len(ORD))); ax[i + 1].set_xticklabels(ORD, rotation=60, ha="right", fontsize=8)
    ax[i + 1].set_title(lab + " (basseynlar mediani)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/figA3_architectures.png", dpi=300); plt.close(fig)

# ------------------------------------------------------------------ hisobot
pd.set_option("display.width", 220, "display.max_columns", 30)
print("\n=== Bazaviy skill, NSE 1986–1990 (median) ===")
print(S.groupby("source").NSE_eval.median().reindex(ORD).round(3).to_string())
print("\n=== T+2 mavsumiy siljish va P+10 hajm: median %, HBV_glac ga nisbat, ishora (9 dan) ===")
print(SUMM[[c for c in SUMM.columns if c.startswith(("T2_", "P10_"))]].round(2).to_string())
print("\n=== To'yinish: T+1/T+2/T+3 median mavsumiy siljish va T+3/T+2 (HBV ~1.5) ===")
print(sat.round(2).to_string())
print("\n=== Trening diapazonidan tashqaridagi dekadalar ulushi, % (median) ===")
print(E.groupby("scenario").out_of_range_pct.median().round(1).to_string())
print("\n=== Karatag, T+2 mavsumiy siljish (HBV_glac bilan) ===")
k = A[(A.basin == "Karatag") & (A.scenario == "T+2")].set_index("source").reindex(ORD)
print(k[["seas", "seas_HBV_glac"]].round(2).to_string())
print(f"\nNatijalar: {OUT}/A3_*.csv, figA3_architectures.png   |   Vaqt: {(time.time() - T3) / 60:.1f} daqiqa")
