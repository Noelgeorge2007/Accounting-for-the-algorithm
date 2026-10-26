import json, time, resource, warnings
import numpy as np
from scipy.stats import qmc, spearmanr
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from model import evaluate, BASE, LB, UB, VAR_NAMES

warnings.filterwarnings("ignore")
RES = {}
STAGES = {}


class Stage:
    def __init__(self, name): self.name = name
    def __enter__(self):
        self.t0, self.c0 = time.perf_counter(), time.process_time(); return self
    def __exit__(self, *a):
        STAGES[self.name] = dict(wall_s=time.perf_counter() - self.t0, cpu_s=time.process_time() - self.c0,
                                 peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        print(f"[{self.name}] {STAGES[self.name]}", flush=True)


class SingleObj(Problem):
    def __init__(self, P): super().__init__(n_var=6, n_obj=1, xl=LB, xu=UB); self.P = P
    def _evaluate(self, X, out, *a, **k): out["F"] = -evaluate(X, self.P)["AITP"]


class TriObj(Problem):
    def __init__(self, P): super().__init__(n_var=6, n_obj=3, xl=LB, xu=UB); self.P = P
    def _evaluate(self, X, out, *a, **k):
        r = evaluate(X, self.P); out["F"] = np.column_stack([-r["AITP"], r["CF"], r["WF"]])


def ga(P, seed, gens=150):
    res = minimize(SingleObj(P), GA(pop_size=100, eliminate_duplicates=True), ("n_gen", gens), seed=seed, verbose=False)
    return res.X, -res.F[0]


# ---------------- S1: GA single objective (replicates the metaheuristic step of the base model)
with Stage("S1_GA_30runs"):
    sols = np.array([np.r_[ga(BASE, s)] for s in range(30)], dtype=float)
xs = sols[:, :6]; fs = sols[:, 6]
best = xs[np.argmax(fs)]
RES["GA"] = dict(mean=xs.mean(0).tolist(), sd=xs.std(0, ddof=1).tolist(), median=np.median(xs, 0).tolist(),
                 AITP_mean=fs.mean(), AITP_median=float(np.median(fs)), AITP_sd=fs.std(ddof=1), AITP_min=fs.min(), AITP_max=fs.max(),
                 best=best.tolist())
xstar = xs.mean(0)
d = evaluate(xstar[None, :], detail=True)
RES["GA_detail"] = {k: (float(v[0]) if v is not None else None) for k, v in d.items()}
x0 = np.array([[xstar[0], 0, 0, 0, 0, 0]])
d0 = evaluate(x0, detail=True)
RES["NoInvest_detail"] = {k: (float(v[0]) if v is not None else None) for k, v in d0.items()}
P_un = dict(BASE, alloc=0)
RES["GA_unallocated"] = {k: float(v[0]) for k, v in evaluate(xstar[None, :], P_un).items()}
RES["NoInvest_unallocated"] = {k: float(v[0]) for k, v in evaluate(x0, P_un).items()}
# local optimality check: profit along each coordinate around GA solution
prof = {}
for i, nm in enumerate(VAR_NAMES):
    grid = np.linspace(max(LB[i], xstar[i] * 0.5), xstar[i] * 1.5 + (1 if xstar[i] < 1 else 0), 41)
    Xg = np.repeat(xstar[None, :], len(grid), 0); Xg[:, i] = grid
    prof[nm] = dict(x=grid.tolist(), y=evaluate(Xg)["AITP"].tolist())
RES["profiles"] = prof

# ---------------- S2: carbon-price scenarios
with Stage("S2_carbon_price"):
    sc = []
    for tau in [0.0, 0.015, 0.03, 0.06, 0.10, 0.15]:
        P = dict(BASE, tau=tau)
        xb, fb = max((ga(P, s, 120) for s in range(5)), key=lambda t: t[1])
        r = evaluate(xb[None, :], P)
        sc.append(dict(tau=tau * 1000, x=xb.tolist(), AITP=float(fb), CF=float(r["CF"][0]), WF=float(r["WF"][0]),
                       WASTE=float(r["WASTE"][0])))
RES["carbon"] = sc

# ---------------- S2b: one-at-a-time sensitivity (+/-20%) at GA optimum, re-optimised
with Stage("S2b_sensitivity"):
    sens = []
    for k in ["e_f", "EFg", "c_v", "c_dis", "eps_m", "h_m", "A_m", "beta0", "delta", "sigma"]:
        for f in [0.8, 1.2]:
            P = dict(BASE); P[k] = BASE[k] * f
            xb, fb = max((ga(P, s, 100) for s in range(3)), key=lambda t: t[1])
            r = evaluate(xb[None, :], P)
            sens.append(dict(param=k, factor=f, x=xb.tolist(), AITP=float(fb), CF=float(r["CF"][0]),
                             WF=float(r["WF"][0])))
RES["sensitivity"] = sens

# ---------------- S3: NSGA-II tri-objective
with Stage("S3_NSGA2"):
    fronts = []
    for s in range(5):
        r = minimize(TriObj(BASE), NSGA2(pop_size=120), ("n_gen", 250), seed=s, verbose=False)
        fronts.append((r.X, r.F))
allF = np.vstack([f for _, f in fronts])
ideal, nadir = allF.min(0), allF.max(0)
ref = np.ones(3) * 1.1
hv = HV(ref_point=ref)
hvs = [float(hv((f - ideal) / (nadir - ideal))) for _, f in fronts]
X3, F3 = fronts[int(np.argmax(hvs))]
# TOPSIS with entropy weights for compromise
M = np.column_stack([-F3[:, 0], F3[:, 1], F3[:, 2]])   # profit (benefit), CF, WF (cost)
Pn = M / M.sum(0)
Ent = -(Pn * np.log(Pn + 1e-12)).sum(0) / np.log(len(M))
wts = (1 - Ent) / (1 - Ent).sum()
N = M / np.sqrt((M ** 2).sum(0)); V = N * wts
best_v = np.array([V[:, 0].max(), V[:, 1].min(), V[:, 2].min()])
worst_v = np.array([V[:, 0].min(), V[:, 1].max(), V[:, 2].max()])
dp, dn = np.linalg.norm(V - best_v, axis=1), np.linalg.norm(V - worst_v, axis=1)
cc = dn / (dp + dn); k = int(np.argmax(cc))
ext = dict(max_profit=int(np.argmin(F3[:, 0])), min_CF=int(np.argmin(F3[:, 1])), min_WF=int(np.argmin(F3[:, 2])), topsis=k)
RES["NSGA2"] = dict(hv=hvs, n_front=len(F3), entropy_w=wts.tolist(),
                    points={nm: dict(x=X3[i].tolist(), AITP=float(-F3[i, 0]), CF=float(F3[i, 1]), WF=float(F3[i, 2]),
                                     WASTE=float(evaluate(X3[i][None, :])["WASTE"][0])) for nm, i in ext.items()},
                    front=np.column_stack([-F3[:, 0], F3[:, 1], F3[:, 2]]).tolist(), X=X3.tolist())

# ---------------- S4: ML surrogate over decisions + uncertain parameters
UNC = dict(e_f=(1.2, 3.6), EFg=(0.4, 0.8), tau=(0.0, 0.10), c_dis=(2.5, 7.5), c_gr=(8.0, 10.0),
           c_v=(0.02, 0.10), c_eff=(2000, 6000), beta0=(0.05, 0.25), delta=(0.01, 0.04), sigma=(0.01, 0.04))
IN_NAMES = VAR_NAMES + list(UNC)
lo = np.r_[LB, [v[0] for v in UNC.values()]]; hi = np.r_[UB, [v[1] for v in UNC.values()]]


def exact_full(Z):
    P = dict(BASE)
    for j, kname in enumerate(UNC): P[kname] = Z[:, 6 + j]
    r = evaluate(Z[:, :6], P)
    return np.column_stack([r["AITP"] / 1e6, r["CF"], r["WF"]])


with Stage("S4a_sampling_exact"):
    Z = qmc.scale(qmc.LatinHypercube(d=len(lo), seed=1).random(20_000), lo, hi)
    Y = exact_full(Z)
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
ntr = 16_000
Ztr, Zte, Ytr, Yte = Z[:ntr], Z[ntr:], Y[:ntr], Y[ntr:]
OUTS = ["AITP", "CF", "WF"]
surr, metr = {}, {}
with Stage("S4b_train_XGB"):
    for j, o in enumerate(OUTS):
        m = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9, n_jobs=2, random_state=0)
        m.fit(Ztr, Ytr[:, j]); surr[o] = m
with Stage("S4c_train_MLP"):
    sc_in = StandardScaler().fit(Ztr); mlp = {}
    for j, o in enumerate(OUTS):
        ys = StandardScaler().fit(Ytr[:, [j]])
        m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400, early_stopping=True, random_state=0)
        m.fit(sc_in.transform(Ztr), ys.transform(Ytr[:, [j]]).ravel()); mlp[o] = (m, ys)
for j, o in enumerate(OUTS):
    px = surr[o].predict(Zte)
    m, ys = mlp[o]; pm = ys.inverse_transform(m.predict(sc_in.transform(Zte))[:, None]).ravel()
    metr[o] = dict(XGB_R2=r2_score(Yte[:, j], px), XGB_MAPE=mean_absolute_percentage_error(Yte[:, j], px),
                   MLP_R2=r2_score(Yte[:, j], pm), MLP_MAPE=mean_absolute_percentage_error(Yte[:, j], pm))
RES["surrogate_metrics"] = metr

# ---------------- S5: explainability — SHAP on surrogate vs Sobol on exact model
import shap
with Stage("S5a_SHAP"):
    shp = {}
    Zs = Zte[:2000]
    for o in OUTS:
        sv = shap.TreeExplainer(surr[o]).shap_values(Zs)
        shp[o] = np.abs(sv).mean(0)
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_an
with Stage("S5b_Sobol_exact"):
    prob = dict(num_vars=len(lo), names=IN_NAMES, bounds=np.column_stack([lo, hi]).tolist())
    Zsob = sobol_sample.sample(prob, 2048, calc_second_order=False, seed=2)
    Ysob = exact_full(Zsob)
    sob = {o: sobol_an.analyze(prob, Ysob[:, j], calc_second_order=False, seed=3) for j, o in enumerate(OUTS)}
RES["explain"] = {o: dict(shap=(shp[o] / shp[o].sum()).tolist(), ST=np.clip(sob[o]["ST"], 0, None).tolist(),
                          S1=sob[o]["S1"].tolist(),
                          spearman=float(spearmanr(shp[o], sob[o]["ST"]).correlation),
                          spearman_top8=float(spearmanr(shp[o][np.argsort(-sob[o]['ST'])[:8]], sob[o]['ST'][np.argsort(-sob[o]['ST'])[:8]]).correlation)) for o in OUTS}
RES["in_names"] = IN_NAMES
RES["n_sobol_evals"] = int(len(Zsob))

# ---------------- S6: robust (expected-value under parameter uncertainty) tri-objective optimisation
rng = np.random.default_rng(7)
NDRAW = 64
Ud = np.column_stack([rng.uniform(*UNC[kk], NDRAW) for kk in UNC])


class Robust(Problem):
    def __init__(self, mode): super().__init__(n_var=6, n_obj=3, xl=LB, xu=UB); self.mode = mode
    def _evaluate(self, X, out, *a, **k):
        n = len(X)
        Zr = np.column_stack([np.repeat(X, NDRAW, 0), np.tile(Ud, (n, 1))])
        if self.mode == "exact": Yr = exact_full(Zr)
        else: Yr = np.column_stack([surr[o].predict(Zr) for o in OUTS])
        Ym = Yr.reshape(n, NDRAW, 3).mean(1)
        out["F"] = np.column_stack([-Ym[:, 0], Ym[:, 1], Ym[:, 2]])


rob = {}
for mode in ["exact", "surrogate"]:
    with Stage(f"S6_robust_{mode}"):
        r = minimize(Robust(mode), NSGA2(pop_size=100), ("n_gen", 150), seed=11, verbose=False)
    rob[mode] = r
# validate surrogate-found robust front with exact model
Fs_exact = Robust("exact"); out = {}; Fs_exact._evaluate(rob["surrogate"].X, out); Fsv = out["F"]
allR = np.vstack([rob["exact"].F, Fsv]); idr, nar = allR.min(0), allR.max(0)
hvr = HV(ref_point=np.ones(3) * 1.1)
RES["robust"] = dict(hv_exact=float(hvr((rob["exact"].F - idr) / (nar - idr))),
                     hv_surrogate_validated=float(hvr((Fsv - idr) / (nar - idr))),
                     surrogate_pred_err_profit=float(np.mean(np.abs(rob["surrogate"].F[:, 0] - Fsv[:, 0]) / np.abs(Fsv[:, 0]))),
                     surrogate_pred_err_CF=float(np.mean(np.abs(rob["surrogate"].F[:, 1] - Fsv[:, 1]) / Fsv[:, 1])),
                     surrogate_pred_err_WF=float(np.mean(np.abs(rob["surrogate"].F[:, 2] - Fsv[:, 2]) / Fsv[:, 2])),
                     front_exact=(rob["exact"].F * [-1, 1, 1]).tolist(), front_surr_validated=(Fsv * [-1, 1, 1]).tolist(),
                     n_draws=NDRAW)

# ---------------- S7: computational footprint of the AI workflow (Green Algorithms, Lannelongue et al. 2021)
Pc, nc, Pmem_per_gb, PUE, EF = 12.0, 2, 0.3725, 1.67, 0.727   # W/core (assumed), cores, W/GB, PUE, kg/kWh
foot = {}
for nm, s in STAGES.items():
    t_h = s["wall_s"] / 3600; uc = min(s["cpu_s"] / max(s["wall_s"], 1e-9) / nc, 1.0)
    E_kwh = t_h * (nc * Pc * uc + s["peak_rss_gb"] * Pmem_per_gb) * PUE / 1000
    foot[nm] = dict(**s, usage=uc, energy_kWh=E_kwh, gCO2e=E_kwh * EF * 1000)
RES["ai_footprint"] = foot
RES["ai_assump"] = dict(Pc=Pc, nc=nc, Pmem=Pmem_per_gb, PUE=PUE, EF=EF)
import platform, os
RES["hardware"] = dict(cpu=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "), cores=os.cpu_count(), mem_gb=os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")/1e9, python=platform.python_version())
json.dump(RES, open("results.json", "w"), indent=1, default=float)
print("done")
