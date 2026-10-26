import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import evaluate, BASE

R = json.load(open("results.json"))
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1d", "#6b6a63", "#e1e0d9"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2, "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})
OUT = "figures/"
import os; os.makedirs(OUT, exist_ok=True)

# Fig 2: profit profiles around the GA optimum
lab = {"Q": "$Q$ (kg)", "G0": "$G_0$ (k\\$/yr)", "G1": "$G_1$ (k\\$/yr)", "mu": "$\\mu$ (k\\$/yr)",
       "gam": "$\\gamma$ (k\\$/yr)", "G2": "$G_2$ (k\\$/yr)"}
xs = R["GA"]["mean"]
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.2))
for i, (k, v) in enumerate(R["profiles"].items()):
    a = ax.flat[i]
    a.plot(v["x"], np.array(v["y"]) / 1e6, color=C[0])
    a.plot([xs[i]], [R["GA"]["AITP_mean"] / 1e6], "o", ms=6, color=C[1], mec="white", mew=1.5)
    a.set_xlabel(lab[k]); a.set_ylabel("E[AITP] (M\\$/yr)" if i % 3 == 0 else "")
    a.ticklabel_format(axis="y", useOffset=False)
fig.tight_layout(); fig.savefig(OUT + "fig_profiles.pdf"); plt.close()

# Fig 3: carbon price scenarios
cs = R["carbon"]; tau = [c["tau"] for c in cs]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
ax[0].plot(tau, [c["AITP"] / 1e6 for c in cs], "-o", color=C[0], ms=5); ax[0].set_ylabel("E[AITP] (M\\$/yr)")
ax[1].plot(tau, [c["CF"] for c in cs], "-o", color=C[1], ms=5); ax[1].set_ylabel("CF (kg CO$_2$e/kg)")
for j, (nm, lb) in enumerate([(1, "$G_0$"), (3, "$\\mu$"), (4, "$\\gamma$"), (5, "$G_2$")]):
    ax[2].plot(tau, [c["x"][nm] for c in cs], "-o", color=C[[0, 1, 2, 6][j]], ms=4, label=lb)
ax[2].set_ylabel("Investment (k\\$/yr)"); ax[2].legend(ncol=2, fontsize=7)
for a in ax: a.set_xlabel("Carbon price (\\$/t CO$_2$e)")
fig.tight_layout(); fig.savefig(OUT + "fig_carbon.pdf"); plt.close()

# Fig 4: Pareto front projections
F = np.array(R["NSGA2"]["front"]); pts = R["NSGA2"]["points"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
sc = ax[0].scatter(F[:, 1], F[:, 0] / 1e6, c=F[:, 2], cmap="Blues_r", s=14, edgecolor="white", linewidth=0.4)
ax[0].set_xlabel("CF (kg CO$_2$e/kg cheese)"); ax[0].set_ylabel("E[AITP] (M\\$/yr)")
cb = fig.colorbar(sc, ax=ax[0]); cb.set_label("WF (L/kg)")
ax[1].scatter(F[:, 2], F[:, 0] / 1e6, c=F[:, 1], cmap="Oranges_r", s=14, edgecolor="white", linewidth=0.4)
ax[1].set_xlabel("WF (L/kg cheese)")
mk = {"max_profit": ("A", "s"), "topsis": ("B", "D"), "min_CF": ("C", "^"), "min_WF": ("D", "v")}
for k, (t, m) in mk.items():
    p = pts[k]
    for a, xv in [(ax[0], p["CF"]), (ax[1], p["WF"])]:
        a.plot(xv, p["AITP"] / 1e6, m, ms=8, color=INK, mfc="none", mew=1.4)
        a.annotate(t, (xv, p["AITP"] / 1e6), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
fig.tight_layout(); fig.savefig(OUT + "fig_pareto.pdf"); plt.close()

# Fig 5: CF contribution breakdown, allocated and unallocated
cases = [("No investment", [xs[0], 0, 0, 0, 0, 0]), ("GA optimum", xs),
         ("B: TOPSIS", pts["topsis"]["x"]), ("C: min-CF", pts["min_CF"]["x"])]
cats = ["Farm milk", "Processor energy", "Plant energy", "Whey disposal", "Wastewater CH$_4$", "Transport + retail"]
brk = {}
for alloc in [1, 0]:
    rows = []
    for nm, x in cases:
        d = evaluate(np.array([x]), dict(BASE, alloc=alloc), detail=True)
        AF = d["AF"][0] if alloc else 1.0
        rows.append([d["E_f"][0] * AF, d["E_p"][0] * AF, d["E_m_energy"][0] * AF, d["E_whey"][0], d["E_ww"][0],
                     d["E_tr"][0] + d["E_r"][0]])
    brk[alloc] = np.array(rows)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for a, alloc, ttl in [(ax[0], 1, "Milk-solids allocation"), (ax[1], 0, "No allocation")]:
    B = brk[alloc]; bottom = np.zeros(len(cases))
    for j in range(len(cats)):
        a.bar(range(len(cases)), B[:, j], bottom=bottom, color=C[j], width=0.6, edgecolor="white", linewidth=1,
              label=cats[j])
        bottom += B[:, j]
    for i, t in enumerate(bottom): a.text(i, t + 0.4, f"{t:.1f}", ha="center", fontsize=8, color=INK)
    a.set_xticks(range(len(cases))); a.set_xticklabels([c[0] for c in cases], rotation=15, fontsize=7.5)
    a.set_title(ttl, fontsize=9, color=INK); a.grid(axis="x", visible=False)
ax[0].set_ylabel("CF (kg CO$_2$e/kg cheese)")
ax[1].legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.55, 1))
fig.tight_layout(); fig.savefig(OUT + "fig_cf_breakdown.pdf"); plt.close()
extra = {k: float(evaluate(np.array([pts[k]["x"]]), dict(BASE, alloc=0))["CF"][0]) for k in pts}
json.dump(dict(**{str(k): v.tolist() for k, v in brk.items()}, unalloc_points=extra), open("cf_breakdown.json", "w"))

# Fig 6: SHAP (surrogate) vs Sobol total-order (exact)
names = R["in_names"]
pretty = {"Q": "$Q$", "G0": "$G_0$", "G1": "$G_1$", "mu": "$\\mu$", "gam": "$\\gamma$", "G2": "$G_2$", "e_f": "$e_f$",
          "EFg": "$EF_g$", "tau": "$\\tau$", "c_dis": "$c_{dis}$", "c_gr": "$c_{gr}$", "c_v": "$c_v$",
          "c_eff": "$c_{eff}$", "beta0": "$\\beta_0$", "delta": "$\\delta$", "sigma": "$\\sigma$"}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.4))
for a, o, ttl in zip(ax, ["AITP", "CF", "WF"], ["E[AITP]", "CF", "WF"]):
    e = R["explain"][o]; st = np.array(e["ST"]); st = st / st.sum(); sh = np.array(e["shap"])
    idx = np.argsort(st)[::-1][:8][::-1]
    y = np.arange(len(idx))
    a.barh(y + 0.2, sh[idx], height=0.38, color=C[0], label="SHAP (surrogate)")
    a.barh(y - 0.2, st[idx], height=0.38, color=C[1], label="Sobol $S_T$ (exact)")
    a.set_yticks(y); a.set_yticklabels([pretty[names[i]] for i in idx])
    a.set_title(f"{ttl}  ($\\rho$={e['spearman']:.2f})", fontsize=9, color=INK); a.grid(axis="y", visible=False)
    a.set_xlabel("Normalised importance")
ax[0].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "fig_explain.pdf"); plt.close()

# Fig 7: robust fronts + compute
fe = np.array(R["robust"]["front_exact"]); fs = np.array(R["robust"]["front_surr_validated"])
st = R["ai_footprint"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.2, 1]))
ax[0].scatter(fe[:, 1], fe[:, 0], s=14, color=C[0], edgecolor="white", linewidth=0.4, label="Exact-model NSGA-II")
ax[0].scatter(fs[:, 1], fs[:, 0], s=14, marker="^", color=C[1], edgecolor="white", linewidth=0.4,
              label="Surrogate NSGA-II (re-evaluated exactly)")
ax[0].set_xlabel("Expected CF (kg CO$_2$e/kg)"); ax[0].set_ylabel("Expected E[AITP] (M\\$/yr)"); ax[0].legend(fontsize=7)
lbl = ["Exact", "Surrogate"]
w_ex = st["S6_robust_exact"]["wall_s"]
w_su = [st["S4a_sampling_exact"]["wall_s"], st["S4b_train_XGB"]["wall_s"], st["S6_robust_surrogate"]["wall_s"]]
ax[1].bar([0], [w_ex], color=C[0], width=0.55)
bt = 0
for j, (v, l) in enumerate(zip(w_su, ["Sampling", "Training", "Optimisation"])):
    ax[1].bar([1], [v], bottom=bt, color=[C[2], C[6], C[1]][j], width=0.55, edgecolor="white", linewidth=1, label=l); bt += v
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(lbl); ax[1].set_ylabel("Wall-clock time (s)")
ax[1].legend(fontsize=7); ax[1].grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(OUT + "fig_robust.pdf"); plt.close()
print("figs ok")
