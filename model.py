"""
Four-echelon dairy (cheese) supply chain: farm (F) -> milk processor (P) -> cheese plant (M) -> retailer (R).
Expected average integrated total profit (renewal-reward), carbon footprint and water footprint
per kg cheese sold. Random defect fractions a_f, a_p, a_m ~ independent Uniform(0, b_*);
expectations by Gauss-Legendre quadrature. Fully vectorised over candidate solutions.

Decision variables x = [Q, G0, G1, mu, gam, G2]
  Q   : cheese production lot per cycle (kg)
  G0  : annualised farm methane-mitigation investment (k$/yr)
  G1  : annualised processor renewable-energy investment (k$/yr)
  mu  : annualised whey-valorisation (biorefinery) investment (k$/yr)
  gam : annualised cheese-whey-wastewater treatment investment (k$/yr)
  G2  : annualised cheese-plant energy-efficiency investment (k$/yr)
"""
import numpy as np

VAR_NAMES = ["Q", "G0", "G1", "mu", "gam", "G2"]
LB = np.array([5_000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
UB = np.array([200_000.0, 300.0, 300.0, 300.0, 300.0, 300.0])

BASE = dict(
    # demand, rates, conversion
    Dc=500_000.0,        # cheese demand, kg/yr
    Pm=800_000.0,        # cheese production rate, kg/yr
    Ps=10_000_000.0,     # processor milk production rate, kg/yr
    kappa=10.0,          # kg milk per kg cheese
    n=4,                 # shipments plant->retailer per cycle
    b_f=0.06, b_p=0.04, b_m=0.05,   # upper bounds of defect fractions
    # prices ($/kg)
    c_gf=0.40, c_df=0.05, c_gp=0.50, c_dp=0.10, c_gm=7.0, c_gr=9.0,
    c_rw=4.0, c_fd=0.30, c_ad=0.0,          # rework, feed, anaerobic digestion outlets for defective cheese
    phi1=0.5, phi2=0.3, phi3=0.2,           # shares of defective cheese to rework/feed/AD
    r_rw=1.0, r_fd=0.05, r_ad=0.10,         # handling costs of defective cheese ($/kg)
    # unit costs
    b_fc=0.30, b_pc=0.03, b_mc=0.80,        # production/processing cost ($/kg)
    y_f=0.002, y_p=0.003, y_m=0.02,         # inspection cost ($/kg)
    t_f=0.01,                               # milk haulage ($/kg)
    A_f=200.0, A_p=1500.0, A_m=4000.0, A_r=300.0,   # setup / ordering costs ($)
    h_f=0.02, h_p=0.03, h_m=0.60, h_r=0.80,         # holding ($/kg/yr)
    p_e=0.10,                               # electricity price $/kWh
    p_w=0.5,                                # fresh water price $/m3
    c_tr=0.8,                               # wastewater treatment O&M $/m3 treated
    c_dis=5.0,                              # effluent charge $/m3 untreated
    c_v=0.06, c_wd=0.01,                    # valorised whey net value; disposal cost ($/kg whey)
    tau=0.03,                               # carbon price $/kg CO2e (= $30/t)
    cap_f=10_000_000.0, cap_p=200_000.0, cap_m=600_000.0,  # annual allowances kg CO2e/yr
    # environmental factors
    e_f=2.4,            # farm-gate kg CO2e/kg milk (FAO 2010 global average)
    EFg=0.727,          # grid kg CO2/kWh (CEA v20, FY2023-24)
    eps_p=0.03,         # processor electricity kWh/kg milk
    eps_m=1.2,          # cheese plant energy kWh/kg cheese (electricity-equivalent)
    eps_h=0.15,         # refrigerated storage kWh per kg-yr held (plant and retail)
    E_cip=800.0, V_cip=40.0,   # energy (kWh) and water (m3) per production setup (CIP)
    e_t=0.05,           # transport kg CO2e/kg cheese plant->retail
    e_wd=0.10,          # kg CO2e per kg unvalorised whey (disposal/land spreading)
    B0=0.25, MCF=0.3, GWP_ch4=27.0,   # wastewater CH4: kg CH4/kg COD (IPCC), correction factor (assumed), AR6 GWP100 non-fossil
    w_f=85.0,           # farm blue water L/kg milk (derived from Kim et al. 2013)
    w_p=1.5, w_m=1.0,   # process water L/kg milk at processor / plant (excl. CIP)
    v_m=4.0,            # cheese-whey wastewater, L per kg milk processed (Carvalho et al. 2013)
    s_c=0.632, s_w=0.065,   # dry-matter fraction of cheese (Kim et al. 2013, cheddar) and whey (assumed)
    alloc=1,                # 1 = milk-solids allocation to valorised whey (IDF 2015); 0 = none
    c_eff=4000.0, c_max=250.0, c_nat=0.0,   # COD mg/L: effluent, discharge limit, natural
    # investment response (per k$/yr)
    beta0=0.15, eta0=0.02,     # farm: max 15% cut in farm GHG (30% enteric x ~half share)
    beta1=0.60, eta1=0.02,     # processor renewable share
    beta2=0.35, eta2=0.03,     # plant energy-efficiency
    delta=0.02,                # whey valorisation efficiency
    sigma=0.02,                # wastewater treatment efficiency
)

import os
_GL_N = int(os.environ.get('GL_N', 6))
_x, _w = np.polynomial.legendre.leggauss(_GL_N)
_u = 0.5 * (_x + 1.0)            # nodes on [0,1]
_wu = 0.5 * _w                   # weights on [0,1]
_U = np.array(np.meshgrid(_u, _u, _u, indexing="ij")).reshape(3, -1)          # (3,K)
_W = np.prod(np.array(np.meshgrid(_wu, _wu, _wu, indexing="ij")).reshape(3, -1), axis=0)  # (K,)


def _p(P, k):
    v = P[k]
    return v[:, None] if isinstance(v, np.ndarray) else v


def evaluate(X, P=None, detail=False):
    """X: (n,6). P: dict of params (scalars or (n,) arrays). Returns dict of (n,) arrays."""
    P = dict(BASE if P is None else P)
    X = np.atleast_2d(np.asarray(X, float))
    Q, G0, G1, mu, gam, G2 = [X[:, i][:, None] for i in range(6)]
    g = lambda k: _p(P, k)
    a_f = g("b_f") * _U[0][None, :]
    a_p = g("b_p") * _U[1][None, :]
    a_m = g("b_m") * _U[2][None, :]

    kap = g("kappa")
    Qg = (1 - a_m) * Q                  # good cheese per cycle
    T = Qg / g("Dc")                    # cycle length (yr)
    Mm = kap * Q                        # milk used by plant
    Rp = Mm / (1 - a_p)                 # milk processed (= good raw milk received)
    Rf = Rp / (1 - a_f)                 # raw milk collected at farm
    whey = (kap - 1) * Q

    # investment effect functions
    red0 = g("beta0") * (1 - np.exp(-g("eta0") * G0))
    ren1 = g("beta1") * (1 - np.exp(-g("eta1") * G1))
    eff2 = g("beta2") * (1 - np.exp(-g("eta2") * G2))
    Vw = 1 - np.exp(-g("delta") * mu)
    Wt = 1 - np.exp(-g("sigma") * gam)

    # ---------- emissions per cycle (kg CO2e)
    E_f = g("e_f") * Rf * (1 - red0)
    E_p = g("eps_p") * Rp * g("EFg") * (1 - ren1)
    hold_m = Qg / 2 * (1 - g("Dc") / ((1 - a_m) * g("Pm"))) * T + a_m * Q * (Q / g("Pm")) / 2   # kg-yr
    hold_r = Qg / (2 * g("n")) * T
    E_m_energy = (g("eps_m") * Q + g("E_cip") + g("eps_h") * hold_m) * g("EFg") * (1 - eff2)
    WWvol = g("v_m") * Mm / 1000.0                          # m3
    E_m = E_m_energy + g("e_wd") * whey * (1 - Vw) + (g("c_eff") / 1000.0 * g("B0") * g("MCF") * g("GWP_ch4")) * WWvol * (1 - Wt) + g("e_t") * Qg
    E_r = g("eps_h") * hold_r * g("EFg")
    # milk-solids allocation between cheese and valorised whey (co-product); disposal burdens stay with cheese
    AF = 1 - g("alloc") * (g("s_w") * whey * Vw) / (g("s_c") * Q + g("s_w") * whey * Vw)
    E_m_alloc = E_m - E_m_energy + E_m_energy * AF
    E_tot = E_f * AF + E_p * AF + E_m_alloc + E_r

    # ---------- water per cycle (L)
    blue = g("w_f") * Rf + g("w_p") * Rp + g("w_m") * Mm + g("V_cip") * 1000.0
    grey = WWvol * 1000.0 * ((1 - Wt) * g("c_eff") + Wt * g("c_max")) / (g("c_max") - g("c_nat"))
    blue = blue * AF
    WF_tot = blue + grey

    # ---------- profits per cycle ($)
    TP_f = (g("c_gf") * Rp + g("c_df") * a_f * Rf
            - (g("A_f") + (g("b_fc") + g("y_f") + g("t_f")) * Rf + g("h_f") * Rf * (Rf / g("Ps")) / 2)
            - g("tau") * (E_f - g("cap_f") * T) - 1000 * G0 * T)
    hold_p = Mm * (Q / g("Pm")) * np.maximum(1 - kap * g("Pm") / ((1 - a_p) * g("Ps")), 0) / 2
    TP_p = (g("c_gp") * Mm + g("c_dp") * a_p * Rp
            - (g("A_p") + (g("c_gf") + g("b_pc") + g("y_p")) * Rp + g("h_p") * hold_p
               + g("p_e") * g("eps_p") * Rp * (1 - ren1) + g("p_w") * g("w_p") * Rp / 1000)
            - g("tau") * (E_p - g("cap_p") * T) - 1000 * G1 * T)
    dq = a_m * Q
    TP_m = (g("c_gm") * Qg + dq * (g("phi1") * (g("c_rw") - g("r_rw")) + g("phi2") * (g("c_fd") - g("r_fd"))
                                   + g("phi3") * (g("c_ad") - g("r_ad")))
            + g("c_v") * whey * Vw
            - (g("A_m") + g("c_gp") * Mm + (g("b_mc") + g("y_m")) * Q + g("h_m") * hold_m
               + g("p_e") * (g("eps_m") * Q + g("E_cip") + g("eps_h") * hold_m) * (1 - eff2)
               + g("p_w") * (g("w_m") * Mm / 1000 + g("V_cip"))
               + g("c_wd") * whey * (1 - Vw)
               + g("c_tr") * WWvol * Wt + g("c_dis") * WWvol * (1 - Wt))
            - g("tau") * (E_m - g("cap_m") * T) - 1000 * (mu + gam + G2) * T)
    TP_r = ((g("c_gr") - g("c_gm")) * Qg - g("A_r") * g("n") - g("h_r") * hold_r
            - g("p_e") * g("eps_h") * hold_r)
    TP = TP_f + TP_p + TP_m + TP_r

    ex = lambda Z: (Z * _W[None, :]).sum(axis=1)
    ET, EQg = ex(T), ex(Qg)
    out = dict(
        AITP=ex(TP) / ET,                 # $/yr
        CF=ex(E_tot) / EQg,               # kg CO2e / kg cheese
        WF=ex(WF_tot) / EQg,              # L / kg cheese
        WF_blue=ex(blue) / EQg, WF_grey=ex(grey) / EQg,
        WASTE=ex(whey * (1 - Vw) + dq * g("phi3")) / EQg,   # kg unvalorised whey + AD cheese per kg
    )
    if detail:
        for k, Z in dict(TP_f=TP_f, TP_p=TP_p, TP_m=TP_m, TP_r=TP_r).items():
            out[k] = ex(Z) / ET
        for k, Z in dict(E_f=E_f, E_p=E_p, E_m=E_m, E_r=E_r, E_m_energy=E_m_energy).items():
            out[k] = ex(Z) / EQg
        out["E_whey"] = ex(g("e_wd") * whey * (1 - Vw)) / EQg
        out["E_ww"] = ex((g("c_eff") / 1000.0 * g("B0") * g("MCF") * g("GWP_ch4")) * WWvol * (1 - Wt)) / EQg
        out["AF"] = ex(AF) / _W.sum()
        out["E_f_alloc"] = ex(E_f * AF) / EQg
        out["E_tr"] = ex(g("e_t") * Qg) / EQg
        out["CF_ann_t"] = out["CF"] * g("Dc") / 1000 if not isinstance(P["Dc"], np.ndarray) else None
    return out


if __name__ == "__main__":
    x0 = np.array([[40_000, 0, 0, 0, 0, 0], [40_000, 50, 50, 50, 50, 50]])
    r = evaluate(x0, detail=True)
    for k, v in r.items():
        print(k, np.round(v, 4) if v is not None else v)
