#!/usr/bin/env python3
from __future__ import annotations

import os

# single-threaded BLAS/OpenMP; must be set before numpy/sklearn/xgboost are imported
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime as dt
import io
import json
import platform
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    BOOSTER_NAME = "XGBoost"
except Exception:   # ImportError, or XGBoostError when libomp is missing
    XGBRegressor = None
    BOOSTER_NAME = "SklearnGB"

warnings.filterwarnings("ignore")

SEED = 42
KB_EV = 8.617333262e-5          # eV/K
ATOMS_PER_FU = 5

T_WINDOW = (0.80, 1.00)
BARTEL_TAU_MAX = 4.18           # tau < 4.18 -> predicted perovskite (Bartel et al. 2019)
HULL_STABLE = 0.025             # eV/atom
PV_GAP = (1.0, 1.8)             # eV
FRACTIONS = np.round(np.linspace(0.05, 0.45, 9), 2)

A_SITES = ["Cs", "Rb", "K", "Na"]
B_SITES = ["Pb", "Sn", "Ge"]
HALIDES = ["I", "Br", "Cl", "F"]

# Shannon radii (A): A-site CN=12, B- and X-site CN=6. Sn2+ CN=6: 1.10 is the usual literature choice.
RADII = {"Cs": 1.88, "Rb": 1.72, "K": 1.64, "Na": 1.39,
         "Pb": 1.19, "Sn": 1.10, "Ge": 0.73,
         "I": 2.20, "Br": 1.96, "Cl": 1.81, "F": 1.33}
ENEG = {"Cs": 0.79, "Rb": 0.82, "K": 0.82, "Na": 0.93,
        "Pb": 2.33, "Sn": 1.96, "Ge": 2.01,
        "I": 2.66, "Br": 2.96, "Cl": 3.16, "F": 3.98}

FEATURES = ["r_A", "r_B", "r_X", "en_A", "en_B", "en_X",
            "t", "mu", "tau", "dEN_BX", "dEN_AX"]

FORMULA_RE = re.compile(r"^([A-Z][a-z]?)([A-Z][a-z]?)([A-Z][a-z]?)3$")
NA_TOKENS = ["NA", "N/A", "n/a", "N/a", "n/A", "na", "Na", "none", "None", "-"]

# Pm-3m (space group 221) entries transcribed from materialsproject.org; NA = no cubic entry found
MP_DATA_CSV = """formula,ehull_eV_atom,gap_eV,material_id,source
CsPbI3,0.039,2.857,,MP_website
CsPbBr3,0.019,3.153,,MP_website
CsPbCl3,0.007,2.395,,MP_website
CsSnI3,0.022,1.266,mp-614013,MP_website
CsSnBr3,0,2.385,,MP_website
CsSnCl3,0.004,2.131,,MP_website
CsGeI3,NA,NA,,MP_website
CsGeBr3,0.013,2.123,,MP_website
CsGeCl3,NA,NA,,MP_website
RbPbI3,NA,NA,,MP_website
RbPbBr3,NA,NA,,MP_website
RbPbCl3,0.038,3.091,,MP_website
RbSnI3,NA,NA,,MP_website
RbSnBr3,NA,NA,,MP_website
RbSnCl3,NA,NA,,MP_website
RbGeI3,0.045,2.447,,MP_website
RbGeBr3,NA,NA,,MP_website
RbGeCl3,NA,NA,,MP_website
KPbI3,NA,NA,,MP_website
KPbBr3,NA,NA,,MP_website
KPbCl3,NA,NA,,MP_website
KSnI3,NA,NA,,MP_website
KSnBr3,NA,NA,,MP_website
KSnCl3,NA,NA,,MP_website
KGeI3,NA,NA,,MP_website
KGeBr3,NA,NA,,MP_website
KGeCl3,NA,NA,,MP_website
NaPbI3,NA,NA,,MP_website
NaPbBr3,NA,NA,,MP_website
NaPbCl3,NA,NA,,MP_website
NaSnI3,NA,NA,,MP_website
NaSnBr3,NA,NA,,MP_website
NaSnCl3,NA,NA,,MP_website
NaGeI3,NA,NA,,MP_website
NaGeBr3,NA,NA,,MP_website
NaGeCl3,NA,NA,,MP_website
CsPbF3,0.004,3.683,,MP_website
CsSnF3,NA,NA,,MP_website
CsGeF3,NA,NA,,MP_website
RbPbF3,0.036,3.681,,MP_website
RbSnF3,NA,NA,,MP_website
RbGeF3,NA,NA,,MP_website
KPbF3,NA,NA,,MP_website
KSnF3,NA,NA,,MP_website
KGeF3,NA,NA,,MP_website
NaPbF3,NA,NA,,MP_website
NaSnF3,NA,NA,,MP_website
NaGeF3,NA,NA,,MP_website
"""


@dataclass
class Cfg:
    ehull_max: float = 0.035     # eV/atom, applied to Ehull_T + sigma_GP
    gap_lo: float = 1.30         # eV
    gap_hi: float = 1.45
    T: float = 300.0             # K
    omega_a: float = 0.0         # mixing enthalpies (eV/f.u.) and gap bowing (eV) are assumed, not computed
    omega_x: float = 0.0
    bow_a: float = 0.0
    bow_x: float = 0.0
    out: str = "results"
    data: str | None = None


_T0 = time.time()
REPORT: list = []


def say(msg: str = "") -> None:
    print(msg, flush=True)
    REPORT.append(str(msg))


def mix(site: dict, table: dict) -> float:
    return sum(frac * table[el] for el, frac in site.items())


def descriptors(A: dict, B: dict, X: dict) -> dict:
    rA, rB, rX = mix(A, RADII), mix(B, RADII), mix(X, RADII)
    eA, eB, eX = mix(A, ENEG), mix(B, ENEG), mix(X, ENEG)
    ratio = rA / rB
    lr = np.log(ratio)
    # Bartel tau with n_A = 1: rX/rB - (1 - (rA/rB)/ln(rA/rB))
    tau = rX / rB - (1.0 - ratio / lr) if abs(lr) > 1e-9 else np.nan
    return {"r_A": rA, "r_B": rB, "r_X": rX, "en_A": eA, "en_B": eB, "en_X": eX,
            "t": (rA + rX) / (np.sqrt(2.0) * (rB + rX)),
            "mu": rB / rX, "tau": tau,
            "dEN_BX": abs(eB - eX), "dEN_AX": abs(eA - eX)}


def load_dataset(csv_path: str | None) -> pd.DataFrame:
    raw = (pd.read_csv(csv_path, na_values=NA_TOKENS) if csv_path
           else pd.read_csv(io.StringIO(MP_DATA_CSV), na_values=NA_TOKENS))
    need = {"formula", "ehull_eV_atom", "gap_eV"}
    if need - set(raw.columns):
        raise ValueError(f"Data file is missing columns: {sorted(need - set(raw.columns))}")
    for c, default in (("material_id", ""), ("source", "user")):
        if c not in raw.columns:
            raw[c] = default
    raw["material_id"] = raw["material_id"].fillna("").astype(str)
    raw["formula"] = raw["formula"].astype(str).str.strip()
    if raw["formula"].duplicated().any():
        raise ValueError(f"Duplicate formulas: {raw.loc[raw.formula.duplicated(), 'formula'].tolist()}")

    rows = []
    for _, r in raw.iterrows():
        m = FORMULA_RE.match(r["formula"])
        if not m or any(el not in RADII for el in m.groups()):
            raise ValueError(f"Cannot parse formula / unknown element: {r['formula']}")
        a, b, x = m.groups()
        e, g = r["ehull_eV_atom"], r["gap_eV"]
        if pd.isna(e) != pd.isna(g):
            raise ValueError(f"{r['formula']}: Ehull and gap must both be filled or both be NA")
        if pd.notna(e) and (e < 0 or g < 0):
            raise ValueError(f"{r['formula']}: negative Ehull or gap")
        d = descriptors({a: 1.0}, {b: 1.0}, {x: 1.0})
        rows.append({"formula": r["formula"], "A": a, "B": b, "X": x, **d,
                     "has_entry": bool(pd.notna(e)),
                     "Ehull": float(e) if pd.notna(e) else np.nan,
                     "Gap": float(g) if pd.notna(g) else np.nan,
                     "material_id": r["material_id"], "source": r["source"]})
    tab = pd.DataFrame(rows)
    expected = {a + b + x + "3" for a in A_SITES for b in B_SITES for x in HALIDES}
    missing = sorted(expected - set(tab["formula"]))
    if missing and not csv_path:
        raise ValueError(f"Built-in table is missing {missing}")
    return tab


def phase1_verdict(has: bool, ehull: float, gap: float) -> str:
    if not has:
        return "No cubic entry"
    in_pv = PV_GAP[0] <= gap <= PV_GAP[1]
    if ehull <= HULL_STABLE:
        return "Promising" if in_pv else "Stable, gap outside PV range"
    return "Metastable, gap in PV range" if in_pv else "Metastable, gap outside PV range"


def phase1_table(tab: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Formula": tab["formula"], "A-site": tab["A"], "B-site": tab["B"], "X-site": tab["X"],
        "Tolerance Factor (t)": tab["t"].round(3),
        "Bartel tau": tab["tau"].round(3),
        "Energy Above Hull (eV/atom)": tab["Ehull"],
        "Band Gap (eV)": tab["Gap"],
        "MP ID": tab["material_id"],
        "Lead-free": tab["B"] != "Pb",
        "Viability Verdict": [phase1_verdict(r.has_entry, r.Ehull, r.Gap) for r in tab.itertuples()],
    })


def phase1_stats(tab: pd.DataFrame) -> dict:
    has = tab["has_entry"]
    res = {"n_total": int(len(tab)), "n_entries": int(has.sum()), "by_site": {}}
    for col, order in (("A", A_SITES), ("B", B_SITES), ("X", HALIDES)):
        g = tab.groupby(col)["has_entry"].agg(["sum", "count"]).reindex(order)
        res["by_site"][col] = {k: f"{int(v['sum'])}/{int(v['count'])}" for k, v in g.iterrows()}
    in_t = tab["t"].between(*T_WINDOW)
    tau_ok = tab["tau"] < BARTEL_TAU_MAX
    res["t_window"] = {"n_in_window": int(in_t.sum()),
                       "entries_in_window": int((in_t & has).sum()),
                       "entries_outside_window": int((~in_t & has).sum()),
                       "outside_window_formulas": tab.loc[~in_t & has, "formula"].tolist(),
                       "no_entry_but_in_window": int((in_t & ~has).sum())}
    res["bartel"] = {"n_tau_lt_cut": int(tau_ok.sum()),
                     "entries_with_tau_lt_cut": int((tau_ok & has).sum()),
                     "entries_with_tau_ge_cut": int((~tau_ok & has).sum()),
                     "tau_ge_cut_formulas": tab.loc[~tau_ok & has, "formula"].tolist()}
    return res


def halide_trend_flags(tab: pd.DataFrame) -> list:
    """Gap should rise I < Br < Cl < F at fixed A, B; list every violation."""
    flags = []
    for (a, b), g in tab[tab["has_entry"]].groupby(["A", "B"]):
        have = [(x, float(g.loc[g["X"] == x, "Gap"].iloc[0])) for x in HALIDES if (g["X"] == x).any()]
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                if have[j][1] < have[i][1]:
                    flags.append(f"{a}{b}: gap({have[j][0]}) = {have[j][1]:.3f} is LOWER than "
                                 f"gap({have[i][0]}) = {have[i][1]:.3f}")
    return flags


def binary_entropy(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    out[m] = -(p[m] * np.log(p[m]) + (1 - p[m]) * np.log(1 - p[m]))
    return out


def build_alloys(xs=FRACTIONS, ys=FRACTIONS) -> pd.DataFrame:
    rows = []
    for x in xs:
        for y in ys:
            d = descriptors({"Cs": 1 - x, "Rb": x}, {"Sn": 1.0}, {"I": 1 - y, "Br": y})
            rows.append({"formula": f"Cs{1 - x:.2f}Rb{x:.2f}Sn(I{1 - y:.2f}Br{y:.2f})3",
                         "x_Rb": float(x), "y_Br": float(y), **d})
    return pd.DataFrame(rows)


def apply_thermo(df: pd.DataFrame, cfg: Cfg, ehull_col="Ehull_GP", gap_col="Gap_GP") -> pd.DataFrame:
    """Ehull_T = max(Ehull_0K + dH_mix/5, 0) - T*S_mix/5, all per atom."""
    x, y = df["x_Rb"].to_numpy(), df["y_Br"].to_numpy()
    s_over_k = binary_entropy(x) + 3.0 * binary_entropy(y)     # 1 A site + 3 X sites per f.u.
    dH_atom = (cfg.omega_a * x * (1 - x) + cfg.omega_x * y * (1 - y)) / ATOMS_PER_FU
    TS_atom = cfg.T * KB_EV * s_over_k / ATOMS_PER_FU
    out = df.copy()
    out["S_over_kB"] = s_over_k
    out["TS_mix_eV_atom"] = TS_atom
    out["Ehull_0K"] = np.maximum(df[ehull_col].to_numpy() + dH_atom, 0.0)
    out["Ehull_T"] = out["Ehull_0K"] - TS_atom
    out["Gap_corr"] = df[gap_col].to_numpy() - cfg.bow_a * x * (1 - x) - cfg.bow_x * y * (1 - y)
    return out


def screen_flags(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    out = df.copy()
    gap_ok = out["Gap_corr"].between(cfg.gap_lo, cfg.gap_hi)
    stab_nom = out["Ehull_T"] <= cfg.ehull_max
    stab_str = (out["Ehull_T"] + out["Ehull_GP_sigma"]) <= cfg.ehull_max
    out["pass_nominal"] = stab_nom & gap_ok
    out["pass_strict"] = stab_str & gap_ok
    out["pass_gap_1sigma"] = out["pass_strict"] & \
        ((out["Gap_corr"] - out["Gap_GP_sigma"]) >= cfg.gap_lo) & \
        ((out["Gap_corr"] + out["Gap_GP_sigma"]) <= cfg.gap_hi)
    out["verdict"] = np.select(
        [out["pass_strict"], out["pass_nominal"], gap_ok & ~stab_nom, stab_nom & ~gap_ok, stab_str & ~gap_ok],
        ["PASS (stable at +1 sigma, gap in window)", "Nominal pass only (fails at +1 sigma)",
         "Fail: unstable", "Fail: gap outside window", "Fail: gap outside window"],
        default="Fail: unstable and gap outside window")
    return out


def anchored_baseline(tab: pd.DataFrame, alloys: pd.DataFrame):
    """ML-free cross-check from the tabulated entries only: Vegard interpolation between CsSnI3
    and CsSnBr3 along I-Br, plus the mean Ehull shift of the Cs/Rb pairs sharing B and X.
    The Cs->Rb gap shift is not applied (the pairs disagree)."""
    ent = tab[tab["has_entry"]]

    def val(a, b, x, col):
        s = ent.loc[(ent.A == a) & (ent.B == b) & (ent.X == x), col]
        return float(s.iloc[0]) if len(s) else np.nan

    eI, eBr = val("Cs", "Sn", "I", "Ehull"), val("Cs", "Sn", "Br", "Ehull")
    gI, gBr = val("Cs", "Sn", "I", "Gap"), val("Cs", "Sn", "Br", "Gap")
    pairs = []
    for (b, x), g in ent.groupby(["B", "X"]):
        if {"Cs", "Rb"} <= set(g["A"]):
            cs, rb = g[g.A == "Cs"].iloc[0], g[g.A == "Rb"].iloc[0]
            pairs.append({"pair": f"Cs{b}{x}3 -> Rb{b}{x}3", "dEhull": rb.Ehull - cs.Ehull, "dGap": rb.Gap - cs.Gap})
    pairs_df = pd.DataFrame(pairs)
    dE = float(pairs_df["dEhull"].mean()) if len(pairs_df) else np.nan
    out = alloys.copy()
    x, y = out["x_Rb"].to_numpy(), out["y_Br"].to_numpy()
    out["Ehull_Anchored"] = (1 - y) * eI + y * eBr + x * dE
    out["Gap_Anchored"] = (1 - y) * gI + y * gBr
    info = {"CsSnI3": (eI, gI), "CsSnBr3": (eBr, gBr), "dEhull_CsToRb_mean": dE,
            "cs_rb_pairs": json.loads(pairs_df.to_json(orient="records")) if len(pairs_df) else []}
    return out, info


def make_models() -> dict:
    kernel = C(1.0, (1e-2, 1e2)) * RBF(3.0, (0.3, 30.0)) + WhiteKernel(1e-3, (1e-6, 1e-1))
    models = {
        "Mean-baseline": lambda: DummyRegressor(),
        "Ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 2, 20))),
        "GP": lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=SEED)),
    }
    if XGBRegressor is not None:
        models["Boost"] = lambda: XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=2,
                                               subsample=0.8, random_state=SEED, verbosity=0, n_jobs=1)
    else:
        models["Boost"] = lambda: GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                            max_depth=2, subsample=0.8, random_state=SEED)
    return models


def gp_predict(model, X):
    return model.steps[-1][1].predict(model[:-1].transform(X), return_std=True)


def cv_predict(factory, X, y, splitter, groups=None):
    pred, sd = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for tr, te in splitter.split(X, y, groups):
        m = factory().fit(X[tr], y[tr])
        if hasattr(m, "steps") and isinstance(m.steps[-1][1], GaussianProcessRegressor):
            pred[te], sd[te] = gp_predict(m, X[te])
        else:
            pred[te] = m.predict(X[te])
    return pred, sd


SCHEMES = {"LOO": None, "leave-A-out": "A", "leave-B-out": "B", "leave-X-out": "X"}


def run_cv(train: pd.DataFrame, models: dict):
    X = train[FEATURES].to_numpy(float)
    rows, preds = [], {}
    for target, unit in (("Ehull", "eV/atom"), ("Gap", "eV")):
        y = train[target].to_numpy(float)
        for mname, factory in models.items():
            for sname, gcol in SCHEMES.items():
                if gcol is None:
                    splitter, groups = LeaveOneOut(), None
                else:
                    splitter, groups = LeaveOneGroupOut(), train[gcol].to_numpy()
                pred, sd = cv_predict(factory, X, y, splitter, groups)
                err = pred - y
                r2 = r2_score(y, pred) if np.var(y) > 0 else np.nan
                cover = float(np.mean(np.abs(err) <= sd)) if np.isfinite(sd).all() else np.nan
                rows.append({"target": target, "model": mname, "scheme": sname, "n": len(y),
                             "MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
                             "R2": float(r2), "coverage_1sigma": cover, "unit": unit})
                preds[(target, mname, sname)] = (pred, sd)
    return pd.DataFrame(rows), preds


def fit_and_predict(train: pd.DataFrame, alloys: pd.DataFrame, models: dict):
    X, Xa = train[FEATURES].to_numpy(float), alloys[FEATURES].to_numpy(float)
    out, fitted = alloys.copy(), {}
    for target in ("Ehull", "Gap"):
        y = train[target].to_numpy(float)
        for mname, factory in models.items():
            m = factory().fit(X, y)
            fitted[(target, mname)] = m
            if mname == "GP":
                out[f"{target}_GP"], out[f"{target}_GP_sigma"] = gp_predict(m, Xa)
            else:
                out[f"{target}_{mname}"] = m.predict(Xa)
        cols = [f"{target}_{n}" for n in ("Ridge", "GP", "Boost")]
        out[f"{target}_spread"] = out[cols].max(axis=1) - out[cols].min(axis=1)
    return out, fitted


def applicability_domain(train: pd.DataFrame, alloys: pd.DataFrame):
    sc = StandardScaler().fit(train[FEATURES].to_numpy(float))
    Z, Za = sc.transform(train[FEATURES].to_numpy(float)), sc.transform(alloys[FEATURES].to_numpy(float))
    D = cdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    thr = float(D.min(axis=1).max())    # largest nearest-neighbour distance within the training set
    return cdist(Za, Z).min(axis=1), thr


def make_figures(tab, preds, train, alloys, fitted, cfg, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Cs": "tab:blue", "Rb": "tab:orange", "K": "tab:green", "Na": "tab:red"}
    ent, noent = tab[tab.has_entry], tab[~tab.has_entry]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axvspan(*T_WINDOW, color="tab:green", alpha=0.08, label="0.80 <= t <= 1.00")
    for a in A_SITES:
        s = ent[ent.A == a]
        if len(s):
            ax.scatter(s.t, s.Ehull, s=55, c=colors[a], label=f"{a}-based (has Pm-3m entry)", zorder=3)
    placed = []
    for r in ent.itertuples():
        clash = any(abs(r.t - t0) < 0.012 and abs(r.Ehull - e0) < 0.004 for t0, e0 in placed)
        ax.annotate(r.formula, (r.t, r.Ehull), fontsize=7, xytext=(4, -11) if clash else (4, 4),
                    textcoords="offset points")
        placed.append((r.t, r.Ehull))
    ax.scatter(noent.t, np.full(len(noent), -0.006), marker="x", c="gray", s=22,
               label=f"no Pm-3m entry in MP (n={len(noent)})", zorder=2)
    ax.axhline(HULL_STABLE, color="k", ls=":", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-0.015, max(0.06, ent.Ehull.max() * 1.2))
    ax.set_xlabel("Goldschmidt tolerance factor t")
    ax.set_ylabel("Energy above hull (eV/atom)")
    ax.set_title("Fig 1. Tolerance factor vs. cubic-phase stability (MP)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(outdir / "fig1_t_vs_ehull.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhspan(*PV_GAP, color="tab:green", alpha=0.10, label=f"PV range {PV_GAP[0]}-{PV_GAP[1]} eV")
    ax.axhspan(cfg.gap_lo, cfg.gap_hi, color="tab:green", alpha=0.25, label=f"Phase-2 window {cfg.gap_lo}-{cfg.gap_hi} eV")
    ax.axvline(HULL_STABLE, color="k", ls=":", lw=1, label=f"Ehull = {HULL_STABLE}")
    for x_, mk in zip(HALIDES, ["o", "s", "^", "D"]):
        s = ent[ent.X == x_]
        if len(s):
            ax.scatter(s.Ehull, s.Gap, marker=mk, s=60, label=f"{x_}", zorder=3)
    for r in ent.itertuples():
        ax.annotate(r.formula, (r.Ehull, r.Gap), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Energy above hull (eV/atom)"); ax.set_ylabel("Band gap, MP (eV)")
    ax.set_title("Fig 2. Stability vs. band gap of the cubic entries")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(outdir / "fig2_ehull_vs_gap.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for i, (target, unit) in enumerate((("Ehull", "eV/atom"), ("Gap", "eV"))):
        y = train[target].to_numpy(float)
        for j, sname in enumerate(SCHEMES):
            ax = axes[i, j]
            p, sd = preds[(target, "GP", sname)]
            ax.errorbar(y, p, yerr=sd, fmt="o", ms=4, capsize=2, lw=0.8)
            lim = [min(y.min(), (p - sd).min()), max(y.max(), (p + sd).max())]
            ax.plot(lim, lim, "k--", lw=1)
            ax.set_title(f"{target}, {sname}\nMAE = {np.mean(np.abs(p - y)):.3f} {unit}", fontsize=9)
            ax.set_xlabel(f"MP {target} ({unit})", fontsize=8); ax.set_ylabel("GP prediction", fontsize=8)
    fig.suptitle("Fig 3. Cross-validated parity plots, Gaussian process (bars = predicted 1 sigma)")
    fig.tight_layout(); fig.savefig(outdir / "fig3_parity_cv.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, target in zip(axes, ("Ehull", "Gap")):
        imp = np.asarray(fitted[(target, "Boost")].feature_importances_, dtype=float)
        order = np.argsort(imp)
        ax.barh(np.array(FEATURES)[order], imp[order])
        ax.set_title(f"{target}: {BOOSTER_NAME} feature importance", fontsize=10)
    fig.suptitle("Fig 4. Descriptor importance (n = %d training compounds: indicative only)" % len(train),
                 fontsize=10)
    fig.tight_layout(); fig.savefig(outdir / "fig4_feature_importance.png", dpi=200); plt.close(fig)

    xs, ys = np.sort(alloys.x_Rb.unique()), np.sort(alloys.y_Br.unique())
    grid = lambda col: alloys.pivot(index="y_Br", columns="x_Rb", values=col).loc[ys, xs].to_numpy()
    Z_stab = grid("Ehull_T") + grid("Ehull_GP_sigma")
    Z_gap = grid("Gap_corr")
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    pm = ax.pcolormesh(xs, ys, Z_stab, shading="nearest", cmap="viridis_r")
    fig.colorbar(pm, ax=ax, label="Ehull,T + sigma_GP (eV/atom)")
    try:
        c1 = ax.contour(xs, ys, Z_gap, levels=[cfg.gap_lo, cfg.gap_hi], colors="white", linestyles="--")
        ax.clabel(c1, fmt="%.2f eV", fontsize=7)
    except Exception:
        pass
    try:
        ax.contour(xs, ys, Z_stab, levels=[cfg.ehull_max], colors="red", linewidths=1.5)
    except Exception:
        pass
    ok = alloys[alloys.pass_strict]
    ax.scatter(ok.x_Rb, ok.y_Br, marker="*", s=130, c="gold", edgecolors="k", label="passes both screens", zorder=4)
    ax.set_xlabel("Rb fraction x"); ax.set_ylabel("Br fraction y")
    ax.set_title("Fig 5. Cs(1-x)Rb(x)Sn(I(1-y)Br(y))3 screen at %d K\n"
                 "(white dashed = gap window, red = Ehull,T+sigma = %.3f)" % (cfg.T, cfg.ehull_max), fontsize=9)
    if len(ok):
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(outdir / "fig5_alloy_heatmap.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.scatter(alloys.Gap_corr, alloys.Ehull_T, c="lightgray", s=18, label="81 alloys")
    ax.scatter(alloys[alloys.pass_nominal].Gap_corr, alloys[alloys.pass_nominal].Ehull_T, c="tab:orange", s=24,
               label="nominal pass")
    ax.scatter(ok.Gap_corr, ok.Ehull_T, c="tab:green", s=30, label="strict pass (+1 sigma)")
    ax.axvspan(cfg.gap_lo, cfg.gap_hi, alpha=0.15, color="tab:green")
    ax.axhline(cfg.ehull_max, color="tab:red", ls="--", lw=1)
    ax.set_xlabel("Predicted band gap (eV)"); ax.set_ylabel("Predicted Ehull,T (eV/atom)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / "fig6_screening_map.png", dpi=200); plt.close(fig)


def _spread(df: pd.DataFrame, n: int, score_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(score_col)
    chosen = [df.index[0]]
    while len(chosen) < min(n, len(df)):
        d = np.min([np.hypot(df["x_Rb"] - df.loc[c, "x_Rb"], df["y_Br"] - df.loc[c, "y_Br"]) for c in chosen], axis=0)
        chosen.append(df.index[int(np.argmax(d))])
    return df.loc[chosen]


def write_verification_queue(alloys: pd.DataFrame, path: Path, n_pass=4, n_ctrl=2) -> None:
    flag = "pass_strict" if alloys["pass_strict"].any() else "pass_nominal"
    good = _spread(alloys[alloys[flag]], n_pass, "Ehull_T").assign(expected="pass")
    bad = _spread(alloys[~alloys[flag]].assign(neg=lambda d: -d["Ehull_T"]), n_ctrl, "neg") \
        .assign(expected="fail (negative control)")
    q = pd.concat([good, bad])[["formula", "x_Rb", "y_Br", "expected", "Ehull_0K", "Ehull_T", "Gap_corr"]].copy()
    q.columns = ["formula", "x_Rb", "y_Br", "expected", "pred_Ehull_0K", "pred_Ehull_T", "pred_Gap"]
    for c in ("ehull_dft_eV_atom", "gap_dft_eV", "source", "method"):
        q[c] = ""
    q.round(4).to_csv(path, index=False)


def evaluate_verification(path: str, cfg: Cfg) -> dict:
    q = pd.read_csv(path)
    q = q[pd.to_numeric(q["ehull_dft_eV_atom"], errors="coerce").notna()].copy()
    if q.empty:
        return {"n_verified": 0, "note": "No DFT results filled in yet."}
    eh, gp = q["ehull_dft_eV_atom"].astype(float), q["gap_dft_eV"].astype(float)
    e, g = eh - q["pred_Ehull_0K"], gp - q["pred_Gap"]
    dft_pass = (eh <= cfg.ehull_max) & gp.between(cfg.gap_lo, cfg.gap_hi)
    return {"n_verified": int(len(q)), "MAE_Ehull_eV_atom": float(e.abs().mean()),
            "MAE_Gap_eV": float(g.abs().mean()), "bias_Ehull": float(e.mean()), "bias_Gap": float(g.mean()),
            "pass_fail_accuracy": float((dft_pass == q["expected"].str.startswith("pass")).mean())}


def main(cfg: Cfg) -> dict:
    np.random.seed(SEED)
    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)
    say("=" * 92)
    say("PEROVSKITE SCREEN   " + dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    say("=" * 92)

    say(f"[{time.time() - _T0:6.1f}s] Loading Materials Project data")
    tab = load_dataset(cfg.data)
    train = tab[tab.has_entry].reset_index(drop=True)
    pd.DataFrame({"formula": tab.formula, "ehull_eV_atom": tab.Ehull, "gap_eV": tab.Gap,
                  "material_id": tab.material_id, "source": tab.source}).to_csv(outdir / "mp_verified_data.csv", index=False)
    n_id = int((train.material_id != "").sum())
    say(f"  {len(tab)} compounds, {len(train)} with a cubic Pm-3m entry, {len(tab) - len(train)} with none.")
    say(f"  sources: {sorted(set(tab.source))}; mp-ID recorded for {n_id} of {len(train)} entries")

    say(f"[{time.time() - _T0:6.1f}s] PHASE 1: tolerance factors, verdicts, statistics")
    p1 = phase1_table(tab)
    p1.to_csv(outdir / "table_A1_all_48.csv", index=False)
    p1[tab.has_entry.to_numpy()].to_csv(outdir / "table_1_cubic_entries.csv", index=False)
    st = phase1_stats(tab)
    flags = halide_trend_flags(tab)

    say("\nTABLE A1 - all compounds (Ehull / gap are Materials Project values, Pm-3m entry only)")
    say(p1[["Formula", "Tolerance Factor (t)", "Bartel tau", "Energy Above Hull (eV/atom)",
            "Band Gap (eV)", "Viability Verdict"]].fillna("n/a").to_string(index=False))
    say("\nHOW MANY COMPOUNDS HAVE A CUBIC Pm-3m ENTRY, by site")
    for col, name in (("A", "A-site"), ("B", "B-site"), ("X", "X-site")):
        say(f"  {name}: " + "   ".join(f"{k} {v}" for k, v in st["by_site"][col].items()))
    tw, br = st["t_window"], st["bartel"]
    say("\nDOES THE TOLERANCE FACTOR EXPLAIN WHICH COMPOUNDS HAVE A CUBIC ENTRY?")
    say(f"  {tw['n_in_window']} of {st['n_total']} compounds have {T_WINDOW[0]} <= t <= {T_WINDOW[1]}; only "
        f"{tw['entries_in_window']} of those have an entry, so {tw['no_entry_but_in_window']} in-window compounds have none.")
    say(f"  {tw['entries_outside_window']} entries lie OUTSIDE the window: {tw['outside_window_formulas']}")
    say(f"  Bartel tau < {BARTEL_TAU_MAX}: {br['n_tau_lt_cut']} compounds pass; {br['entries_with_tau_ge_cut']} of "
        f"{st['n_entries']} entries have tau >= {BARTEL_TAU_MAX}: {br['tau_ge_cut_formulas']}")
    prec = tw["entries_in_window"] / max(tw["n_in_window"], 1)
    say(f"  precision of the t-window = {tw['entries_in_window']}/{tw['n_in_window']} = {prec:.0%}, versus a base rate of "
        f"{st['n_entries']}/{st['n_total']} = {st['n_entries'] / st['n_total']:.0%} for any compound having an entry.")
    say("  Compare that with the per-A-site entry rates printed above.")
    promising = p1.loc[p1["Viability Verdict"] == "Promising", "Formula"].tolist()
    say(f"\nPHASE-1 'Promising' (Ehull <= {HULL_STABLE}, gap in {PV_GAP}): {promising if promising else 'none'}")
    say(f"  Lead-free entries: {p1.loc[(tab.has_entry & (tab.B != 'Pb')).to_numpy(), 'Formula'].tolist()}")
    say("\nDATA-QUALITY FLAGS (halide gap should rise I < Br < Cl < F for fixed A,B):")
    if flags:
        for f in flags:
            say("  ! " + f)
        say("  Check the originating MP calculation (functional, structure) for these entries before using halide trends.")
    else:
        say("  none")

    say(f"[{time.time() - _T0:6.1f}s] PHASE 2: cross-validating models on {len(train)} cubic entries (n is small)")
    models = make_models()
    cv_table, preds = run_cv(train, models)
    cv_table.round(4).to_csv(outdir / "cv_metrics.csv", index=False)
    say("\nCROSS-VALIDATION  (MAE; lower is better.  'Mean-baseline' = always predict the training mean)")
    for target, unit in (("Ehull", "eV/atom"), ("Gap", "eV")):
        sub = cv_table[cv_table.target == target]
        say(f"\n  {target} MAE ({unit})")
        say(sub.pivot_table(index="model", columns="scheme", values="MAE")[list(SCHEMES)].round(3).to_string())
        say(f"  {target} R^2")
        say(sub.pivot_table(index="model", columns="scheme", values="R2")[list(SCHEMES)].round(2).to_string())
    gpc = cv_table[cv_table.model == "GP"].groupby("scheme")["coverage_1sigma"].mean()
    say("\n  GP uncertainty check: fraction of held-out points inside +/-1 sigma "
        "(68% if well calibrated):  " + "  ".join(f"{k} {v:.0%}" for k, v in gpc.items()))
    if gpc.mean() < 0.60:
        say("  ! GP sigma is overconfident (coverage well below 68%); Ehull,T + sigma_GP is not a calibrated bound.")

    say(f"[{time.time() - _T0:6.1f}s] Fitting final models on all cubic entries and predicting 81 alloys")
    alloys0 = build_alloys()
    alloys, fitted = fit_and_predict(train, alloys0, models)
    alloys, base_info = anchored_baseline(tab, alloys)
    nn, thr = applicability_domain(train, alloys)
    alloys["nn_dist"] = nn
    alloys["in_domain"] = nn <= thr
    for t_ in ("Ehull", "Gap"):
        say(f"  GP kernel for {t_}: {fitted[(t_, 'GP')].steps[-1][1].kernel_}")
    alloys = apply_thermo(alloys, cfg)
    alloys = screen_flags(alloys, cfg)
    alloys.round(5).to_csv(outdir / "table_2_phase2_all81.csv", index=False)

    cols = ["formula", "x_Rb", "y_Br", "t", "tau", "S_over_kB", "TS_mix_eV_atom", "Ehull_0K", "Ehull_T",
            "Ehull_GP_sigma", "Gap_corr", "Gap_GP_sigma", "Ehull_spread", "Gap_spread", "Ehull_Anchored",
            "Gap_Anchored", "in_domain", "verdict"]
    cands = alloys[alloys.pass_nominal].sort_values("Ehull_T")[cols]
    cands.round(4).to_csv(outdir / "table_2_phase2_candidates.csv", index=False)

    n_nom, n_str, n_g1 = int(alloys.pass_nominal.sum()), int(alloys.pass_strict.sum()), int(alloys.pass_gap_1sigma.sum())
    say(f"\nPHASE 2 SCREEN at T = {cfg.T:.0f} K   (Omega_A={cfg.omega_a}, Omega_X={cfg.omega_x} eV/fu; "
        f"bowing_A={cfg.bow_a}, bowing_X={cfg.bow_x} eV)   <- assumed values, see sensitivity below")
    say(f"  criterion 1: Ehull,T + sigma_GP <= {cfg.ehull_max} eV/atom;  criterion 2: {cfg.gap_lo} <= Eg <= {cfg.gap_hi} eV")
    say(f"  nominal pass (no sigma): {n_nom}/81   strict pass (+1 sigma): {n_str}/81   "
        f"also gap +/-1 sigma inside window: {n_g1}/81")
    say(f"  median GP sigma: Ehull {alloys.Ehull_GP_sigma.median():.3f} eV/atom, gap {alloys.Gap_GP_sigma.median():.2f} eV "
        f"(gap window is only {cfg.gap_hi - cfg.gap_lo:.2f} eV wide)")
    say(f"  max entropy credit  T*S_mix/5 = {alloys.TS_mix_eV_atom.max():.4f} eV/atom;  mean {alloys.TS_mix_eV_atom.mean():.4f}")
    n_stab_fail = int((alloys.Ehull_T > cfg.ehull_max).sum())
    n_gap_fail = int((~alloys.Gap_corr.between(cfg.gap_lo, cfg.gap_hi)).sum())
    say(f"  alloys failing the stability test alone: {n_stab_fail}/81;  failing the gap window alone: {n_gap_fail}/81")
    say(f"  predicted Ehull range over all 81 alloys: {alloys.Ehull_GP.min():.3f} to {alloys.Ehull_GP.max():.3f} eV/atom "
        f"(training Ehull range {train.Ehull.min():.3f} to {train.Ehull.max():.3f})")
    if n_stab_fail == 0:
        say("  -> NO alloy fails on stability: the models were trained only on compounds with a cubic entry and cannot")
        say("     predict instability, so the screen is decided by the band-gap model (the weaker one).")
    xs_ = np.sort(alloys.x_Rb.unique())
    g_lo_ = alloys[alloys.x_Rb == xs_[0]].sort_values("y_Br").Gap_GP.to_numpy()
    g_hi_ = alloys[alloys.x_Rb == xs_[-1]].sort_values("y_Br").Gap_GP.to_numpy()
    say(f"  GP-implied gap increase from x = {xs_[0]} to x = {xs_[-1]} Rb (at fixed Br): {np.min(g_hi_ - g_lo_):+.2f} to "
        f"{np.max(g_hi_ - g_lo_):+.2f} eV.  The tabulated Cs->Rb pairs below do not agree on this.")
    say(f"  applicability domain (descriptor-space check only; it cannot see the missing Rb-Sn data): "
        f"{int(alloys.in_domain.sum())}/81 alloys lie within the training set's own nearest-neighbour spacing")
    say("  No Rb-Sn compound has a cubic entry -> every alloy with x > 0 extrapolates the Rb effect")
    say("  from Pb/Ge compounds.  Phase-2 numbers are screening predictions.")
    shown = cands.head(14).copy()
    shown["in_domain"] = shown["in_domain"].map({True: "yes", False: "NO"})
    say("\n  Top nominal candidates (sorted by Ehull,T):" if len(cands) else "\n  No alloy passes the nominal screen.")
    if len(cands):
        say(shown[["formula", "t", "Ehull_T", "Ehull_GP_sigma", "Gap_corr", "Gap_GP_sigma",
                   "Ehull_spread", "Gap_spread", "in_domain", "verdict"]].round(3).to_string(index=False))

    say("\nML-FREE CROSS-CHECK using only the tabulated entries")
    say(f"  CsSnI3  (Ehull, gap) = {base_info['CsSnI3']}    CsSnBr3 = {base_info['CsSnBr3']}")
    for p in base_info["cs_rb_pairs"]:
        say(f"  Cs->Rb pair {p['pair']}: dEhull = {p['dEhull']:+.3f} eV/atom, dGap = {p['dGap']:+.3f} eV")
    say(f"  mean dEhull(Cs->Rb) = {base_info['dEhull_CsToRb_mean']:+.3f} eV/atom (used); the gap shifts disagree, so no Rb gap term is applied.")
    d_e = (alloys.Ehull_GP - alloys.Ehull_Anchored).abs()
    d_g = (alloys.Gap_GP - alloys.Gap_Anchored).abs()
    say(f"  mean |GP - anchored baseline| over 81 alloys: Ehull {d_e.mean():.3f} eV/atom, gap {d_g.mean():.3f} eV")
    base = alloys.assign(Ehull_GP=alloys.Ehull_Anchored, Gap_GP=alloys.Gap_Anchored)
    base = screen_flags(apply_thermo(base, cfg), cfg)
    say(f"  alloys that pass using the anchored baseline instead of ML (nominal): {int(base.pass_nominal.sum())}/81")
    edge = build_alloys(xs=[0.0], ys=FRACTIONS)
    Xe = edge[FEATURES].to_numpy(float)
    line_g = (1 - edge.y_Br) * base_info["CsSnI3"][1] + edge.y_Br * base_info["CsSnBr3"][1]
    for mname in ("Ridge", "Boost"):
        pg = fitted[("Gap", mname)].predict(Xe)
        say(f"  gap along the Cs-only I-Br edge, {mname} vs linear interpolation: mean |diff| = {np.mean(np.abs(pg - line_g)):.3f} eV")
    pg, _ = gp_predict(fitted[("Gap", "GP")], Xe)
    say(f"  gap along the Cs-only I-Br edge, GP vs linear interpolation: mean |diff| = {np.mean(np.abs(pg - line_g)):.3f} eV")

    say("\nSENSITIVITY: number of STRICT passes vs. the two assumed physics parameters (Omega_X: I/Br mixing")
    say("enthalpy per formula unit; bowing: optical gap bowing).  These are assumptions, not measurements.")
    sens = {}
    for om in (0.0, 0.3, 0.6, 1.0, 2.0):
        sens[f"Omega_X={om:.1f} eV/fu"] = {}
        for bw in (0.0, 0.2, 0.4):
            c2 = Cfg(**{**cfg.__dict__, "omega_x": om, "bow_x": bw})
            n = int(screen_flags(apply_thermo(alloys, c2), c2).pass_strict.sum())   # apply_thermo overwrites the thermo columns
            sens[f"Omega_X={om:.1f} eV/fu"][f"bowing_X={bw:.1f} eV"] = n
    sens = pd.DataFrame(sens).T
    sens.to_csv(outdir / "sensitivity.csv")
    say(sens.to_string())

    say(f"[{time.time() - _T0:6.1f}s] Making figures")
    make_figures(tab, preds, train, alloys, fitted, cfg, outdir)
    write_verification_queue(alloys, outdir / "dft_verification_queue.csv")

    imp_lines = []
    for t_ in ("Ehull", "Gap"):
        imp = pd.Series(np.asarray(fitted[(t_, "Boost")].feature_importances_, dtype=float), index=FEATURES)
        imp_lines.append(f"  {t_}: " + ", ".join(f"{k} {v:.0%}" for k, v in imp.sort_values(ascending=False).head(3).items()))
    say(f"\nTOP-3 {BOOSTER_NAME} FEATURE IMPORTANCES (n = {len(train)}, unstable - indicative only)")
    for ln in imp_lines:
        say(ln)

    say("\nLIMITATIONS IMPLIED BY THIS RUN")
    say(f"  1. Selection bias: the models only see the {len(train)} compounds that have a cubic entry; the largest")
    say(f"     Ehull in the data is {train.Ehull.max():.3f} eV/atom, so no unstable compound is ever seen.")
    say(f"  2. n = {len(train)}: compare the cross-validation errors above with the mean baseline.")
    say("  3. Rb-Sn end members are missing, so the Rb dependence is borrowed from Pb/Ge compounds.")
    say("  4. Entropy credit assumes ideal mixing; Omega and bowing are assumed (default 0).")
    say("  5. MP gaps are DFT gaps, not experimental; comparison with the Shockley-Queisser window is a screen.")
    say(f"  6. Booster used: {BOOSTER_NAME}" + ("" if BOOSTER_NAME == "XGBoost" else " (xgboost not installed; sklearn GradientBoostingRegressor with the same settings)."))

    results = {
        "meta": {"date": dt.date.today().isoformat(), "seed": SEED, "T_K": cfg.T, "ehull_max": cfg.ehull_max,
                 "gap_window_eV": [cfg.gap_lo, cfg.gap_hi], "omega_a": cfg.omega_a, "omega_x": cfg.omega_x,
                 "bow_a": cfg.bow_a, "bow_x": cfg.bow_x, "booster": BOOSTER_NAME, "n_total": len(tab),
                 "n_train": len(train), "n_alloys": len(alloys), "n_nominal_pass": n_nom, "n_strict_pass": n_str,
                 "python": sys.version.split()[0], "platform": platform.platform(),
                 "data_sources": sorted(set(tab.source))},
        "phase1_stats": st, "halide_trend_flags": flags, "anchored_baseline": base_info,
        # to_json turns NaN into null, which plain json.dumps would not
        "cv_metrics": json.loads(cv_table.round(4).to_json(orient="records")),
        "sensitivity_strict_pass": sens.to_dict(),
        "candidates": json.loads(cands.round(4).to_json(orient="records")),
    }
    (outdir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    say(f"[{time.time() - _T0:6.1f}s] Done.")
    say(f"\nFiles written to {outdir}/ : run_report.txt, results.json, mp_verified_data.csv, table_A1_all_48.csv,")
    say("  table_1_cubic_entries.csv, table_2_phase2_all81.csv, table_2_phase2_candidates.csv, cv_metrics.csv,")
    say("  sensitivity.csv, dft_verification_queue.csv, fig1..fig6 (.png)")
    (outdir / "run_report.txt").write_text("\n".join(REPORT))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Two-phase screen of ABX3 halide perovskites")
    ap.add_argument("--data", default=None, help="CSV with columns formula, ehull_eV_atom, gap_eV [, material_id, source]")
    ap.add_argument("--out", default="results")
    ap.add_argument("--evaluate", default=None, help="score a filled-in dft_verification_queue.csv and exit")
    ap.add_argument("--ehull-max", type=float, default=Cfg.ehull_max)
    ap.add_argument("--gap-lo", type=float, default=Cfg.gap_lo)
    ap.add_argument("--gap-hi", type=float, default=Cfg.gap_hi)
    ap.add_argument("--temp", type=float, default=Cfg.T)
    ap.add_argument("--omega-a", type=float, default=0.0, help="assumed A-site mixing enthalpy, eV per formula unit")
    ap.add_argument("--omega-x", type=float, default=0.0, help="assumed X-site mixing enthalpy, eV per formula unit")
    ap.add_argument("--bow-a", type=float, default=0.0, help="assumed A-site gap bowing, eV")
    ap.add_argument("--bow-x", type=float, default=0.0, help="assumed X-site gap bowing, eV")
    args, _ = ap.parse_known_args()     # tolerate extra argv inside Colab / Jupyter
    cfg = Cfg(ehull_max=args.ehull_max, gap_lo=args.gap_lo, gap_hi=args.gap_hi, T=args.temp,
              omega_a=args.omega_a, omega_x=args.omega_x, bow_a=args.bow_a, bow_x=args.bow_x,
              out=args.out, data=args.data)
    if args.evaluate:
        print(json.dumps(evaluate_verification(args.evaluate, cfg), indent=2))
    else:
        main(cfg)
