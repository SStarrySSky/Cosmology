
from __future__ import annotations

import os
import json
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, minimize


from Minimal import (
    ensure_dirs,
    save_json,
    to_jsonable,
    load_pantheon,
    load_bao,
    load_growth,
    FixedParams,
    MinimalNestedParams,
    fit_lcdm,
    solve_minimal_nested_background,
    solve_growth_minimal_nested,
    mu_geom_from_interp,
    profile_sn_chi2,
    chi2_bao,
    profile_linear_amplitude,
    model_selection_stats,
)

OUT_DIR = "outputs_epsilon_g_profile"
FIG_DIR = os.path.join(OUT_DIR, "figures")
SUM_DIR = os.path.join(OUT_DIR, "summaries")


def ensure_local_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(SUM_DIR, exist_ok=True)


def save_local_json(name: str, payload):
    path = os.path.join(SUM_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    print(f"[saved] {path}")


def epsilon_g_from_params(Omega_m0: float, lambda1: float, Omega_c: float, gamma: float) -> float:
    # mu(0)-1 = 2 lambda1 gamma Omega_c / (Omega_c + Omega_m0)^2
    return 2.0 * lambda1 * gamma * Omega_c / (Omega_c + Omega_m0) ** 2


def lambda1_from_epsilon_g(Omega_m0: float, epsilon_g: float, Omega_c: float, gamma: float) -> float:
    # lambda1 = epsilon_g (Omega_c + Omega_m0)^2 / (2 gamma Omega_c)
    return epsilon_g * (Omega_c + Omega_m0) ** 2 / (2.0 * gamma * Omega_c)


def evaluate_nested_from_epsg(
    eps_g: float,
    theta_rest,
    sn,
    bao,
    growth,
    fixed: FixedParams,
    zmax_model: float,
):
    """
    theta_rest = [Omega_m0, Omega_c, eta]
    epsilon_g fixed, map to lambda1.
    """
    Omega_m0 = float(theta_rest[0])
    Omega_c = float(theta_rest[1])
    eta = float(theta_rest[2])

    if not (0.15 < Omega_m0 < 0.50):
        return None
    if not (0.02 < Omega_c < 2.00):
        return None
    if not (0.02 < eta < 0.05):
        return None
    if eps_g < 0.0:
        return None

    lambda1 = lambda1_from_epsilon_g(Omega_m0, eps_g, Omega_c, fixed.gamma)

    # 保持和你之前相同的先验边界
    if not (0.0 <= lambda1 < 0.20):
        return None

    pars = MinimalNestedParams(
        Omega_m0=Omega_m0,
        lambda1=lambda1,
        Omega_c=Omega_c,
        eta=eta,
    )

    try:
        bg = solve_minimal_nested_background(pars, fixed, zmax=zmax_model, n_eval=2500)

        if np.any(bg["E_grid"] <= 0.0):
            return None
        if float(np.max(bg["E_grid"])) > fixed.emax_prior:
            return None
        if float(np.max(np.abs(bg["q_grid"]))) > fixed.qmax_prior:
            return None

        mu_geom = mu_geom_from_interp(sn.z, bg["chi_interp"])
        c2_sn, Mcal_best, _ = profile_sn_chi2(sn.mu, mu_geom, sn.cho_cov)
        c2_bao, _ = chi2_bao(bao, bg["E_interp"], bg["chi_interp"], eta)

        gr = solve_growth_minimal_nested(bg, pars, growth, fixed)
        c2_gr, sigma8_best, _ = profile_linear_amplitude(growth.fs8, gr["template_fs8"], growth.cho_cov)

        total = float(c2_sn + c2_bao + c2_gr)

        return {
            "pars": pars,
            "epsilon_g": float(eps_g),
            "lambda1": float(lambda1),
            "Mcal": float(Mcal_best),
            "sigma8_0": float(sigma8_best),
            "chi2_sn": float(c2_sn),
            "chi2_bao": float(c2_bao),
            "chi2_growth": float(c2_gr),
            "chi2_total": total,
            "bg": bg,
            "gr": gr,
        }
    except Exception:
        return None


def objective_profile_rest(
    theta_rest,
    eps_g: float,
    sn,
    bao,
    growth,
    fixed: FixedParams,
    zmax_model: float,
):
    out = evaluate_nested_from_epsg(eps_g, theta_rest, sn, bao, growth, fixed, zmax_model)
    if out is None:
        return 1.0e12
    return out["chi2_total"]


def profile_scan_epsg(
    eps_grid,
    sn,
    bao,
    growth,
    fixed: FixedParams,
    zmax_model: float,
):
    results = []

    prev_best = np.array([0.37, 0.10, 0.032], dtype=float)  # [Omega_m0, Omega_c, eta]

    bounds = [
        (0.15, 0.50),  # Omega_m0
        (0.02, 2.00),  # Omega_c
        (0.02, 0.05),  # eta
    ]

    for i, eps_g in enumerate(eps_grid):
        print(f"[profile] {i+1}/{len(eps_grid)}   epsilon_g = {eps_g:.6f}")

        # 先全局粗搜
        de = differential_evolution(
            func=lambda th: objective_profile_rest(th, eps_g, sn, bao, growth, fixed, zmax_model),
            bounds=bounds,
            maxiter=25,
            popsize=10,
            polish=False,
            disp=False,
            seed=1000 + i,
        )

        x0 = de.x
        if np.isfinite(objective_profile_rest(prev_best, eps_g, sn, bao, growth, fixed, zmax_model)):

            if objective_profile_rest(prev_best, eps_g, sn, bao, growth, fixed, zmax_model) < de.fun:
                x0 = prev_best.copy()

        local = minimize(
            fun=lambda th: objective_profile_rest(th, eps_g, sn, bao, growth, fixed, zmax_model),
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
        )

        out = evaluate_nested_from_epsg(eps_g, local.x, sn, bao, growth, fixed, zmax_model)
        if out is None:
            result = {
                "epsilon_g": float(eps_g),
                "success": False,
                "chi2_total": None,
            }
        else:
            prev_best = np.array([
                out["pars"].Omega_m0,
                out["pars"].Omega_c,
                out["pars"].eta,
            ])
            result = {
                "epsilon_g": float(eps_g),
                "success": True,
                "Omega_m0": float(out["pars"].Omega_m0),
                "Omega_c": float(out["pars"].Omega_c),
                "eta": float(out["pars"].eta),
                "lambda1": float(out["lambda1"]),
                "sigma8_0": float(out["sigma8_0"]),
                "Mcal": float(out["Mcal"]),
                "chi2_sn": float(out["chi2_sn"]),
                "chi2_bao": float(out["chi2_bao"]),
                "chi2_growth": float(out["chi2_growth"]),
                "chi2_total": float(out["chi2_total"]),
            }

        results.append(result)

    return results


def extract_profile_limits(profile_results):
    good = [r for r in profile_results if r["success"] and np.isfinite(r["chi2_total"])]
    if not good:
        return None

    eps = np.array([r["epsilon_g"] for r in good], dtype=float)
    chi2 = np.array([r["chi2_total"] for r in good], dtype=float)

    idx = np.argmin(chi2)
    chi2_min = float(chi2[idx])
    eps_best = float(eps[idx])
    dchi2 = chi2 - chi2_min


    def upper_limit(target):
        mask = eps >= eps_best
        eps_r = eps[mask]
        dchi2_r = dchi2[mask]
        above = np.where(dchi2_r >= target)[0]
        if len(above) == 0:
            return None
        j = int(above[0])
        if j == 0:
            return float(eps_r[0])
        x0, x1 = eps_r[j - 1], eps_r[j]
        y0, y1 = dchi2_r[j - 1], dchi2_r[j]
        if abs(y1 - y0) < 1e-12:
            return float(x1)
        return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))

    return {
        "epsilon_g_best": eps_best,
        "chi2_min": chi2_min,
        "upper_68": upper_limit(1.0),
        "upper_95": upper_limit(3.84),
        "eps_grid": eps.tolist(),
        "chi2_grid": chi2.tolist(),
        "dchi2_grid": dchi2.tolist(),
    }


def plot_profile(profile_summary, show=False):
    eps = np.array(profile_summary["eps_grid"], dtype=float)
    dchi2 = np.array(profile_summary["dchi2_grid"], dtype=float)

    plt.figure(figsize=(7, 5))
    plt.plot(eps, dchi2, lw=2)
    plt.axhline(1.0, color="gray", linestyle="--", label=r"$\Delta\chi^2=1$")
    plt.axhline(3.84, color="k", linestyle="--", label=r"$\Delta\chi^2=3.84$")
    if profile_summary["upper_68"] is not None:
        plt.axvline(profile_summary["upper_68"], color="gray", linestyle=":")
    if profile_summary["upper_95"] is not None:
        plt.axvline(profile_summary["upper_95"], color="k", linestyle=":")
    plt.xlabel(r"$\epsilon_g=\mu(0)-1$")
    plt.ylabel(r"$\Delta\chi^2$")
    plt.title(r"Profile likelihood for growth modification")
    plt.grid(True)
    plt.legend()
    path = os.path.join(FIG_DIR, "profile_epsilon_g.eps")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"[saved] {path}")
    if show:
        plt.show()
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Profile scan for epsilon_g = mu(0)-1")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--eps-max", type=float, default=0.12)
    parser.add_argument("--n-grid", type=int, default=21)
    return parser.parse_args()


def main():
    ensure_dirs()
    ensure_local_dirs()
    args = parse_args()

    sn = load_pantheon(args.data_dir)
    bao = load_bao(args.data_dir)
    growth = load_growth(args.data_dir)

    fixed = FixedParams()
    zmax_model = 1.05 * max(float(np.max(sn.z)), float(np.max(bao.z)), float(np.max(growth.z)), 2.5)
    ndata = len(sn.z) + len(bao.z) + len(growth.z)

    print("=== Fitting flat LCDM reference ===")
    lcdm_fit = fit_lcdm(sn, bao, growth, fixed, zmax_model)
    lcdm_aic, lcdm_bic = model_selection_stats(lcdm_fit["chi2_total"], 4, ndata)

    eps_grid = np.linspace(0.0, args.eps_max, args.n_grid)

    print("\n=== Running epsilon_g profile scan ===")
    profile_results = profile_scan_epsg(eps_grid, sn, bao, growth, fixed, zmax_model)
    save_local_json("profile_scan_raw.json", profile_results)

    profile_summary = extract_profile_limits(profile_results)
    if profile_summary is None:
        raise RuntimeError("No successful profile points found.")

    plot_profile(profile_summary, show=args.show)


    good = [r for r in profile_results if r["success"]]
    chi2_vals = np.array([r["chi2_total"] for r in good], dtype=float)
    best_idx = int(np.argmin(chi2_vals))
    best_profile = good[best_idx]

    nested_aic, nested_bic = model_selection_stats(best_profile["chi2_total"], 6, ndata)

    summary = {
        "ndata_total": int(ndata),
        "lcdm_reference": {
            "Omega_m0": lcdm_fit["params"]["Omega_m0"],
            "eta": lcdm_fit["params"]["eta"],
            "sigma8_0": lcdm_fit["params"]["sigma8_0"],
            "chi2_sn": lcdm_fit["chi2_sn"],
            "chi2_bao": lcdm_fit["chi2_bao"],
            "chi2_growth": lcdm_fit["chi2_growth"],
            "chi2_total": lcdm_fit["chi2_total"],
            "AIC": lcdm_aic,
            "BIC": lcdm_bic,
        },
        "best_profiled_nested": {
            **best_profile,
            "AIC": nested_aic,
            "BIC": nested_bic,
        },
        "epsilon_g_profile": {
            "epsilon_g_best": profile_summary["epsilon_g_best"],
            "upper_68": profile_summary["upper_68"],
            "upper_95": profile_summary["upper_95"],
        },
        "deltas_vs_lcdm": {
            "Delta_chi2": best_profile["chi2_total"] - lcdm_fit["chi2_total"],
            "Delta_AIC": nested_aic - lcdm_aic,
            "Delta_BIC": nested_bic - lcdm_bic,
        }
    }

    save_local_json("profile_summary.json", summary)

    print("\n=== Profile summary ===")
    print(json.dumps(to_jsonable(summary), indent=2))
    print(f"\nOutputs written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()