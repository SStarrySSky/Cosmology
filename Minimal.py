
from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, minimize
from scipy.linalg import cho_factor, cho_solve, eigh



OUT_DIR = "outputs_minimal_growth_fit"
FIG_DIR = os.path.join(OUT_DIR, "figures")
SUM_DIR = os.path.join(OUT_DIR, "summaries")


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(SUM_DIR, exist_ok=True)




def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(name: str, payload):
    path = os.path.join(SUM_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    print(f"[saved] {path}")


def regularize_covariance(cov: np.ndarray):
    cov = np.array(cov, dtype=float, copy=True)
    cov = 0.5 * (cov + cov.T)

    evals = eigh(cov, eigvals_only=True, check_finite=False)
    min_eig = float(np.min(evals))
    scale = max(1.0, float(np.max(np.diag(cov))))

    jitter = 0.0
    if min_eig <= 0.0:
        jitter = (-min_eig + 1.0e-10 * scale)
        cov = cov + jitter * np.eye(cov.shape[0])

    cho = cho_factor(cov, lower=True, check_finite=False)
    return cov, cho, min_eig, jitter


def solve_linear_system_cov(cho_cov, vec):
    return cho_solve(cho_cov, vec, check_finite=False)


def model_selection_stats(chi2_min: float, k: int, ndata: int):
    aic = chi2_min + 2.0 * k
    bic = chi2_min + k * math.log(ndata)
    return float(aic), float(bic)


def transition_redshift(z: np.ndarray, q: np.ndarray):
    idx = np.where(np.sign(q[:-1]) * np.sign(q[1:]) < 0)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    z0, z1 = z[i], z[i + 1]
    q0, q1 = q[i], q[i + 1]
    if abs(q1 - q0) < 1.0e-14:
        return None
    return float(z0 - q0 * (z1 - z0) / (q1 - q0))


def detect_boundary_hits(params: dict, bounds: dict, frac_tol: float = 0.01):
    hits = {}
    for key, val in params.items():
        if key not in bounds:
            continue
        lo, hi = bounds[key]
        rng = hi - lo
        if rng <= 0:
            continue
        near_lo = (val - lo) / rng < frac_tol
        near_hi = (hi - val) / rng < frac_tol
        if near_lo or near_hi:
            hits[key] = {"value": float(val), "lower": float(lo), "upper": float(hi)}
    return hits



@dataclass
class SNData:
    z: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    cho_cov: tuple
    min_eig_before: float
    jitter_added: float


@dataclass
class BAOData:
    z: np.ndarray
    obs: np.ndarray
    sigma: np.ndarray
    kind: np.ndarray


@dataclass
class GrowthData:
    z: np.ndarray
    fs8: np.ndarray
    sigma: np.ndarray
    cov: np.ndarray
    cho_cov: tuple
    min_eig_before: float
    jitter_added: float


@dataclass
class FixedParams:
    gamma: float = 2.0
    emax_prior: float = 8.0
    qmax_prior: float = 5.0
    z_ini_growth: float = 50.0


@dataclass
class MinimalNestedParams:
    Omega_m0: float
    lambda1: float
    Omega_c: float
    eta: float

    @property
    def Omega_L0(self) -> float:
        return 1.0 - self.Omega_m0

    @property
    def s0(self) -> float:
        return 1.0 / (1.0 + self.Omega_m0 / self.Omega_c)




def load_pantheon(data_dir: str) -> SNData:
    mu_path = os.path.join(data_dir, "pantheon_plus_mu.csv")
    cov_path = os.path.join(data_dir, "pantheon_plus_cov.npy")

    df = pd.read_csv(mu_path)
    if "z" not in df.columns or "mu" not in df.columns:
        raise ValueError("pantheon_plus_mu.csv must contain columns: z, mu")

    z = df["z"].to_numpy(dtype=float)
    mu = df["mu"].to_numpy(dtype=float)
    cov = np.load(cov_path)

    if cov.shape[0] != cov.shape[1] or cov.shape[0] != len(z):
        raise ValueError("Pantheon covariance shape does not match SN data length.")

    cov_reg, cho_cov, min_eig_before, jitter_added = regularize_covariance(cov)

    return SNData(
        z=z,
        mu=mu,
        cov=cov_reg,
        cho_cov=cho_cov,
        min_eig_before=min_eig_before,
        jitter_added=jitter_added,
    )


def load_bao(data_dir: str) -> BAOData:
    path = os.path.join(data_dir, "bao.csv")
    df = pd.read_csv(path)

    required = {"z", "obs", "sigma", "kind"}
    if not required.issubset(df.columns):
        raise ValueError("bao.csv must contain columns: z, obs, sigma, kind")

    kind = df["kind"].astype(str).to_numpy()
    allowed = {"DM_over_rd", "DH_over_rd", "DV_over_rd"}
    if not set(kind).issubset(allowed):
        raise ValueError("BAO kind must be one of: DM_over_rd, DH_over_rd, DV_over_rd")

    return BAOData(
        z=df["z"].to_numpy(dtype=float),
        obs=df["obs"].to_numpy(dtype=float),
        sigma=df["sigma"].to_numpy(dtype=float),
        kind=kind,
    )


def load_growth(data_dir: str) -> GrowthData:
    csv_path = os.path.join(data_dir, "growth_fsigma8.csv")
    cov_path = os.path.join(data_dir, "growth_cov.npy")

    df = pd.read_csv(csv_path)
    required = {"z", "fs8", "sigma"}
    if not required.issubset(df.columns):
        raise ValueError("growth_fsigma8.csv must contain columns: z, fs8, sigma")

    z = df["z"].to_numpy(dtype=float)
    fs8 = df["fs8"].to_numpy(dtype=float)
    sigma = df["sigma"].to_numpy(dtype=float)

    if os.path.exists(cov_path):
        cov = np.load(cov_path)
        if cov.shape[0] != cov.shape[1] or cov.shape[0] != len(z):
            raise ValueError("growth_cov.npy shape does not match growth data length.")
    else:
        cov = np.diag(sigma ** 2)

    cov_reg, cho_cov, min_eig_before, jitter_added = regularize_covariance(cov)

    return GrowthData(
        z=z,
        fs8=fs8,
        sigma=sigma,
        cov=cov_reg,
        cho_cov=cho_cov,
        min_eig_before=min_eig_before,
        jitter_added=jitter_added,
    )




def solve_lcdm_background(Omega_m0: float, zmax: float, n_eval: int = 3000):
    z = np.linspace(0.0, zmax, n_eval)
    E = np.sqrt(Omega_m0 * (1.0 + z) ** 3 + (1.0 - Omega_m0))
    chi = cumulative_trapezoid(1.0 / E, z, initial=0.0)

    A_tot = -0.5 * Omega_m0 * (1.0 + z) ** 3 + (1.0 - Omega_m0)
    q = -A_tot / (E ** 2)

    # Also build x-grid views for growth
    x = -np.log(1.0 + z)
    x_rev = x[::-1]
    E_rev = E[::-1]
    q_rev = q[::-1]

    E_interp = interp1d(z, E, kind="cubic", bounds_error=False, fill_value="extrapolate")
    chi_interp = interp1d(z, chi, kind="cubic", bounds_error=False, fill_value="extrapolate")
    q_interp = interp1d(z, q, kind="cubic", bounds_error=False, fill_value="extrapolate")

    E_x_interp = interp1d(x_rev, E_rev, kind="cubic", bounds_error=False, fill_value="extrapolate")
    q_x_interp = interp1d(x_rev, q_rev, kind="cubic", bounds_error=False, fill_value="extrapolate")

    return {
        "z_grid": z,
        "x_grid": x,
        "E_grid": E,
        "chi_grid": chi,
        "q_grid": q,
        "E_interp": E_interp,
        "chi_interp": chi_interp,
        "q_interp": q_interp,
        "E_x_interp": E_x_interp,
        "q_x_interp": q_x_interp,
    }




def s_eq(omega_m_bg: np.ndarray | float, pars: MinimalNestedParams):
    omega_m_bg = np.asarray(omega_m_bg)
    return 1.0 / (1.0 + omega_m_bg / pars.Omega_c)


def s_eq_prime(omega_m_bg: np.ndarray | float, pars: MinimalNestedParams):
    omega_m_bg = np.asarray(omega_m_bg)
    return -(1.0 / pars.Omega_c) / (1.0 + omega_m_bg / pars.Omega_c) ** 2


def R_rel(delta_s: np.ndarray | float, pars: MinimalNestedParams, fixed: FixedParams):
    return pars.lambda1 * np.tanh(fixed.gamma * np.asarray(delta_s))


def rhs_minimal_nested_in_x(x: float, y: np.ndarray, pars: MinimalNestedParams, fixed: FixedParams):
    h = float(y[0])
    h_safe = max(h, 1.0e-10)

    omega_m_bg = pars.Omega_m0 * math.exp(-3.0 * x) / (h_safe ** 2)
    s = float(s_eq(omega_m_bg, pars))
    delta_s = s - pars.s0
    R = float(R_rel(delta_s, pars, fixed))

    dhdx = (-h_safe**2 - 0.5 * pars.Omega_m0 * math.exp(-3.0 * x) + pars.Omega_L0 + R) / h_safe
    return [dhdx]


def event_h_zero_minimal_nested(x: float, y: np.ndarray, pars: MinimalNestedParams, fixed: FixedParams):
    return y[0] - 1.0e-8


event_h_zero_minimal_nested.terminal = True
event_h_zero_minimal_nested.direction = -1


def solve_minimal_nested_background(pars: MinimalNestedParams, fixed: FixedParams, zmax: float, n_eval: int = 3000):
    x_min = -math.log(1.0 + zmax)
    y0 = np.array([1.0], dtype=float)

    x_eval = np.linspace(0.0, x_min, n_eval)
    sol = solve_ivp(
        fun=lambda x, y: rhs_minimal_nested_in_x(x, y, pars, fixed),
        t_span=(0.0, x_min),
        y0=y0,
        t_eval=x_eval,
        events=lambda x, y: event_h_zero_minimal_nested(x, y, pars, fixed),
        rtol=1.0e-8,
        atol=1.0e-10,
        max_step=0.01,
    )

    if not sol.success:
        raise RuntimeError(f"Minimal nested backward integration failed: {sol.message}")

    x = sol.t
    h = sol.y[0]

    z = np.exp(-x) - 1.0
    E = h
    chi = cumulative_trapezoid(1.0 / E, z, initial=0.0)

    omega_m_bg = pars.Omega_m0 * np.exp(-3.0 * x) / (E ** 2)
    s = s_eq(omega_m_bg, pars)
    delta_s = s - pars.s0
    R = R_rel(delta_s, pars, fixed)

    A_tot = -0.5 * pars.Omega_m0 * np.exp(-3.0 * x) + pars.Omega_L0 + R
    q = -A_tot / (E ** 2)

    # growth interpolants need ascending x
    x_rev = x[::-1]
    E_rev = E[::-1]
    q_rev = q[::-1]
    s_rev = s[::-1]
    om_rev = omega_m_bg[::-1]

    E_interp = interp1d(z, E, kind="cubic", bounds_error=False, fill_value="extrapolate")
    chi_interp = interp1d(z, chi, kind="cubic", bounds_error=False, fill_value="extrapolate")
    q_interp = interp1d(z, q, kind="cubic", bounds_error=False, fill_value="extrapolate")
    s_interp = interp1d(z, s, kind="cubic", bounds_error=False, fill_value="extrapolate")

    E_x_interp = interp1d(x_rev, E_rev, kind="cubic", bounds_error=False, fill_value="extrapolate")
    q_x_interp = interp1d(x_rev, q_rev, kind="cubic", bounds_error=False, fill_value="extrapolate")
    s_x_interp = interp1d(x_rev, s_rev, kind="cubic", bounds_error=False, fill_value="extrapolate")
    om_x_interp = interp1d(x_rev, om_rev, kind="cubic", bounds_error=False, fill_value="extrapolate")

    return {
        "z_grid": z,
        "x_grid": x,
        "E_grid": E,
        "chi_grid": chi,
        "q_grid": q,
        "s_grid": s,
        "omega_m_grid": omega_m_bg,
        "E_interp": E_interp,
        "chi_interp": chi_interp,
        "q_interp": q_interp,
        "s_interp": s_interp,
        "E_x_interp": E_x_interp,
        "q_x_interp": q_x_interp,
        "s_x_interp": s_x_interp,
        "omega_m_x_interp": om_x_interp,
    }



def mu_geom_from_interp(z, chi_interp):
    z = np.asarray(z, dtype=float)
    chi = chi_interp(z)
    dL = (1.0 + z) * chi
    dL = np.maximum(dL, 1.0e-12)
    return 5.0 * np.log10(dL)


def bao_theory(z, kind, E_interp, chi_interp, eta):
    E = float(E_interp(z))
    chi = float(chi_interp(z))

    if kind == "DM_over_rd":
        return chi / eta
    elif kind == "DH_over_rd":
        return 1.0 / (E * eta)
    elif kind == "DV_over_rd":
        return (z * chi**2 / E) ** (1.0 / 3.0) / eta
    else:
        raise ValueError(f"Unknown BAO kind: {kind}")


def profile_linear_amplitude(y_obs: np.ndarray, template: np.ndarray, cho_cov):
    Cinv_t = solve_linear_system_cov(cho_cov, template)
    Cinv_y = solve_linear_system_cov(cho_cov, y_obs)

    a = float(template @ Cinv_t)
    b = float(template @ Cinv_y)

    amp_best = b / a
    resid = y_obs - amp_best * template
    Cinv_resid = solve_linear_system_cov(cho_cov, resid)
    chi2_best = float(resid @ Cinv_resid)

    return max(chi2_best, 0.0), float(amp_best), resid


def profile_sn_chi2(mu_obs: np.ndarray, mu_geom: np.ndarray, cho_cov):
    ones = np.ones_like(mu_obs)
    d0 = mu_obs - mu_geom
    return profile_linear_amplitude(d0, ones, cho_cov)


def chi2_bao(data: BAOData, E_interp, chi_interp, eta):
    vals = np.array([bao_theory(z, k, E_interp, chi_interp, eta) for z, k in zip(data.z, data.kind)])
    pulls = (data.obs - vals) / data.sigma
    return max(float(np.sum(pulls**2)), 0.0), vals




def solve_growth_lcdm(bg, Omega_m0: float, growth: GrowthData, fixed: FixedParams, z_ini: float | None = None):
    if z_ini is None:
        z_ini = fixed.z_ini_growth

    x_ini = -math.log(1.0 + z_ini)

    def rhs(x, y):
        delta, ddelta = y
        q = float(bg["q_x_interp"](x))
        E = float(bg["E_x_interp"](x))
        source = 1.5 * Omega_m0 * math.exp(-3.0 * x) / (E ** 2)
        friction = 1.0 - q
        return [ddelta, -friction * ddelta + source * delta]

    y0 = [math.exp(x_ini), math.exp(x_ini)]  # delta ~ a, delta' = delta wrt ln a
    x_eval = np.linspace(x_ini, 0.0, 2500)

    sol = solve_ivp(rhs, (x_ini, 0.0), y0, t_eval=x_eval, rtol=1.0e-8, atol=1.0e-10)
    if not sol.success:
        raise RuntimeError(f"LCDM growth solve failed: {sol.message}")

    x = sol.t
    delta = sol.y[0]
    ddelta = sol.y[1]

    D = delta / delta[-1]
    f = ddelta / delta
    z = np.exp(-x) - 1.0

    D_interp = interp1d(z[::-1], D[::-1], kind="cubic", bounds_error=False, fill_value="extrapolate")
    f_interp = interp1d(z[::-1], f[::-1], kind="cubic", bounds_error=False, fill_value="extrapolate")
    template = f_interp(growth.z) * D_interp(growth.z)

    return {
        "z_grid": z[::-1],
        "D_grid": D[::-1],
        "f_grid": f[::-1],
        "mu_grid": np.ones_like(z[::-1]),
        "template_fs8": template,
        "D_interp": D_interp,
        "f_interp": f_interp,
    }


def solve_growth_minimal_nested(bg, pars: MinimalNestedParams, growth: GrowthData, fixed: FixedParams, z_ini: float | None = None):
    if z_ini is None:
        z_ini = fixed.z_ini_growth

    x_ini = -math.log(1.0 + z_ini)

    def mu_of_x(x):
        E = float(bg["E_x_interp"](x))
        s = float(bg["s_x_interp"](x))
        om = float(bg["omega_m_x_interp"](x))
        delta_s = s - pars.s0
        sp = float(s_eq_prime(om, pars))
        return 1.0 - 2.0 * pars.lambda1 * fixed.gamma * (1.0 / math.cosh(fixed.gamma * delta_s)) ** 2 * sp / (E ** 2)

    def rhs(x, y):
        delta, ddelta = y
        q = float(bg["q_x_interp"](x))
        E = float(bg["E_x_interp"](x))
        mu = mu_of_x(x)
        source = 1.5 * pars.Omega_m0 * math.exp(-3.0 * x) / (E ** 2) * mu
        friction = 1.0 - q
        return [ddelta, -friction * ddelta + source * delta]

    y0 = [math.exp(x_ini), math.exp(x_ini)]
    x_eval = np.linspace(x_ini, 0.0, 2500)

    sol = solve_ivp(rhs, (x_ini, 0.0), y0, t_eval=x_eval, rtol=1.0e-8, atol=1.0e-10)
    if not sol.success:
        raise RuntimeError(f"Minimal nested growth solve failed: {sol.message}")

    x = sol.t
    delta = sol.y[0]
    ddelta = sol.y[1]

    D = delta / delta[-1]
    f = ddelta / delta
    z = np.exp(-x) - 1.0
    mu_vals = np.array([mu_of_x(xx) for xx in x])

    z_plot = z[::-1]
    D_plot = D[::-1]
    f_plot = f[::-1]
    mu_plot = mu_vals[::-1]

    D_interp = interp1d(z_plot, D_plot, kind="cubic", bounds_error=False, fill_value="extrapolate")
    f_interp = interp1d(z_plot, f_plot, kind="cubic", bounds_error=False, fill_value="extrapolate")
    mu_interp = interp1d(z_plot, mu_plot, kind="cubic", bounds_error=False, fill_value="extrapolate")

    template = f_interp(growth.z) * D_interp(growth.z)

    return {
        "z_grid": z_plot,
        "D_grid": D_plot,
        "f_grid": f_plot,
        "mu_grid": mu_plot,
        "template_fs8": template,
        "D_interp": D_interp,
        "f_interp": f_interp,
        "mu_interp": mu_interp,
    }

def objective_lcdm(theta, sn: SNData, bao: BAOData, growth: GrowthData, fixed: FixedParams, zmax_model: float):
    try:
        Omega_m0 = float(theta[0])
        eta = float(theta[1])

        if not (0.15 < Omega_m0 < 0.60):
            return 1.0e12
        if not (0.02 < eta < 0.05):
            return 1.0e12

        bg = solve_lcdm_background(Omega_m0, zmax=zmax_model, n_eval=1800)
        mu_geom = mu_geom_from_interp(sn.z, bg["chi_interp"])
        c2_sn, _, _ = profile_sn_chi2(sn.mu, mu_geom, sn.cho_cov)
        c2_bao, _ = chi2_bao(bao, bg["E_interp"], bg["chi_interp"], eta)

        gr = solve_growth_lcdm(bg, Omega_m0, growth, fixed)
        c2_gr, _, _ = profile_linear_amplitude(growth.fs8, gr["template_fs8"], growth.cho_cov)

        total = c2_sn + c2_bao + c2_gr
        if not np.isfinite(total):
            return 1.0e12
        return float(total)

    except Exception:
        return 1.0e12


def unpack_minimal_theta(theta) -> MinimalNestedParams:
    return MinimalNestedParams(
        Omega_m0=float(theta[0]),
        lambda1=float(theta[1]),
        Omega_c=float(theta[2]),
        eta=float(theta[3]),
    )


def objective_minimal_nested(theta, sn: SNData, bao: BAOData, growth: GrowthData, fixed: FixedParams, zmax_model: float):
    try:
        pars = unpack_minimal_theta(theta)

        if not (0.15 < pars.Omega_m0 < 0.50):
            return 1.0e12
        if not (0.00 <= pars.lambda1 < 0.20):
            return 1.0e12
        if not (0.02 < pars.Omega_c < 2.00):
            return 1.0e12
        if not (0.02 < pars.eta < 0.05):
            return 1.0e12

        bg = solve_minimal_nested_background(pars, fixed, zmax=zmax_model, n_eval=1800)

        if np.any(bg["E_grid"] <= 0.0):
            return 1.0e12
        if float(np.max(bg["E_grid"])) > fixed.emax_prior:
            return 1.0e12
        if float(np.max(np.abs(bg["q_grid"]))) > fixed.qmax_prior:
            return 1.0e12

        mu_geom = mu_geom_from_interp(sn.z, bg["chi_interp"])
        c2_sn, _, _ = profile_sn_chi2(sn.mu, mu_geom, sn.cho_cov)
        c2_bao, _ = chi2_bao(bao, bg["E_interp"], bg["chi_interp"], pars.eta)

        gr = solve_growth_minimal_nested(bg, pars, growth, fixed)
        c2_gr, _, _ = profile_linear_amplitude(growth.fs8, gr["template_fs8"], growth.cho_cov)

        total = c2_sn + c2_bao + c2_gr
        if not np.isfinite(total):
            return 1.0e12
        return float(total)

    except Exception:
        return 1.0e12



def fit_lcdm(sn, bao, growth, fixed, zmax_model):
    bounds = [
        (0.15, 0.60),  # Omega_m0
        (0.02, 0.05),  # eta
    ]

    result_de = differential_evolution(
        func=lambda th: objective_lcdm(th, sn, bao, growth, fixed, zmax_model),
        bounds=bounds,
        maxiter=35,
        popsize=12,
        polish=False,
        disp=True,
        seed=1234,
    )
    result_local = minimize(
        fun=lambda th: objective_lcdm(th, sn, bao, growth, fixed, zmax_model),
        x0=result_de.x,
        method="L-BFGS-B",
        bounds=bounds,
    )

    Omega_m0, eta = map(float, result_local.x)
    bg = solve_lcdm_background(Omega_m0, zmax=zmax_model, n_eval=3500)
    gr = solve_growth_lcdm(bg, Omega_m0, growth, fixed)

    mu_geom = mu_geom_from_interp(sn.z, bg["chi_interp"])
    c2_sn, Mcal_best, sn_resid = profile_sn_chi2(sn.mu, mu_geom, sn.cho_cov)
    c2_bao, bao_model = chi2_bao(bao, bg["E_interp"], bg["chi_interp"], eta)
    c2_gr, sigma8_best, gr_resid = profile_linear_amplitude(growth.fs8, gr["template_fs8"], growth.cho_cov)

    zt = transition_redshift(bg["z_grid"], bg["q_grid"])

    return {
        "params": {
            "Omega_m0": Omega_m0,
            "eta": eta,
            "Mcal": Mcal_best,
            "sigma8_0": sigma8_best,
        },
        "chi2_sn": float(c2_sn),
        "chi2_bao": float(c2_bao),
        "chi2_growth": float(c2_gr),
        "chi2_total": float(result_local.fun),
        "q0": float(bg["q_grid"][0]),
        "z_t": zt,
        "bg": bg,
        "gr": gr,
        "sn_residuals": sn_resid,
        "bao_model": bao_model,
        "growth_model": sigma8_best * gr["template_fs8"],
        "growth_residuals": gr_resid,
    }


def fit_minimal_nested(sn, bao, growth, fixed, zmax_model):
    bounds = [
        (0.15, 0.50),  # Omega_m0
        (0.00, 0.20),  # lambda1
        (0.02, 2.00),  # Omega_c
        (0.02, 0.05),  # eta
    ]

    result_de = differential_evolution(
        func=lambda th: objective_minimal_nested(th, sn, bao, growth, fixed, zmax_model),
        bounds=bounds,
        maxiter=40,
        popsize=14,
        polish=False,
        disp=True,
        seed=4321,
    )
    result_local = minimize(
        fun=lambda th: objective_minimal_nested(th, sn, bao, growth, fixed, zmax_model),
        x0=result_de.x,
        method="L-BFGS-B",
        bounds=bounds,
    )

    pars = unpack_minimal_theta(result_local.x)
    bg = solve_minimal_nested_background(pars, fixed, zmax=zmax_model, n_eval=3500)
    gr = solve_growth_minimal_nested(bg, pars, growth, fixed)

    mu_geom = mu_geom_from_interp(sn.z, bg["chi_interp"])
    c2_sn, Mcal_best, sn_resid = profile_sn_chi2(sn.mu, mu_geom, sn.cho_cov)
    c2_bao, bao_model = chi2_bao(bao, bg["E_interp"], bg["chi_interp"], pars.eta)
    c2_gr, sigma8_best, gr_resid = profile_linear_amplitude(growth.fs8, gr["template_fs8"], growth.cho_cov)

    zt = transition_redshift(bg["z_grid"], bg["q_grid"])

    return {
        "params": {
            **asdict(pars),
            "Omega_L0": float(pars.Omega_L0),
            "s0": float(pars.s0),
            "Mcal": Mcal_best,
            "sigma8_0": sigma8_best,
        },
        "fixed": asdict(fixed),
        "chi2_sn": float(c2_sn),
        "chi2_bao": float(c2_bao),
        "chi2_growth": float(c2_gr),
        "chi2_total": float(result_local.fun),
        "q0": float(bg["q_grid"][0]),
        "z_t": zt,
        "bg": bg,
        "gr": gr,
        "sn_residuals": sn_resid,
        "bao_model": bao_model,
        "growth_model": sigma8_best * gr["template_fs8"],
        "growth_residuals": gr_resid,
    }




def save_or_show(path: str, show: bool):
    plt.tight_layout()
    if path.endswith('.eps'):
        plt.savefig(path, format='eps', bbox_inches="tight")
    else:
        plt.savefig(path, bbox_inches="tight")
    print(f"[saved] {path}")
    if show:
        plt.show()
    plt.close()


def plot_background_compare(lcdm_fit, nested_fit, zmax_plot=3.0, show=False):
    z_plot = np.linspace(0.0, zmax_plot, 1200)

    E_l = lcdm_fit["bg"]["E_interp"](z_plot)
    q_l = lcdm_fit["bg"]["q_interp"](z_plot)

    E_n = nested_fit["bg"]["E_interp"](z_plot)
    q_n = nested_fit["bg"]["q_interp"](z_plot)
    s_n = nested_fit["bg"]["s_interp"](z_plot)

    plt.figure(figsize=(7, 5))
    plt.plot(z_plot, E_l, label=r"flat $\Lambda$CDM")
    plt.plot(z_plot, E_n, label="minimal nested")
    plt.xlabel("z")
    plt.ylabel(r"$H(z)/H_0$")
    plt.title("Background expansion comparison")
    plt.grid(True)
    plt.legend()
    save_or_show(os.path.join(FIG_DIR, "fig_compare_background_expansion.eps"), show)

    plt.figure(figsize=(7, 5))
    plt.plot(z_plot, q_l, label=r"flat $\Lambda$CDM")
    plt.plot(z_plot, q_n, label="minimal nested")
    plt.axhline(0.0, color="k", linestyle="--")
    plt.xlabel("z")
    plt.ylabel(r"$q(z)$")
    plt.title("Deceleration parameter comparison")
    plt.grid(True)
    plt.legend()
    save_or_show(os.path.join(FIG_DIR, "fig_compare_qz.eps"), show)

    plt.figure(figsize=(7, 5))
    plt.plot(z_plot, s_n, label=r"$s_H(z)$")
    plt.xlabel("z")
    plt.ylabel(r"$s_H$")
    plt.title("Minimal nested activation history")
    plt.grid(True)
    plt.legend()
    save_or_show(os.path.join(FIG_DIR, "fig_minimal_nested_activation_history.eps"), show)


def plot_growth_compare(growth: GrowthData, lcdm_fit, nested_fit, show=False):
    z_plot = np.linspace(0.0, max(float(np.max(growth.z)) * 1.1, 1.5), 800)

    fs8_l = lcdm_fit["params"]["sigma8_0"] * lcdm_fit["gr"]["f_interp"](z_plot) * lcdm_fit["gr"]["D_interp"](z_plot)
    fs8_n = nested_fit["params"]["sigma8_0"] * nested_fit["gr"]["f_interp"](z_plot) * nested_fit["gr"]["D_interp"](z_plot)

    plt.figure(figsize=(7, 5))
    plt.errorbar(growth.z, growth.fs8, yerr=growth.sigma, fmt="o", ms=4, label="growth data")
    plt.plot(z_plot, fs8_l, label=r"flat $\Lambda$CDM")
    plt.plot(z_plot, fs8_n, label="minimal nested")
    plt.xlabel("z")
    plt.ylabel(r"$f\sigma_8(z)$")
    plt.title(r"Growth-rate comparison")
    plt.grid(True)
    plt.legend()
    save_or_show(os.path.join(FIG_DIR, "fig_compare_fsigma8.eps"), show)

    plt.figure(figsize=(7, 5))
    plt.plot(nested_fit["gr"]["z_grid"], nested_fit["gr"]["mu_grid"], label=r"$\mu(z)$")
    plt.axhline(1.0, color="k", linestyle="--")
    plt.xlabel("z")
    plt.ylabel(r"$\mu(z)$")
    plt.title("Effective clustering modifier")
    plt.grid(True)
    plt.legend()
    save_or_show(os.path.join(FIG_DIR, "fig_minimal_nested_mu.eps"), show)


def plot_sn_residuals(sn: SNData, lcdm_fit, nested_fit, show=False):
    idx = np.argsort(sn.z)
    z = sn.z[idx]
    res_l = lcdm_fit["sn_residuals"][idx]
    res_n = nested_fit["sn_residuals"][idx]

    plt.figure(figsize=(7, 5))
    plt.plot(z, res_l, ".", ms=3, label=r"flat $\Lambda$CDM")
    plt.plot(z, res_n, ".", ms=3, label="minimal nested")
    plt.axhline(0.0, color="k", linestyle="--")
    plt.xlabel("z")
    plt.ylabel(r"$\mu_{\rm obs}-\mu_{\rm model}$")
    plt.title("SN residuals")
    plt.grid(True)
    plt.legend()
    save_or_show(os.path.join(FIG_DIR, "fig_SN_residuals.eps"), show)


def plot_bao_pulls(bao: BAOData, lcdm_fit, nested_fit, show=False):
    pull_l = (bao.obs - lcdm_fit["bao_model"]) / bao.sigma
    pull_n = (bao.obs - nested_fit["bao_model"]) / bao.sigma

    x = np.arange(len(bao.z))
    width = 0.38

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, pull_l, width=width, label=r"flat $\Lambda$CDM")
    plt.bar(x + width/2, pull_n, width=width, label="minimal nested")
    plt.axhline(0.0, color="k", linestyle="--")
    plt.axhline(1.0, color="gray", linestyle=":")
    plt.axhline(-1.0, color="gray", linestyle=":")
    labels = [f"{k}@{z:.2f}" for z, k in zip(bao.z, bao.kind)]
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("pull")
    plt.title("BAO residual pulls")
    plt.legend()
    plt.grid(True, axis="y")
    save_or_show(os.path.join(FIG_DIR, "fig_BAO_pulls.eps"), show)




def self_test():
    print("=== Running self-test ===")

    cov = np.eye(3)
    cov_reg, cho_cov, min_eig, jitter = regularize_covariance(cov)
    mu_obs = np.array([1.0, 2.0, 3.0])
    mu_geom = np.array([0.5, 2.0, 4.0])

    c2_sn, Mcal_best, delta_best = profile_sn_chi2(mu_obs, mu_geom, cho_cov)
    assert c2_sn >= 0.0
    assert np.isfinite(Mcal_best)
    print("[ok] SN profile works")

    template = np.array([0.4, 0.5, 0.6])
    y = 0.8 * template
    c2_gr, sigma8_best, resid = profile_linear_amplitude(y, template, cho_cov)
    assert abs(sigma8_best - 0.8) < 1.0e-10
    assert c2_gr < 1.0e-12
    print("[ok] Growth amplitude profiling works")

    fixed = FixedParams()
    pars = MinimalNestedParams(Omega_m0=0.30, lambda1=0.0, Omega_c=0.5, eta=0.03)
    bg_l = solve_lcdm_background(pars.Omega_m0, zmax=2.5, n_eval=1200)
    bg_n = solve_minimal_nested_background(pars, fixed, zmax=2.5, n_eval=1200)
    err = np.max(np.abs(bg_l["E_grid"] - bg_n["E_grid"]))
    assert err < 1.0e-5, f"LCDM limit mismatch too large: {err}"
    print("[ok] Minimal nested exact LCDM limit verified")

    print("=== Self-test passed ===")




def parse_args():
    parser = argparse.ArgumentParser(description="Minimal nested background+growth fitter")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    ensure_dirs()
    args = parse_args()

    if args.self_test:
        self_test()
        return

    sn = load_pantheon(args.data_dir)
    bao = load_bao(args.data_dir)
    growth = load_growth(args.data_dir)

    zmax_model = 1.05 * max(float(np.max(sn.z)), float(np.max(bao.z)), float(np.max(growth.z)), 2.5)
    ndata = len(sn.z) + len(bao.z) + len(growth.z)

    fixed = FixedParams()

    print("=== SN covariance diagnostics ===")
    print(f"  min eigenvalue before regularization = {sn.min_eig_before:.6e}")
    print(f"  jitter added = {sn.jitter_added:.6e}")

    print("\n=== Growth covariance diagnostics ===")
    print(f"  min eigenvalue before regularization = {growth.min_eig_before:.6e}")
    print(f"  jitter added = {growth.jitter_added:.6e}")

    print("\n=== Fitting flat LCDM baseline ===")
    lcdm_fit = fit_lcdm(sn, bao, growth, fixed, zmax_model=zmax_model)

    print("\n=== Fitting minimal nested model ===")
    nested_fit = fit_minimal_nested(sn, bao, growth, fixed, zmax_model=zmax_model)


    k_lcdm = 4
    k_nested = 6

    lcdm_aic, lcdm_bic = model_selection_stats(lcdm_fit["chi2_total"], k_lcdm, ndata)
    nested_aic, nested_bic = model_selection_stats(nested_fit["chi2_total"], k_nested, ndata)

    lcdm_bounds = {
        "Omega_m0": (0.15, 0.60),
        "eta": (0.02, 0.05),
    }

    nested_bounds = {
        "Omega_m0": (0.15, 0.50),
        "lambda1": (0.00, 0.20),
        "Omega_c": (0.02, 2.00),
        "eta": (0.02, 0.05),
    }

    summary = {
        "ndata_total": int(ndata),
        "sn_covariance_diagnostics": {
            "min_eigenvalue_before_regularization": sn.min_eig_before,
            "jitter_added": sn.jitter_added,
        },
        "growth_covariance_diagnostics": {
            "min_eigenvalue_before_regularization": growth.min_eig_before,
            "jitter_added": growth.jitter_added,
        },
        "fixed": asdict(fixed),
        "lcdm": {
            **lcdm_fit["params"],
            "chi2_sn": lcdm_fit["chi2_sn"],
            "chi2_bao": lcdm_fit["chi2_bao"],
            "chi2_growth": lcdm_fit["chi2_growth"],
            "chi2_total": lcdm_fit["chi2_total"],
            "AIC": lcdm_aic,
            "BIC": lcdm_bic,
            "q0": lcdm_fit["q0"],
            "z_t": lcdm_fit["z_t"],
            "boundary_hits": detect_boundary_hits(lcdm_fit["params"], lcdm_bounds),
        },
        "minimal_nested": {
            **nested_fit["params"],
            "chi2_sn": nested_fit["chi2_sn"],
            "chi2_bao": nested_fit["chi2_bao"],
            "chi2_growth": nested_fit["chi2_growth"],
            "chi2_total": nested_fit["chi2_total"],
            "AIC": nested_aic,
            "BIC": nested_bic,
            "q0": nested_fit["q0"],
            "z_t": nested_fit["z_t"],
            "boundary_hits": detect_boundary_hits(nested_fit["params"], nested_bounds),
        },
        "deltas": {
            "Delta_chi2": nested_fit["chi2_total"] - lcdm_fit["chi2_total"],
            "Delta_AIC": nested_aic - lcdm_aic,
            "Delta_BIC": nested_bic - lcdm_bic,
        }
    }

    save_json("minimal_growth_fit_summary.json", summary)

    plot_background_compare(lcdm_fit, nested_fit, zmax_plot=min(3.0, zmax_model), show=args.show)
    plot_growth_compare(growth, lcdm_fit, nested_fit, show=args.show)
    plot_sn_residuals(sn, lcdm_fit, nested_fit, show=args.show)
    plot_bao_pulls(bao, lcdm_fit, nested_fit, show=args.show)

    print("\n=== Summary ===")
    print(json.dumps(to_jsonable(summary), indent=2))
    print(f"\nOutputs written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()