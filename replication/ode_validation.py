"""
ode_validation.py
=================
Step 4 of replicating Ohi et al. (2020): validate the grid environment against
the standard SEIR ODE model.

Produces three comparisons from the paper:
  * Table 1   -- reproduction rate (mean/median) vs population density,
                 and the key finding that R0 is roughly density-INDEPENDENT.
  * Figure 7  -- active-case waves grow with density.
  * Figure 8  -- ODE vs grid S/I/R/D curves at densities 0.01, 0.02, 0.03.
Plus a herd-immunity / final-attack-rate check.

The pseudocode (Supplement 1) does not cover the ODE or validation; the ODE is
Eq. 5 from the paper. See the accompanying notes for the judgment calls
(R0 = beta/gamma, ODE-vs-grid timing, omitting the exposed state, the
herd-immunity inconsistency, and R0 aggregation).

Depends on: pandemic_env.py (Step 1), state_representation.py (Step 3), numpy.
matplotlib optional (plots only).
"""

from __future__ import annotations
import numpy as np

from state_representation import StateBuilder
from pandemic_env import PandemicEnv

# --- paper's reported R0 (mean, mean_std, median, median_std) from Table 1 ---
PAPER_R0 = {
    0.01: (2.87, 0.19, 2.84, 0.11),
    0.02: (3.20, 0.30, 2.84, 0.02),
    0.03: (3.40, 0.23, 2.94, 0.08),
    0.04: (3.40, 0.18, 2.76, 0.11),
    0.10: (3.30, 0.40, 2.73, 0.05),
    0.20: (3.40, 0.12, 2.90, 0.05),
}

# Number of grid replicates per density. The paper uses 10; lower it for a
# faster check (the density-independence finding is visible at 3-5 runs).
N_RUNS = 10


# ===========================================================================
# ODE SEIR model (Eq. 5)
# ===========================================================================
def seir_ode(N=10_000, I0=70, beta=0.12, alpha=1.0, gamma=1.0 / 27,
             mu=0.009, days=310, dt=0.1):
    """Integrate the paper's SEIR ODE with dependency-free RK4.

    Eq. 5 parameters: R0 = beta/gamma = 3.24 (the paper's definition; the
    textbook form would be beta/(gamma+mu) ~ 2.6). Returns daily-resolution
    arrays as PERCENT of N. Initial state matches the grid default: 0.7%
    infectious (S=9930, I=70).
    """
    def deriv(y):
        S, E, I, R, D = y
        dS = -beta * I * S / N
        dE = beta * S * I / N - alpha * E
        dI = alpha * E - (gamma + mu) * I
        dR = gamma * I
        dD = mu * I
        return np.array([dS, dE, dI, dR, dD])

    y = np.array([N - I0, 0.0, float(I0), 0.0, 0.0])
    steps = int(round(days / dt))
    per_day = int(round(1.0 / dt))
    out = {k: [] for k in ("t", "S", "E", "I", "R", "D")}
    for step in range(steps + 1):
        if step % per_day == 0:
            out["t"].append(step * dt)
            for k, v in zip("SEIRD", y):
                out[k].append(v)
        if step < steps:
            k1 = deriv(y)
            k2 = deriv(y + 0.5 * dt * k1)
            k3 = deriv(y + 0.5 * dt * k2)
            k4 = deriv(y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return {k: (np.array(v) if k == "t" else 100.0 * np.array(v) / N)
            for k, v in out.items()}


# ===========================================================================
# Grid replicate runner
# ===========================================================================
def run_grid_once(density, seed, max_days=400, init_infectious=70):
    """Run one level-0 grid episode.

    Returns (reports, r0_seed_cohort, r0_early_cohort): the tau-free count of
    distinct secondary infections per infectious agent, averaged over (a) the
    seed cohort and (b) agents that turned infectious while S/N >= 0.95. This
    replaces the old population estimator's mean/median, which was pinned near 1
    by a closed-population conservation identity (see notes)."""
    env = PandemicEnv(density=density, init_infectious=init_infectious, seed=seed)
    builder = StateBuilder(env)            # daily dim-5 R0 observable still populated
    reports = []
    while env.day < max_days:
        rep = env.run_day(restriction_level=0)
        builder.build(rep)
        reports.append(rep)
        if rep["exposed"] == 0 and rep["infectious"] == 0:
            break

    fin = env.completed
    is_seed = (env.sfrac0 > 0.99) & fin
    early = (env.sfrac0 >= 0.95) & fin
    r0_seed = float(env.n_sec[is_seed].mean()) if is_seed.any() else 0.0
    r0_early = float(env.n_sec[early].mean()) if early.any() else 0.0
    return reports, r0_seed, r0_early


def reports_to_pct(reports, N):
    """Extract S/I/R/D as percent-of-N daily arrays from grid reports."""
    days = np.array([r["day"] for r in reports])
    S = np.array([100.0 * r["susceptible"] / N for r in reports])
    I = np.array([r["active_pct"] for r in reports])
    R = np.array([r["cured_cum_pct"] for r in reports])
    D = np.array([r["death_cum_pct"] for r in reports])
    return days, S, I, R, D


# ===========================================================================
# Comparisons
# ===========================================================================
def table1_comparison(densities=(0.01, 0.02, 0.03, 0.10), n_runs=N_RUNS, base_seed=0):
    """Cohort reproduction rate per density vs the paper."""
    print(f"\n=== Table 1: cohort reproduction rate vs density ({n_runs} runs each) ===")
    print(f"{'density':>8} | {'seed cohort R0':>18} | {'S>=0.95 cohort R0':>20} | "
          f"{'paper (mean/med)':>18}")
    print("-" * 74)
    results = {}
    for d in densities:
        seeds_r0, early_r0 = [], []
        for k in range(n_runs):
            _, rs, re = run_grid_once(d, seed=base_seed + 1000 * int(d * 1000) + k)
            seeds_r0.append(rs); early_r0.append(re)
        sm, ss = np.mean(seeds_r0), np.std(seeds_r0)
        em, es = np.mean(early_r0), np.std(early_r0)
        results[d] = (sm, ss, em, es)
        pap = PAPER_R0.get(round(d, 2))
        pstr = f"{pap[0]:.2f}/{pap[2]:.2f}" if pap else "n/a"
        print(f"{d:>8.2f} | {sm:>9.2f} +/- {ss:<5.2f} | {em:>11.2f} +/- {es:<5.2f} | {pstr:>18}")
    print("\nNote: cohort R0 RISES with density (3x3 neighbourhoods never saturate "
          "at these occupancies), so Table 1's density-independence does NOT "
          "reproduce under a faithful R0. The seed cohort at density 0.01 (~3.3) "
          "is the cleanest match to the ODE's R0 = beta/gamma = 3.24.")
    return results


def herd_immunity_check(density=0.02, seed=0):
    """Compare final attack rate (ODE vs grid) and the textbook threshold."""
    R0 = 0.12 / (1.0 / 27)            # = 3.24 (paper's beta/gamma)
    threshold = 100.0 * (1 - 1 / R0)  # textbook 1 - 1/R0 = ~69%
    ode = seir_ode()
    ode_attack = 100.0 - ode["S"][-1]
    reports, _, _ = run_grid_once(density, seed=seed)
    _, S, _, _, _ = reports_to_pct(reports, 10_000)
    grid_attack = 100.0 - S[-1]
    print(f"\n=== Herd immunity / final attack rate (density {density}) ===")
    print(f"textbook 1 - 1/R0 (R0={R0:.2f}) : {threshold:5.1f}%   "
          f"(paper writes 88.14%, which is inconsistent with this formula)")
    print(f"ODE final attack rate           : {ode_attack:5.1f}%")
    print(f"grid final attack rate          : {grid_attack:5.1f}%   "
          f"(paper reports ~88-92% infected)")


# ===========================================================================
# Plots (optional)
# ===========================================================================
def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return None


def plot_fig7(densities=(0.01, 0.02, 0.03, 0.04, 0.10, 0.20),
              max_days=400, path="figure7_active_waves.png"):
    plt = _mpl()
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    for d in densities:
        reports, _, _ = run_grid_once(d, seed=0, max_days=max_days)
        days, _, I, _, _ = reports_to_pct(reports, 10_000)
        plt.plot(days, I, label=f"Density={d}")
    plt.xlabel("Days"); plt.ylabel("Active cases (%)")
    plt.title("Fig. 7 replica: active-case waves by density")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=120)
    print(f"saved plot -> {path}")


def plot_fig8(densities=(0.01, 0.02, 0.03), path="figure8_ode_vs_grid.png"):
    plt = _mpl()
    if plt is None:
        return
    ode = seir_ode()
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("ODE", None)] + [(f"Grid density {d}", d) for d in densities]
    for axi, (title, d) in zip(ax.flat, panels):
        if d is None:
            axi.plot(ode["t"], ode["S"], label="Susceptible")
            axi.plot(ode["t"], ode["I"], label="Infectious")
            axi.plot(ode["t"], ode["R"], label="Recovered")
            axi.plot(ode["t"], ode["D"], label="Dead")
        else:
            reports, _, _ = run_grid_once(d, seed=0)
            days, S, I, R, D = reports_to_pct(reports, 10_000)
            axi.plot(days, S, label="Susceptible")
            axi.plot(days, I, label="Infectious")
            axi.plot(days, R, label="Recovered")
            axi.plot(days, D, label="Dead")
        axi.set_title(title); axi.set_xlabel("Days"); axi.set_ylabel("Population (%)")
        axi.set_xlim(0, 300); axi.legend(fontsize=8)
    fig.suptitle("Fig. 8 replica: ODE vs grid (exposed omitted)")
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"saved plot -> {path}")


if __name__ == "__main__":
    print(f"ODE: R0 = beta/gamma = {0.12 / (1/27):.2f}, "
          f"implied mortality mu/(gamma+mu) = {0.009 / (1/27 + 0.009):.1%}")
    print(f"(Table 1 uses N_RUNS={N_RUNS} replicates per density -- "
          f"this can take several minutes; lower N_RUNS for a quick check.)")

    table1_comparison(densities=(0.01, 0.02, 0.03))
    herd_immunity_check(density=0.02)
    plot_fig7()
    plot_fig8()