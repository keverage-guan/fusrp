"""
state_representation.py
=======================
Step 3 of replicating Ohi et al. (2020): the 7-dimensional daily state vector
that the DRL agent observes.

The seven dimensions (paper "State generation" + Fig. 2):
    1. active cases (%)              -- from Step 1
    2. newly infected (%)            -- from Step 1  (E -> I per day)
    3. cumulative cured (%)          -- from Step 1
    4. cumulative deaths (%)         -- from Step 1
    5. reproduction rate R0          -- NEW (see disagreement #2)
    6. economy (ratio, 0-1)          -- NEW normalisation (see disagreement #4)
    7. current movement restriction  -- from Step 1  (0 / 1 / 2)

The pseudocode (Supplement 1) does not cover state generation, so this step is
reconstructed from the paper text and Figs. 16-21. See the accompanying notes
for the judgment calls (R0 estimator, economy<->restriction coupling, the
new-infected vs new-exposed distinction, and percentage denominators).

Depends on: pandemic_env.py (Step 1), numpy. matplotlib optional (demo plot).
"""

from __future__ import annotations
import numpy as np

from pandemic_env import PandemicEnv

# Canonical ordering of the 7-dim state vector.
STATE_FIELDS = [
    "active_pct", "new_infected_pct", "cured_cum_pct",
    "death_cum_pct", "r0", "economy_ratio", "restriction_level",
]


# ===========================================================================
# State builder
# ===========================================================================
class StateBuilder:
    """Turns a Step-1 daily report into the 7-dim state vector.

    Parameters
    ----------
    env : PandemicEnv
        Used only to read N, econ_high, and the infectious-duration timing.
    mean_infectious_duration : float | None
        tau in the R0 estimator. Defaults to infectious_days + infectious_rand/2
        (= 24 for the paper's 21-27 day window).
    r0_smooth : int
        Optional trailing-window average for R0 (1 = raw daily, as in the
        figures). The paper does not smooth; kept off by default.
    r0_scale : float
        Only used by normalize(): R0 is divided by this and clipped to 1.
    """

    def __init__(self, env, mean_infectious_duration=None, r0_smooth=1, r0_scale=10.0):
        self.N = env.N
        self.econ_high = env.econ_high
        if mean_infectious_duration is None:
            mean_infectious_duration = env.infectious_days + env.infectious_rand / 2.0
        self.tau = float(mean_infectious_duration)
        self.r0_smooth = max(1, int(r0_smooth))
        self.r0_scale = float(r0_scale)
        self._r0_hist: list[float] = []

    # -- dimension 5 ---------------------------------------------------------
    def reproduction_rate(self, report: dict) -> float:
        """Effective reproduction number (disagreement #2).

        R_t = tau * (people newly infected by the infectious population today)
                    / (current active cases)

        'newly infected by the infectious population' = new EXPOSURES (S->E)
        this day, because a contact turns a susceptible into an exposed person
        (disagreement #3). When active cases -> 0 the ratio is undefined; we
        report 0 (no infectious population means no reproduction). The estimator
        spikes at low prevalence, matching Figs. 16-21, and time-averages near
        ~3, matching Table 1.
        """
        active = report["infectious"]
        new_exposed = report["new_exposed"]
        r0 = (self.tau * new_exposed / active) if active > 0 else 0.0
        self._r0_hist.append(r0)
        if self.r0_smooth > 1:
            window = self._r0_hist[-self.r0_smooth:]
            return float(sum(window) / len(window))
        return float(r0)

    # -- full 7-dim state ----------------------------------------------------
    def build(self, report: dict) -> dict:
        """Return the 7-dim state as a dict (raw, figure-matching units)."""
        # dimension 6: economy ratio in [0, 1], relative to the theoretical
        # maximum of everyone alive and healthy contributing econ_high.
        economy_ratio = report["economy_today"] / (self.N * self.econ_high)
        return {
            "active_pct": report["active_pct"],            # 0-100
            "new_infected_pct": report["new_infected_pct"],# 0-100 (E->I)
            "cured_cum_pct": report["cured_cum_pct"],      # 0-100
            "death_cum_pct": report["death_cum_pct"],      # 0-100
            "r0": self.reproduction_rate(report),          # unbounded, ~0-20
            "economy_ratio": economy_ratio,                # 0-1
            "restriction_level": float(report["restriction_level"]),  # 0/1/2
        }

    def to_vector(self, state: dict) -> np.ndarray:
        """7-dim raw vector in canonical STATE_FIELDS order."""
        return np.array([state[f] for f in STATE_FIELDS], dtype=np.float32)

    def normalize(self, state: dict) -> np.ndarray:
        """Optional [0,1]-ish scaling for the network input (Step 6).

        The paper specifies no normalisation; these are documented, reasonable
        choices and should be revisited when the DDQN agent is built.
        """
        return np.array([
            state["active_pct"] / 100.0,
            state["new_infected_pct"] / 100.0,
            state["cured_cum_pct"] / 100.0,
            state["death_cum_pct"] / 100.0,
            min(state["r0"] / self.r0_scale, 1.0),   # clip R0 spikes
            state["economy_ratio"],                  # already 0-1
            state["restriction_level"] / 2.0,
        ], dtype=np.float32)


# ===========================================================================
# Convenience: assemble the M-day rolling history (mainly for Step 6).
# ===========================================================================
def rolling_window(vectors: list[np.ndarray], M: int, pad: str = "repeat") -> np.ndarray:
    """Most recent M state vectors as an (M, 7) array.

    Before M days have elapsed, pads the front by repeating the earliest state
    (pad='repeat') or with zeros (pad='zero'). This is the network input shape
    the agent will consume in Step 6 (memory length M, default 30).
    """
    dim = vectors[0].shape[0]
    recent = vectors[-M:]
    if len(recent) < M:
        if pad == "zero":
            head = [np.zeros(dim, dtype=np.float32)] * (M - len(recent))
        else:  # repeat earliest
            head = [recent[0]] * (M - len(recent))
        recent = head + list(recent)
    return np.stack(recent, axis=0)


# ===========================================================================
# Demo (level-0, no lockdown) -- compare R0 and economy with Fig. 16
# ===========================================================================
def _plot(days, r0, econ, active, path="figure16_state.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(days, active); ax[0].set_title("Active cases (%)"); ax[0].set_xlabel("Days")
    ax[1].plot(days, r0); ax[1].axhline(1.0, ls=":", color="gray")
    ax[1].set_title("Reproduction rate R0"); ax[1].set_xlabel("Days")
    ax[2].plot(days, econ); ax[2].set_ylim(0, 1); ax[2].set_title("Economy ratio"); ax[2].set_xlabel("Days")
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"saved plot -> {path}")


if __name__ == "__main__":
    env = PandemicEnv(population=10_000, density=0.02, init_infectious=70, seed=0)
    builder = StateBuilder(env)
    print(f"grid {env.L}x{env.L}, realized density {env.realized_density:.5f}, tau={builder.tau}")

    days, r0s, econs, actives, vectors = [], [], [], [], []
    while env.day < 2000:
        report = env.run_day(restriction_level=0)        # no lockdown baseline
        state = builder.build(report)
        vectors.append(builder.to_vector(state))
        days.append(report["day"]); r0s.append(state["r0"])
        econs.append(state["economy_ratio"]); actives.append(state["active_pct"])
        if report["exposed"] == 0 and report["infectious"] == 0:
            break

    # show a few representative states
    peak_i = int(np.argmax(actives))
    for label, idx in [("first full day", 0), ("active-case peak", peak_i), ("last day", -1)]:
        s = builder.build  # not re-run; just print the recorded vector
        v = vectors[idx]
        print(f"\n{label} (day {days[idx]}):")
        for f, val in zip(STATE_FIELDS, v):
            print(f"   {f:18s} = {val:8.3f}")

    print(f"\nmean R0 over run        : {np.mean(r0s):.2f}")
    print(f"max  R0 (low-prev spike): {np.max(r0s):.2f}")
    print(f"economy ratio (start/min/end): "
          f"{econs[0]:.2f} / {min(econs):.2f} / {econs[-1]:.2f}")

    # example 30-day network-input window (Step 6 shape)
    window = rolling_window(vectors, M=30)
    print(f"\nrolling 30-day window shape (Step 6 input): {window.shape}")

    _plot(days, r0s, econs, actives)