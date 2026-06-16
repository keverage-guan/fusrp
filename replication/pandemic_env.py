"""
pandemic_env.py
===============
Step 1 of replicating Ohi et al. (2020), "Exploring optimal control of epidemic
spread using reinforcement learning" (Sci. Rep. 10:22106).

This file implements the *virtual environment* only: a 2D-grid SEIR simulator
with random agent movement, neighbourhood-based infection, randomised
exposed/infectious durations, ~20% mortality, and a per-day economy signal.
It follows Algorithm 1 (Supplementary file 1) where the paper and the
pseudocode agree, and the paper where they disagree. The disagreements and the
choices made are documented inline below.

Later steps (NOT implemented here): the reproduction rate R0, the normalised
7-dimensional state vector, the reward function (Eq. 1), and the DDQN agent.
The per-day report produced here already exposes most of the raw quantities
those steps need.

Dependencies: numpy (required), matplotlib (optional, only for the demo plot).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from math import sqrt
import numpy as np

# ---- state codes -----------------------------------------------------------
SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED, DEAD = 0, 1, 2, 3, 4

# ---- movement-restriction multipliers (paper: level-0/1/2 = 100/75/25%) ----
# Applied to the number of daily movement steps. Default play uses level 0.
# NOTE: the economy<->restriction coupling (the M_t term in the reward, Eq. 1)
# is deliberately deferred to the reward step; here we report raw economy.
RESTRICTION_MULT = {0: 1.0, 1: 0.75, 2: 0.25}


@dataclass
class PandemicEnv:
    # --- population / grid ---
    population: int = 10_000          # N, total population
    density: float = 0.02             # used to derive grid side length L
    init_infectious: int = 70         # M, initial infectious (0.7%, matches Fig.15/ODE)
    daily_moves: int = 15             # Mt, random moves per person per day

    # --- disease timing (in DAYS) ---
    # Exposed transitions when (days_in_state - rand[0..exposed_rand]) >= exposed_days
    #   -> with exposed_days=1, exposed_rand=1 this yields 1-2 days (paper).
    exposed_days: int = 1
    exposed_rand: int = 1
    # Infectious transitions when (days_in_state - rand[0..infectious_rand]) >= infectious_days
    #   -> with infectious_days=21, infectious_rand=6 this yields 21-27 days (paper).
    infectious_days: int = 21
    infectious_rand: int = 6

    # --- outcomes / economy ---
    death_prob: float = 0.20          # ~20% mortality among the infectious (paper)
    econ_low: float = 0.8             # per-person daily economy draw in [0.8, 1]
    econ_high: float = 1.0

    seed: int | None = None

    # --- internal state (set in reset) ---
    L: int = field(init=False, default=0)
    rng: np.random.Generator = field(init=False, default=None)

    def __post_init__(self):
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self):
        """Initialise the grid and population. Returns the day-0 report."""
        self.rng = np.random.default_rng(self.seed)
        self.N = int(self.population)
        self.L = int(round(sqrt(self.N / self.density)))
        self.realized_density = self.N / (self.L * self.L)

        self.x = self.rng.integers(0, self.L, size=self.N)
        self.y = self.rng.integers(0, self.L, size=self.N)
        self.state = np.full(self.N, SUSCEPTIBLE, dtype=np.int8)
        self.counter = np.zeros(self.N, dtype=np.int32)  # days in current state

        # --- cohort-R0 bookkeeping (tau-free reproduction number) ---
        self.n_sec = np.zeros(self.N, dtype=np.int32)     # secondaries caused by each agent
        self.sfrac0 = np.full(self.N, np.nan)             # S/N when the agent turned infectious
        self.completed = np.zeros(self.N, dtype=bool)     # agent has left the infectious state

        inf_idx = self.rng.choice(self.N, size=self.init_infectious, replace=False)
        self.state[inf_idx] = INFECTIOUS
        self.sfrac0[inf_idx] = (self.N - self.init_infectious) / self.N  # seed cohort

        self.day = 0
        self.economy_cumulative = 0.0
        return self._report(restriction_level=0, new_exposed=0,
                            new_infectious=0, new_cured=0, new_deaths=0,
                            economy_today=0.0)

    # -------------------------------------------------------------- one day
    def run_day(self, restriction_level: int = 0) -> dict:
        """Advance the environment by one day under the given restriction level.

        Daily order (see disagreement notes #1 and #4 in the accompanying
        write-up):
          1. age existing E/I and process transitions (once per day),
          2. movement + neighbourhood infection (Mt steps),
          3. economy contribution (once per day).
        """
        self.day += 1

        # 1. aging + transitions (counter is in DAYS, not steps)
        new_infectious, new_cured, new_deaths = self._age_and_transition()

        # 2. movement + infection; restriction reduces the number of daily moves
        steps = max(1, int(round(self.daily_moves * RESTRICTION_MULT[restriction_level])))
        new_exposed = 0
        for _ in range(steps):
            self._move()
            new_exposed += self._infect()

        # 3. economy (per day, per living non-infectious person; scaled by the
        #    restriction multiplier so lockdown reduces economic output, Fig. 18)
        economy_today = self._economy(RESTRICTION_MULT[restriction_level])
        self.economy_cumulative += economy_today

        return self._report(restriction_level, new_exposed,
                            new_infectious, new_cured, new_deaths, economy_today)

    def simulate(self, restriction_level: int = 0, max_days: int = 2000,
                 progress: bool = False) -> list[dict]:
        """Run until the disease fully mitigates (no E and no I) or max_days.

        Set progress=True for a tqdm bar (falls back to silent if tqdm is not
        installed). The run length is unknown in advance, so the bar advances
        toward max_days and shows the current active-case count.
        """
        bar = None
        if progress:
            try:
                from tqdm import tqdm
                bar = tqdm(total=max_days, desc="simulating", unit="day")
            except ImportError:
                print("tqdm not installed (pip install tqdm); running without a bar.")

        reports = []
        while self.day < max_days:
            rep = self.run_day(restriction_level)
            reports.append(rep)
            if bar is not None:
                bar.update(1)
                bar.set_postfix(active=f"{rep['active_pct']:.2f}%", I=rep["infectious"])
            if rep["exposed"] == 0 and rep["infectious"] == 0:
                break
        if bar is not None:
            bar.close()
        return reports

    # ----------------------------------------------------------- mechanics
    def _move(self):
        """Every living agent takes one random step in {-1,0,+1}^2, clamped."""
        alive = self.state != DEAD
        n = int(alive.sum())
        if n == 0:
            return
        self.x[alive] = np.clip(self.x[alive] + self.rng.integers(-1, 2, size=n), 0, self.L - 1)
        self.y[alive] = np.clip(self.y[alive] + self.rng.integers(-1, 2, size=n), 0, self.L - 1)

    def _infect(self) -> int:
        """Susceptibles in a 3x3 (Chebyshev<=1) neighbourhood of an infectious
        agent become EXPOSED. NB: this follows the PAPER (S->E). Pseudocode
        line 36 literally writes the contact into I, which would erase the
        latent period -- that is a pseudocode bug, not the intended model.

        Cohort bookkeeping: each newly exposed agent is attributed to one
        infectious neighbour and that infector's n_sec is incremented. This is
        the basis of the faithful, tau-free reproduction number (see notes)."""
        inf = self.state == INFECTIOUS
        sus = self.state == SUSCEPTIBLE
        if not inf.any() or not sus.any():
            return 0

        L = self.L
        # one infector id per occupied cell (last-writer-wins; fine on a sparse grid)
        owner = np.full((L, L), -1, dtype=np.int32)
        inf_idx = np.where(inf)[0]
        owner[self.x[inf_idx], self.y[inf_idx]] = inf_idx

        # propagate infector ids into the 3x3 neighbourhood (no edge wrap)
        pad = np.full((L + 2, L + 2), -1, dtype=np.int32)
        pad[1:-1, 1:-1] = owner
        big = np.full((L, L), -1, dtype=np.int32)
        for dx in (0, 1, 2):
            for dy in (0, 1, 2):
                src = pad[dx:dx + L, dy:dy + L]
                take = (big < 0) & (src >= 0)
                big[take] = src[take]

        sus_idx = np.where(sus)[0]
        infector = big[self.x[sus_idx], self.y[sus_idx]]
        hit = infector >= 0
        newly = sus_idx[hit]
        if newly.size:
            np.add.at(self.n_sec, infector[hit], 1)
            self.state[newly] = EXPOSED
            self.counter[newly] = 0
        return int(newly.size)

    def _age_and_transition(self):
        """Once-per-day aging and E->I, I->(R or D) transitions."""
        mask_E = self.state == EXPOSED
        mask_I = self.state == INFECTIOUS
        sfrac = float((self.state == SUSCEPTIBLE).sum()) / self.N  # for cohort tagging
        self.counter[mask_E] += 1
        self.counter[mask_I] += 1

        new_infectious = new_cured = new_deaths = 0

        # Exposed -> Infectious
        e_idx = np.where(mask_E)[0]
        if e_idx.size:
            r = self.rng.integers(0, self.exposed_rand + 1, size=e_idx.size)
            trans = (self.counter[e_idx] - r) >= self.exposed_days
            t_idx = e_idx[trans]
            self.state[t_idx] = INFECTIOUS
            self.counter[t_idx] = 0
            self.sfrac0[t_idx] = sfrac               # S/N at the moment this agent turns infectious
            new_infectious = int(t_idx.size)

        # Infectious -> Recovered / Dead
        i_idx = np.where(mask_I)[0]
        if i_idx.size:
            r = self.rng.integers(0, self.infectious_rand + 1, size=i_idx.size)
            done = (self.counter[i_idx] - r) >= self.infectious_days
            d_idx = i_idx[done]
            if d_idx.size:
                dead = self.rng.random(d_idx.size) < self.death_prob
                dead_idx = d_idx[dead]
                rec_idx = d_idx[~dead]
                self.state[dead_idx] = DEAD
                self.state[rec_idx] = RECOVERED
                self.counter[rec_idx] = 0
                self.completed[d_idx] = True          # infectious life finished -> n_sec is final
                new_deaths = int(dead_idx.size)
                new_cured = int(rec_idx.size)

        return new_infectious, new_cured, new_deaths

    def _economy(self, mult: float = 1.0) -> float:
        """Daily economy = sum over living, non-infectious agents of
        mult * U[econ_low, econ_high].

        `mult` is the movement-restriction multiplier (level-0/1/2 -> 1.0/0.75/
        0.25). The paper (Fig. 18) requires the economy to collapse under
        lockdown: contribution is tied to movement, so a restriction scales each
        person's contribution. At level-2 this drives the economy ratio to
        ~0.18, matching Fig. 18.
        """
        contrib = np.isin(self.state, (SUSCEPTIBLE, EXPOSED, RECOVERED))
        k = int(contrib.sum())
        if k == 0:
            return 0.0
        return float(mult * self.rng.uniform(self.econ_low, self.econ_high, size=k).sum())

    # -------------------------------------------------------------- reporting
    def _report(self, restriction_level, new_exposed, new_infectious,
                new_cured, new_deaths, economy_today) -> dict:
        s = int((self.state == SUSCEPTIBLE).sum())
        e = int((self.state == EXPOSED).sum())
        i = int((self.state == INFECTIOUS).sum())
        r = int((self.state == RECOVERED).sum())
        d = int((self.state == DEAD).sum())
        N = self.N
        return {
            "day": self.day,
            "susceptible": s, "exposed": e, "infectious": i,
            "recovered": r, "dead": d,
            # percentages use the INITIAL total population N (matches Eq. 1)
            "active_pct": 100.0 * i / N,             # active cases (%)
            "new_infected_pct": 100.0 * new_infectious / N,  # newly infected (%)
            "cured_cum_pct": 100.0 * r / N,          # cumulative cured (%)
            "death_cum_pct": 100.0 * d / N,          # cumulative deaths (%)
            "new_exposed": new_exposed,
            "new_cured": new_cured,
            "new_deaths": new_deaths,
            "economy_today": economy_today,
            "economy_cumulative": self.economy_cumulative,
            "restriction_level": restriction_level,
            # R0 and the normalised 7-dim state vector are later steps.
        }


# ============================================================================
# Demo / quick validation (matches the Fig. 16 "level-0, no lockdown" scenario)
# ============================================================================
def plot_simulation(reports, path="figure16_replica.png"):
    """4-panel plot in the spirit of Fig. 16 (R0 panel omitted, deferred)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    days = [r["day"] for r in reports]
    infected_cum = [r["cured_cum_pct"] + r["death_cum_pct"] + r["active_pct"] for r in reports]
    cured = [r["cured_cum_pct"] for r in reports]
    dead = [r["death_cum_pct"] for r in reports]
    active = [r["active_pct"] for r in reports]
    # normalised daily economy ratio relative to the full population
    econ_ratio = [r["economy_today"] / (0.9 * reports[0]["susceptible"] + 1e-9) for r in reports]

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(days, infected_cum, label="Infected")
    ax[0, 0].plot(days, cured, label="Cured")
    ax[0, 0].plot(days, dead, label="Death")
    ax[0, 0].set_title("Cumulative population (%)"); ax[0, 0].legend(); ax[0, 0].set_xlabel("Days")
    ax[1, 0].plot(days, active); ax[1, 0].set_title("Active cases (%)"); ax[1, 0].set_xlabel("Days")
    ax[0, 1].axis("off")
    ax[0, 1].text(0.05, 0.5, "Reproduction rate (R0)\nis a later step", fontsize=11)
    ax[1, 1].plot(days, econ_ratio); ax[1, 1].set_title("Economy ratio (rough)"); ax[1, 1].set_xlabel("Days")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved plot -> {path}")


if __name__ == "__main__":
    env = PandemicEnv(population=10_000, density=0.02, init_infectious=70, seed=0)
    print(f"grid: {env.L}x{env.L}  realized density: {env.realized_density:.5f}")

    reports = env.simulate(restriction_level=0, progress=True)
    last = reports[-1]
    peak_active = max(r["active_pct"] for r in reports)
    peak_day = max(reports, key=lambda r: r["active_pct"])["day"]

    print(f"days to mitigation : {last['day']}")
    print(f"peak active cases  : {peak_active:.2f}%  (day {peak_day})")
    print(f"final cured        : {last['cured_cum_pct']:.2f}%")
    print(f"final deaths       : {last['death_cum_pct']:.2f}%")
    print(f"final susceptible  : {100*last['susceptible']/env.N:.2f}% (never infected)")
    print(f"total economy      : {last['economy_cumulative']:.0f}")

    plot_simulation(reports)