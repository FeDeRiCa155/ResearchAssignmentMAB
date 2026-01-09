# Bernoulli Bandit implementation

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Protocol


class BernoulliBandit:
    """
    Stochastic K-armed bandit with Bernoulli rewards.
    """

    def __init__(self, true_means):
        self.true_means = np.array(true_means, dtype=float) # list of success probabilities
        self.K = len(self.true_means) # number of arms

    def pull(self, arm): # sample from chosen arm
        p = self.true_means[arm]
        return np.random.rand() < p # 0 or 1 reward

    @property
    def optimal_mean(self):
        return float(np.max(self.true_means)) # highest success prob among arms

    @property
    def optimal_arm(self):
        return int(np.argmax(self.true_means)) # index of best arm

class Policy(Protocol): # rules algos follow
    def select_arm(self): # which arm to pull next round
        ...

    def update(self, arm, reward): # learn from obs. reward
        ...


@dataclass
class UCB1:
    n_arms: int
    c: float = 2.0  # exploration constant

    def __post_init__(self):
        self.counts = np.zeros(self.n_arms, dtype=int) # N_i(t)
        self.estimates = np.zeros(self.n_arms, dtype=float) # \hat{\mu}_i(t)
        self.t = 0

    def select_arm(self):
        self.t += 1

        # play each arm once initially
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # UCB index for each arm
        ucb_values = self.estimates + np.sqrt((self.c * np.log(self.t)) / self.counts)
        return int(np.argmax(ucb_values)) # arm with highest value

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.estimates[arm] += (reward - self.estimates[arm]) / n # incremental mean update


@dataclass
class ThompsonSamplingBernoulli:
    n_arms: int
    alpha0: float = 1.0 # uniform prior Beta(1,1)
    beta0: float = 1.0

    def __post_init__(self): # posterior for each arm
        self.alpha = np.ones(self.n_arms) * self.alpha0  # successes + alpha0
        self.beta = np.ones(self.n_arms) * self.beta0  # failures + beta0

    def select_arm(self): # sample a mean from each arm’s posterior
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples)) # arm with highest samples value

    def update(self, arm, reward): # Bayesian update
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

def run_bandit(bandit, policy, n_steps, show= False, algo_name: str | None = None):
    """
    Run a bandit algorithm on a given environment and compute metrics.
    For both stationary (BernoulliBandit) and non-stationary
    (NonStationaryBernoulliBandit with .get_means(t)) cases.
    """
    K = bandit.K
    arms = np.zeros(n_steps, dtype=int) # arm pulled at each step
    rewards = np.zeros(n_steps, dtype=float) # received reward at each step

    # store the true means at each time for regret and optimal-arm stats
    means_over_time = np.zeros((n_steps, K), dtype=float)

    # --- main interaction loop
    for t in range(n_steps):
        arm = policy.select_arm()

        try: # time argument (non-stationary)
            reward = bandit.pull(arm, t)
            means_t = bandit.get_means(t)
        except TypeError:
            # stationary bandit: no time argument
            reward = bandit.pull(arm)
            means_t = bandit.true_means

        policy.update(arm, reward)

        arms[t] = arm
        rewards[t] = reward
        means_over_time[t, :] = means_t

    # metrics
    cumulative_rewards = np.cumsum(rewards)
    avg_reward = cumulative_rewards / np.arange(1, n_steps + 1)

    best_means = means_over_time.max(axis=1) # best mean at each time
    played_means = means_over_time[np.arange(n_steps), arms] # mean of arm played

    instant_regret = best_means - played_means
    cumulative_regret = np.cumsum(instant_regret)
    avg_regret = cumulative_regret / np.arange(1, n_steps + 1)

    best_arms = means_over_time.argmax(axis=1) # best arm at each time
    is_optimal = (arms == best_arms) # was correct arm chosen?
    optimal_pull_fraction = is_optimal.mean() # optimal pulls fraction
    running_optimal_fraction = np.cumsum(is_optimal) / np.arange(1, n_steps + 1)

    pull_counts = np.bincount(arms, minlength=K) # times arm i  was pulled
    reward_sums = np.bincount(arms, weights=rewards, minlength=K) # reward from arm i

    empirical_means = np.zeros(K, dtype=float)
    nonzero = pull_counts > 0
    empirical_means[nonzero] = reward_sums[nonzero] / pull_counts[nonzero] # emp mean for each arm
    recommended_arm = int(np.argmax(empirical_means)) # recommended arm based on emp data

    # simple regret wrt the FINAL regime
    final_means = means_over_time[-1, :]
    simple_regret = final_means.max() - final_means[recommended_arm]

    # initial optimal arm/mean for reference
    initial_means = means_over_time[0, :]
    initial_optimal_arm = int(initial_means.argmax())
    initial_optimal_mean = float(initial_means.max())

    results = {
        "arms": arms,
        "rewards": rewards,
        "cumulative_rewards": cumulative_rewards,
        "avg_reward": avg_reward,
        "instant_regret": instant_regret,
        "cumulative_regret": cumulative_regret,
        "avg_regret": avg_regret,
        "simple_regret": simple_regret,
        "optimal_pull_fraction": optimal_pull_fraction,
        "running_optimal_fraction": running_optimal_fraction,
        "best_arms": best_arms, # time-varying optimal arm
        "pull_counts": pull_counts,
        "empirical_means": empirical_means,
        "recommended_arm": recommended_arm,
        "initial_optimal_arm": initial_optimal_arm,
        "initial_optimal_mean": initial_optimal_mean,
    }

    if show:
        if algo_name is None:
            algo_name = "Algorithm"
        _plot_single_run(bandit, results, algo_name)
    return results


def _plot_single_run(bandit: BernoulliBandit, res, algo_name= "Algorithm"):
    T = len(res["arms"])
    t = np.arange(1, T + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{algo_name} – Single Run Diagnostics")

    # 1) Cumulative regret
    ax = axes[0]
    ax.plot(t, res["cumulative_regret"])
    ax.set_xlabel("t")
    ax.set_ylabel("Cumulative pseudo-regret")
    ax.set_title("Cumulative Regret")

    # 2) Running fraction of optimal arm
    ax = axes[1]
    ax.plot(t, res["running_optimal_fraction"])
    ax.set_xlabel("t")
    ax.set_ylabel("Fraction optimal arm")
    ax.set_ylim(0, 1.05)
    ax.set_title("Running Optimal-Arm Fraction")

    # 3) Pull counts per arm
    ax = axes[2]
    K = bandit.K
    arms = np.arange(K)
    ax.bar(arms, res["pull_counts"])
    ax.set_xticks(arms)
    ax.set_xlabel("Arm")
    ax.set_ylabel("# pulls")
    ax.set_title("Arm Pull Counts")

    for i in range(K):
        ax.text(i, res["pull_counts"][i], f"{bandit.true_means[i]:.2f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    plt.show()


def run_many(bandit_constructor, PolicyClass, n_steps, n_runs, policy_name,**policy_kwargs):
    cumulative_regrets = np.zeros((n_runs, n_steps))

    for r in range(n_runs): # independent runs
        bandit = bandit_constructor()
        policy = PolicyClass(n_arms=bandit.K, **policy_kwargs)
        res = run_bandit(bandit, policy, n_steps)
        cumulative_regrets[r, :] = res["cumulative_regret"]

    return policy_name, cumulative_regrets


def run_many_stats(bandit_constructor, PolicyClass, n_steps, n_runs, **kwargs):
    """
    Run n_runs simulations and return metrics.
    """
    cum_regrets = []
    avg_regrets = []
    simple_regrets = []
    opt_fracs = []

    for _ in range(n_runs): # collect stats for each run
        bandit = bandit_constructor()
        policy = PolicyClass(n_arms=bandit.K, **kwargs)
        res = run_bandit(bandit, policy, n_steps)

        cum_regrets.append(res["cumulative_regret"][-1])
        avg_regrets.append(res["avg_regret"][-1])
        simple_regrets.append(res["simple_regret"])
        opt_fracs.append(res["optimal_pull_fraction"])

    return {
        "cum_regret_mean": float(np.mean(cum_regrets)),
        "cum_regret_std": float(np.std(cum_regrets)),
        "avg_regret_mean": float(np.mean(avg_regrets)),
        "avg_regret_std": float(np.std(avg_regrets)),
        "simple_regret_mean": float(np.mean(simple_regrets)),
        "simple_regret_std": float(np.std(simple_regrets)),
        "optimal_frac_mean": float(np.mean(opt_fracs)),
        "optimal_frac_std": float(np.std(opt_fracs)),
    }


def plot_cumulative_regret_multi(results_dict):
    plt.figure(figsize=(7, 4))
    for name, cum_regrets in results_dict.items():
        n_runs, n_steps = cum_regrets.shape
        t = np.arange(1, n_steps + 1)

        mean_regret = cum_regrets.mean(axis=0)
        lower = np.percentile(cum_regrets, 10, axis=0) # confidence bands
        upper = np.percentile(cum_regrets, 90, axis=0)

        plt.plot(t, mean_regret, label=name)
        plt.fill_between(t, lower, upper, alpha=0.2)

    plt.xlabel("t")
    plt.ylabel("Cumulative pseudo-regret")
    plt.title("Cumulative Regret – Multiple Runs")
    plt.legend()
    plt.tight_layout()
    plt.show()


class NonStationaryBernoulliBandit:
    """
    Piecewise constant Bernoulli bandit.
    true_means_before: list of p's before the switch
    true_means_after: list of p's after the switch
    switch_time:  time t at which the change occurs
    """
    def __init__(self, true_means_before, true_means_after, switch_time):
        assert len(true_means_before) == len(true_means_after)
        self.true_means_before = np.array(true_means_before, float)
        self.true_means_after = np.array(true_means_after, float)
        self.switch_time = switch_time
        self.K = len(true_means_before)

    def get_means(self, t): # choose correct means based on time
        if t < self.switch_time:
            return self.true_means_before
        else:
            return self.true_means_after

    def pull(self, arm, t): # time-dependent sampling
        p = self.get_means(t)[arm]
        return np.random.rand() < p

    @property
    def true_means(self): # initial means
        return self.true_means_before

    @property
    def optimal_mean(self):
        return max(self.true_means_before)

    @property
    def optimal_arm(self):
        return int(np.argmax(self.true_means_before))



if __name__ == "__main__":
    np.random.seed(0)

    # true_means = [0.1, 0.2, 0.3, 0.35] # means1

    # K = 20
    # base = 0.4
    # gap = 0.04
    # true_means = [base] * (K - 1) + [base + gap] # means2

    means_before = [0.30, 0.35, 0.40]  # arm 2 is best
    means_after = [0.40, 0.35, 0.30]  # arm 0 becomes best
    switch_time = 5000 # means3
    bandit = NonStationaryBernoulliBandit(means_before, means_after, switch_time)
    true_means = bandit.true_means_before

    T = 10_000
    N_RUNS = 50

    # # for means1 and means2:
    # ucb_stats = run_many_stats(lambda: BernoulliBandit(true_means),
    #                            UCB1, T, N_RUNS, c=2.0)
    #
    # ts_stats = run_many_stats(lambda: BernoulliBandit(true_means),
    #                           ThompsonSamplingBernoulli, T, N_RUNS,
    #                           alpha0=1.0, beta0=1.0)

    # for means3:
    ucb_stats = run_many_stats(
        lambda: NonStationaryBernoulliBandit(means_before, means_after, switch_time),
        UCB1,
        T,
        N_RUNS,
        c=2.0)

    ts_stats = run_many_stats(
        lambda: NonStationaryBernoulliBandit(means_before, means_after, switch_time),
        ThompsonSamplingBernoulli,
        T,
        N_RUNS,
        alpha0=1.0,
        beta0=1.0)

    print("UCB1:", ucb_stats)
    print("TS:", ts_stats)

    # # for means1 and means2:
    # name_ucb, cum_ucb = run_many(true_means, UCB1, T, N_RUNS,
    #                              policy_name="UCB1", c=2.0)
    # name_ts, cum_ts = run_many(true_means, ThompsonSamplingBernoulli, T, N_RUNS,
    #                            policy_name="TS", alpha0=1.0, beta0=1.0)

    # for means3:
    name_ucb, cum_ucb  = run_many(
        lambda: NonStationaryBernoulliBandit(means_before, means_after, switch_time),
        UCB1,
        T,
        N_RUNS,
        policy_name="UCB1",
        c=2.0)
    name_ts, cum_ts = run_many(
        lambda: NonStationaryBernoulliBandit(means_before, means_after, switch_time),
        ThompsonSamplingBernoulli,
        T,
        N_RUNS,
        policy_name="TS",
        alpha0=1.0, beta0=1.0)

    plot_cumulative_regret_multi({
        name_ucb: cum_ucb,
        name_ts: cum_ts,
    })
