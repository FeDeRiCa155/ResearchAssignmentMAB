# Gaussian Bandit implementation

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Protocol


class GaussianBandit:
    """
    Stochastic K-armed bandit with Gaussian rewards.
    X_{i,t} ~ N(mu_i, sigma_i^2)
    """
    def __init__(self, means, sigmas):
        self.true_means = np.array(means, dtype=float) # mean list
        self.sigmas = np.array(sigmas, dtype=float) # stand. dev. list
        assert len(self.true_means) == len(self.sigmas)
        self.K = len(self.true_means) # number of arms

    def pull(self, arm):
        mu = self.true_means[arm]
        sigma = self.sigmas[arm]
        return np.random.normal(mu, sigma) # continuous reward


class NonStationaryGaussianBandit:
    """
    Gaussian bandit with a switch point.
    Before switch: N(mu_before[i], sigma_before[i]^2)
    After switch:  N(mu_after[i],  sigma_after[i]^2)
    """
    def __init__(self, means_before, means_after, sigmas_before, sigmas_after, switch_time):
        self.means_before = np.array(means_before, float)
        self.means_after = np.array(means_after, float)
        self.sigmas_before = np.array(sigmas_before, float)
        self.sigmas_after = np.array(sigmas_after, float)
        assert len(self.means_before) == len(self.means_after) == len(self.sigmas_before) == len(self.sigmas_after)
        self.switch_time = switch_time
        self.K = len(self.means_before)

    def get_means(self, t): # choose correct means based on time
        return self.means_before if t < self.switch_time else self.means_after

    def get_sigmas(self, t): # choose correct s.d. based on time
        return self.sigmas_before if t < self.switch_time else self.sigmas_after

    def pull(self, arm, t): # time-dependent
        means_t = self.get_means(t)
        sigmas_t = self.get_sigmas(t)
        return np.random.normal(means_t[arm], sigmas_t[arm])

    @property
    def true_means(self):
        return self.means_before


class Policy(Protocol):
    def select_arm(self):
        ...

    def update(self, arm, reward):
        ...


@dataclass
class GaussianUCB:
    n_arms: int
    sigma: float
    c: float = 1.0 # exploration constant

    def __post_init__(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.estimates = np.zeros(self.n_arms, dtype=float)
        self.t = 0

    def select_arm(self):
        self.t += 1

        # play each arm once initially
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # UCB index
        bonus = np.sqrt(2 * (self.sigma ** 2) * np.log(self.t) / self.counts) # 2*\sigma^2 for variance
        ucb_values = self.estimates + self.c * bonus
        return int(np.argmax(ucb_values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.estimates[arm] += (reward - self.estimates[arm]) / n



@dataclass
class ThompsonSamplingGaussian:
    n_arms: int
    sigma: float
    m0: float = 0.0 # prior mean
    v0: float = 1.0 # prior variance

    def __post_init__(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sum_rewards = np.zeros(self.n_arms, dtype=float)

    def _posterior_params(self): # posterior for each arm
        n = self.counts
        S = self.sum_rewards
        sigma2 = self.sigma ** 2

        tau0 = 1.0 / self.v0
        tau = 1.0 / sigma2

        # bayesian posterior
        post_prec = tau0 + n * tau
        v = 1.0 / post_prec # posterior variance
        m = v * (tau0 * self.m0 + tau * S)  # posterior mean
        return m, v

    def select_arm(self):
        m, v = self._posterior_params()
        samples = np.random.normal(m, np.sqrt(v))
        return int(np.argmax(samples)) # arm with highest sample

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward



def run_bandit(bandit, policy: Policy, n_steps, show=False, algo_name: str | None=None):
    """
    Run a bandit algorithm on a given environment and compute metrics.
    For both stationary (GaussianBandit) and non-stationary
    (NonStationaryGaussianBandit with .get_means(t)) cases.
    """
    K = bandit.K
    arms = np.zeros(n_steps, dtype=int) # arm pulled at each step
    rewards = np.zeros(n_steps, dtype=float) # received reward at each step
    means_over_time = np.zeros((n_steps, K), dtype=float) # true means

    # --- main interaction loop
    for t in range(n_steps):
        arm = policy.select_arm()

        try: # non-stationary
            reward = bandit.pull(arm, t)
            means_t = bandit.get_means(t)
        except TypeError: # stationary bandit
            reward = bandit.pull(arm)
            means_t = bandit.true_means

        policy.update(arm, reward)

        arms[t] = arm
        rewards[t] = reward
        means_over_time[t, :] = means_t

    # metrics
    cumulative_rewards = np.cumsum(rewards)
    avg_reward = cumulative_rewards / np.arange(1, n_steps + 1)

    best_means = means_over_time.max(axis=1)
    played_means = means_over_time[np.arange(n_steps), arms]

    instant_regret = best_means - played_means
    cumulative_regret = np.cumsum(instant_regret)
    avg_regret = cumulative_regret / np.arange(1, n_steps + 1)

    best_arms = means_over_time.argmax(axis=1)
    is_optimal = (arms == best_arms)
    optimal_pull_fraction = is_optimal.mean()
    running_optimal_fraction = np.cumsum(is_optimal) / np.arange(1, n_steps + 1)

    pull_counts = np.bincount(arms, minlength=K)
    reward_sums = np.bincount(arms, weights=rewards, minlength=K)

    empirical_means = np.zeros(K, dtype=float)
    nonzero = pull_counts > 0
    empirical_means[nonzero] = reward_sums[nonzero] / pull_counts[nonzero]

    recommended_arm = int(np.argmax(empirical_means))

    # simple regret wrt the final regime
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
        "best_arms": best_arms,
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


def _plot_single_run(bandit: GaussianBandit, res, algo_name = "Algorithm"):
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


def run_many(bandit_constructor, PolicyClass, n_steps,n_runs, policy_name, **policy_kwargs):
    cumulative_regrets = np.zeros((n_runs, n_steps))

    for r in range(n_runs): # independent runs
        bandit = bandit_constructor()
        policy = PolicyClass(n_arms=bandit.K, **policy_kwargs)
        res = run_bandit(bandit, policy, n_steps)
        cumulative_regrets[r, :] = res["cumulative_regret"]

    return policy_name, cumulative_regrets


def run_many_stats(bandit_constructor, PolicyClass, n_steps,n_runs,**kwargs):
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


if __name__ == "__main__":
    np.random.seed(0)

    # means1
    # means = [0.0, 0.03, 0.06, 0.08]
    # sigmas = [0.05, 0.05, 0.05, 0.05]   # equal noise

    # means2
    K = 20
    base = 0.04
    gap = 0.01
    means = [base] * (K - 1) + [base + gap]
    sigmas = [0.05] * K

    # means3
    means_before = [0.02, 0.03, 0.06]  # arm 2 is best
    sigmas_before = [0.03, 0.03, 0.03]
    sigmas_after = [0.05, 0.03, 0.03]
    means_after = [0.06, 0.03, 0.02]  # arm 0 becomes best
    switch_time = 5000  # means3


    T = 10_000
    N_RUNS = 50

    # means1 and means2:
    # ucb_stats = run_many_stats(
    #     lambda: GaussianBandit(means, sigmas),
    #     GaussianUCB,
    #     T,
    #     N_RUNS,
    #     sigma=0.05,
    #     c=1.0)
    #
    # ts_stats = run_many_stats(
    #     lambda: GaussianBandit(means, sigmas),
    #     ThompsonSamplingGaussian,
    #     T,
    #     N_RUNS,
    #     sigma=0.05,
    #     m0=0.0,
    #     v0=1.0)

    # for means3:
    ucb_stats = run_many_stats(
        lambda: NonStationaryGaussianBandit(means_before, means_after, sigmas_before, sigmas_after, switch_time),
        GaussianUCB,
        T,
        N_RUNS,
        sigma=0.05,
        c=1.0)

    ts_stats = run_many_stats(
        lambda: NonStationaryGaussianBandit(means_before, means_after, sigmas_before, sigmas_after, switch_time),
        ThompsonSamplingGaussian,
        T,
        N_RUNS,
        sigma=0.05,
        m0=0.0,
        v0=1.0)

    print("UCB:", ucb_stats)
    print("TS :", ts_stats)

    # curves for the plot (means1 and means2):
    # name_ucb, cum_ucb = run_many(
    #     lambda: GaussianBandit(means, sigmas),
    #     GaussianUCB,
    #     T,
    #     N_RUNS,
    #     policy_name="UCB",
    #     sigma=0.05,
    #     c=1.0)
    #
    # name_ts, cum_ts = run_many(
    #     lambda: GaussianBandit(means, sigmas),
    #     ThompsonSamplingGaussian,
    #     T,
    #     N_RUNS,
    #     policy_name="TS",
    #     sigma=0.05,
    #     m0=0.0,
    #     v0=1.0)

    # means3:
    name_ucb, cum_ucb = run_many(
        lambda: NonStationaryGaussianBandit(means_before, means_after, sigmas_before, sigmas_after, switch_time),
        GaussianUCB,
        T,
        N_RUNS,
        policy_name="UCB1",
        sigma=0.05,
        c=1.0)

    name_ts, cum_ts = run_many(
        lambda: NonStationaryGaussianBandit(means_before, means_after, sigmas_before, sigmas_after, switch_time),
        ThompsonSamplingGaussian,
        T,
        N_RUNS,
        policy_name="TS",
        sigma=0.05,
        m0=0.0,
        v0=1.0)

    plot_cumulative_regret_multi({
        name_ucb: cum_ucb,
        name_ts: cum_ts})
