# "Energy Dispatch" scenario

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Protocol, Callable

# Environment
class EnergyDispatchBandit:
    """
    Each arm i is a dispatch option with realized cost:
        C_{i,t} = a_i + b_i * D_t + eps_{i,t}
    with:
        D_t ~ N(m_D, s_D^2) i.i.d. (exogenous demand deviation)
        eps_{i,t} ~ N(0, sigma_i^2) independent (option-specific noise)

    Bandit reward: X_{i,t} = - C_{i,t}
    """
    def __init__(
        self,
        a,
        b,
        sigmas,
        m_D= 0.0,
        s_D = 1.0,
    ):
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.sigmas = np.array(sigmas, dtype=float)
        assert len(self.a) == len(self.b) == len(self.sigmas)
        self.K = len(self.a)

        self.m_D = float(m_D)
        self.s_D = float(s_D)

        # Expected cost and reward (stationary)
        self._mu_cost = self.a + self.b * self.m_D
        self._mu_reward = -self._mu_cost

    @property
    def true_means(self):
        return self._mu_reward

    @property
    def expected_costs(self):
        return self._mu_cost

    @property
    def gaps(self):
        mu = self.true_means
        mu_star = mu.max()
        return mu_star - mu

    def pull(self, arm):
        # ex. deviation sampled once per round (but only observed through the selected arm)
        D_t = np.random.normal(self.m_D, self.s_D)
        cost = self.a[arm] + self.b[arm] * D_t + np.random.normal(0.0, self.sigmas[arm])
        reward = -cost
        return float(reward)


# Policies
class Policy(Protocol):
    def select_arm(self):
        ...
    def update(self, arm, reward):
        ...


@dataclass
class GaussianUCB:
    n_arms: int
    sigma: float
    c: float = 1.0

    def __post_init__(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.estimates = np.zeros(self.n_arms, dtype=float)
        self.t = 0

    def select_arm(self):
        self.t += 1

        # play each arm once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        bonus = np.sqrt(2 * (self.sigma ** 2) * np.log(self.t) / self.counts)
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
    m0: float = 0.0
    v0: float = 1.0

    def __post_init__(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sum_rewards = np.zeros(self.n_arms, dtype=float)

    def _posterior_params(self):
        n = self.counts
        S = self.sum_rewards
        sigma2 = self.sigma ** 2

        tau0 = 1.0 / self.v0
        tau = 1.0 / sigma2

        post_prec = tau0 + n * tau
        v = 1.0 / post_prec
        m = v * (tau0 * self.m0 + tau * S)
        return m, v

    def select_arm(self):
        m, v = self._posterior_params()
        samples = np.random.normal(m, np.sqrt(v))
        return int(np.argmax(samples))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward


# Run
def run_bandit(
    bandit,
    policy: Policy,
    n_steps,
):
    K = bandit.K
    arms = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps, dtype=float)

    means_over_time = np.zeros((n_steps, K), dtype=float)

    for t in range(n_steps):
        arm = policy.select_arm()
        reward = bandit.pull(arm)
        policy.update(arm, reward)

        arms[t] = arm
        rewards[t] = reward
        means_over_time[t, :] = bandit.true_means # stationary

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

    final_means = means_over_time[-1, :]
    simple_regret = final_means.max() - final_means[recommended_arm]

    return {
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
        "pull_counts": pull_counts,
        "empirical_means": empirical_means,
        "recommended_arm": recommended_arm,
    }


def run_many(
    bandit_constructor: Callable[[], object],
    PolicyClass,
    n_steps,
    n_runs,
    policy_name,
    **policy_kwargs,
):
    cumulative_regrets = np.zeros((n_runs, n_steps))
    for r in range(n_runs):
        bandit = bandit_constructor()
        policy = PolicyClass(n_arms=bandit.K, **policy_kwargs)
        res = run_bandit(bandit, policy, n_steps)
        cumulative_regrets[r, :] = res["cumulative_regret"]
    return policy_name, cumulative_regrets


def run_many_stats(
    bandit_constructor: Callable[[], object],
    PolicyClass,
    n_steps,
    n_runs,
    **policy_kwargs,
):
    cum_regrets, avg_regrets, simple_regrets, opt_fracs = [], [], [], []

    for _ in range(n_runs):
        bandit = bandit_constructor()
        policy = PolicyClass(n_arms=bandit.K, **policy_kwargs)
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



# Plots
def theory_shaped_curve(gaps, t, k = 1.0):
    gaps = np.asarray(gaps, dtype=float)
    mask = gaps > 1e-12
    if not np.any(mask):
        return np.zeros_like(t, dtype=float)

    coef = np.sum(1.0 / gaps[mask])
    return k * coef * np.log1p(t)


def plot_cumulative_regret_multi(
    results_dict,
    theory = None,
    theory_label = "Gap-shaped reference",
):
    plt.figure(figsize=(7, 4))
    for name, cum_regrets in results_dict.items():
        n_runs, n_steps = cum_regrets.shape
        t = np.arange(1, n_steps + 1)

        mean_regret = cum_regrets.mean(axis=0)
        median_regret = np.median(cum_regrets, axis=0)
        lower = np.percentile(cum_regrets, 10, axis=0)
        upper = np.percentile(cum_regrets, 90, axis=0)

        plt.plot(t, mean_regret, label=name)
        # plt.plot(t, median_regret, label=name)
        plt.fill_between(t, lower, upper, alpha=0.2)

    if theory is not None:
        t = np.arange(1, results_dict[next(iter(results_dict))].shape[1] + 1)
        plt.plot(t, theory, linestyle="--", linewidth=2, label=theory_label)

    plt.xlabel("t")
    plt.ylabel("Cumulative pseudo-regret")
    plt.title("Cumulative Regret - Multiple Runs")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Parameters
def make_dispatch_instance_easy():
    a = [10.0, 10.5, 11.0, 9.6, 10.2, 10.8]
    b = [0.8,  0.6,  0.4,  0.3,  0.5,  0.7]
    sigmas = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    return EnergyDispatchBandit(a=a, b=b, sigmas=sigmas, m_D=0.0, s_D=1.0)


def make_dispatch_instance_hard():
    a = [10.0, 10.05, 10.10, 10.02, 10.08, 10.06]
    b = [0.60, 0.55,  0.50,  0.58,  0.52,  0.54]
    sigmas = [0.40, 0.90, 0.60, 1.10, 0.50, 0.80]
    return EnergyDispatchBandit(a=a, b=b, sigmas=sigmas, m_D=0.0, s_D=1.0)


# Main
if __name__ == "__main__":
    np.random.seed(0)

    T = 10000
    N_RUNS = 50

    # bandit_constructor = make_dispatch_instance_easy
    bandit_constructor = make_dispatch_instance_hard

    env0 = bandit_constructor()

    # For heteroscedastic sigmas: sigma = max(sigmas)
    sigma_alg = float(np.max(env0.sigmas))

    ucb_stats = run_many_stats(
        bandit_constructor,
        GaussianUCB,
        T,
        N_RUNS,
        sigma=sigma_alg,
        c=1.0,
    )

    ts_stats = run_many_stats(
        bandit_constructor,
        ThompsonSamplingGaussian,
        T,
        N_RUNS,
        sigma=sigma_alg,
        m0=0.0,
        v0=1.0,
    )

    print("\nUCB:", ucb_stats)
    print("TS :", ts_stats)

    name_ucb, cum_ucb = run_many(
        bandit_constructor,
        GaussianUCB,
        T,
        N_RUNS,
        policy_name="UCB",
        sigma=sigma_alg,
        c=1.0,
    )

    name_ts, cum_ts = run_many(
        bandit_constructor,
        ThompsonSamplingGaussian,
        T,
        N_RUNS,
        policy_name="TS",
        sigma=sigma_alg,
        m0=0.0,
        v0=1.0,
    )

    t = np.arange(1, T + 1)
    theory = theory_shaped_curve(env0.gaps, t, k=0.1)

    plot_cumulative_regret_multi(
        {name_ucb: cum_ucb, name_ts: cum_ts},
        theory=None,
        theory_label=rf"$k \sum_{{i\neq i^*}} \log(1+t)/\Delta_i$ (k={0.1})",
    )

