# "Server Routing" contextual scenario

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Protocol


# environment
class ServerRoutingBandit:
    """
    Gaussian Linear Contextual Bandit

    Context: Traffic load s_t in [0, 1]
    Arms: Servers A, B, C
    Linear model: Latency_i(s_t) = θ_{i,0} + θ_{i,1} * s_t + noise
    Reward = -Latency
    """
    def __init__(self, complex_mode=False):
        self.complex_mode = complex_mode

        if not complex_mode:
            self.latency_params = [
                np.array([[0.1], [1.5]]), # A:"lightweight"
                np.array([[0.5], [0.5]]), # B: "balanced"
                np.array([[0.9], [0.1]]) # C: "heavy-duty"
            ]
            self.noise_stds = [0.05, 0.05, 0.05]
        else:
            self.latency_params = [
                np.array([[0.10], [3.5]]),
                np.array([[0.35], [2.3]]),
                np.array([[0.58], [1.4]]),
                np.array([[0.79], [0.9]]),
                np.array([[0.98], [0.4]]),
                np.array([[1.15], [0.1]])
            ]
            self.noise_stds = [0.03, 0.35, 0.08, 0.45, 0.12, 0.30]

        self.K = len(self.latency_params)

    def sample_context(self):
        load = np.random.rand()
        features = np.array([1.0, load])
        return load, features

    def pull(self, arm, features):
        x = features.reshape(-1, 1)
        true_latency = (self.latency_params[arm].T @ x).item()
        obs_latency = true_latency + np.random.normal(0, self.noise_stds[arm])
        return -obs_latency

    def optimal_arm(self, features):
        x = features.reshape(-1, 1)
        latencies = [(theta.T @ x).item() for theta in self.latency_params]
        return int(np.argmin(latencies))

    def optimal_reward(self, features):
        x = features.reshape(-1, 1)
        latencies = [(theta.T @ x).item() for theta in self.latency_params]
        return -min(latencies)

# policies
class ContextualPolicy(Protocol):
    def select_arm(self, features):
        ...
    def update(self, arm, features, reward):
        ...


@dataclass
class ContextFreeUCB: #Standard UCB1: ignores context (baseline)
    n_arms: int
    n_features: int = 0

    def __post_init__(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.estimates = np.zeros(self.n_arms)
        self.t = 0

    def select_arm(self, features):
        self.t += 1
        # Play each arm once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # UCB formula
        bonus = np.sqrt(2 * np.log(self.t) / self.counts)
        ucb_values = self.estimates + bonus
        return int(np.argmax(ucb_values))

    def update(self, arm, features, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.estimates[arm] += (reward - self.estimates[arm]) / n


@dataclass
class LinUCB:
    # LinUCB with disjoint linear models with separate Ridge regression for each arm
    n_arms: int
    n_features: int
    alpha: float = 0.1 # param

    def __post_init__(self):
        # For each arm: A = X^T X + I, b = X^T y
        self.A = [np.identity(self.n_features) for _ in range(self.n_arms)]
        self.b = [np.zeros((self.n_features, 1)) for _ in range(self.n_arms)]

    def select_arm(self, features):
        x = features.reshape(-1, 1)
        ucb_values = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            theta_hat = np.linalg.solve(self.A[a], self.b[a])
            mean = (theta_hat.T @ x).item()
            A_inv_x = np.linalg.solve(self.A[a], x)
            bonus = self.alpha * np.sqrt((x.T @ A_inv_x).item())

            ucb_values[a] = mean + bonus

        return int(np.argmax(ucb_values))

    def update(self, arm, features, reward):
        x = features.reshape(-1, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x


@dataclass
class LinearThompsonSampling:
    # Thompson Sampling for linear contextual bandits
    # Bayesian approach with Gaussian posterior

    n_arms: int
    n_features: int
    lambda_prior: float = 1.0 # param
    noise_std: float = 0.1 # param

    def __post_init__(self):
        # For each arm: posterior N(mu_hat, v^2 * B^-1)
        self.B = [self.lambda_prior * np.identity(self.n_features)
                  for _ in range(self.n_arms)]
        self.f = [np.zeros((self.n_features, 1)) for _ in range(self.n_arms)]
        self.mu_hat = [np.zeros((self.n_features, 1)) for _ in range(self.n_arms)]

    def select_arm(self, features):
        x = features.reshape(-1, 1)
        samples = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            B_inv = np.linalg.inv(self.B[a])
            cov = (self.noise_std ** 2) * B_inv

            theta_sample = np.random.multivariate_normal(
                self.mu_hat[a].ravel(),
                cov
            )
            samples[a] = theta_sample @ x.ravel()

        return int(np.argmax(samples))

    def update(self, arm, features, reward):
        x = features.reshape(-1, 1)
        self.B[arm] += x @ x.T
        self.f[arm] += reward * x
        self.mu_hat[arm] = np.linalg.solve(self.B[arm], self.f[arm])


# simulation
def run_single_simulation(bandit, policy: ContextualPolicy, n_steps):
    cumulative_regret = np.zeros(n_steps)
    running_regret = 0.0
    optimal_choices = 0

    for t in range(n_steps):
        ctx_raw, features = bandit.sample_context()

        arm = policy.select_arm(features)

        if isinstance(bandit, ServerRoutingBandit):
            reward = bandit.pull(arm, features)  # Uses feature vector
            optimal_arm = bandit.optimal_arm(features)
            opt_reward = bandit.optimal_reward(features)
            x = features.reshape(-1, 1)
            expected_latency = (bandit.latency_params[arm].T @ x).item()
            expected_reward = -expected_latency

        policy.update(arm, features, reward)

        if arm == optimal_arm:
            optimal_choices += 1

        inst_regret = opt_reward - expected_reward
        running_regret += inst_regret
        cumulative_regret[t] = running_regret

    optimal_fraction = optimal_choices / n_steps
    return cumulative_regret, optimal_fraction


def run_experiments(
        bandit_factory,
        n_steps = 10000,
        n_runs = 20,
        feature_dim = 2
):

    algorithms = [
        # ("UCB (Context-Free)", ContextFreeUCB, {}),
        ("LinUCB", LinUCB, {'alpha': 0.1}),
        ("Linear TS", LinearThompsonSampling, {'noise_std': 0.05, 'lambda_prior': 2.0})
    ]

    bandit_name = bandit_factory().__class__.__name__
    print("\n")
    print(f"Scenario: {bandit_name}")
    print(f"{'Algorithm':<25} | {'Final Regret':<20} | {'Avg Regret':<12} | {'Opt %':<8}")

    results = {}

    for name, PolicyClass, kwargs in algorithms:
        regret_curves = []
        opt_fractions = []

        for _ in range(n_runs):
            bandit = bandit_factory()
            policy = PolicyClass(n_arms=bandit.K, n_features=feature_dim, **kwargs)
            regret_curve, opt_frac = run_single_simulation(bandit, policy, n_steps)
            regret_curves.append(regret_curve)
            opt_fractions.append(opt_frac)

        regret_curves = np.array(regret_curves)

        # Statistics
        final_regret_mean = regret_curves[:, -1].mean()
        final_regret_std = regret_curves[:, -1].std()
        avg_regret_mean = final_regret_mean / n_steps
        avg_regret_std = final_regret_std / n_steps
        opt_pct = np.mean(opt_fractions) * 100

        print(f"{name:<25} | {final_regret_mean:>8.1f} ± {final_regret_std:<7.1f} | "
              f"{avg_regret_mean:>6.4f} ± {avg_regret_std:<6.4f} | {opt_pct:>6.1f}%")

        results[name] = {
            'curves': regret_curves,
            'mean_curve': regret_curves.mean(axis=0),
            'final_regret': final_regret_mean,
            'std': final_regret_std,
            'opt_frac': np.mean(opt_fractions)
        }

    return results


def plot_results(results, title):
    plt.figure(figsize=(10, 6))

    for name, data in results.items():
        mean_curve = data['mean_curve']
        all_curves = data['curves']

        t = np.arange(len(mean_curve))

        std_curve = all_curves.std(axis=0)
        n_runs = all_curves.shape[0]
        se_curve = std_curve / np.sqrt(n_runs)

        plt.plot(t, mean_curve, label=name, linewidth=2)

        lower = mean_curve - 2 * se_curve
        upper = mean_curve + 2 * se_curve
        plt.fill_between(t, lower, upper, alpha=0.2)

    plt.xlabel("t", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.title(title)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


#main
if __name__ == "__main__":
    np.random.seed(42)

    N_STEPS = 10000
    N_RUNS = 20

    server = ServerRoutingBandit()

    server_results = run_experiments(
        lambda: ServerRoutingBandit(complex_mode=True),
        n_steps=N_STEPS,
        n_runs=N_RUNS,
        feature_dim=2
    )

    plot_results(
        server_results,
        "Cumulative Rgeret - Multiple Runs"
    )

#
#
# def run_single_simulation_diagnostics(bandit, policy, n_steps):
#
#     loads = np.zeros(n_steps)
#     chosen_arms = np.zeros(n_steps, dtype=int)
#     inst_regret = np.zeros(n_steps)
#     optimal_ind = np.zeros(n_steps, dtype=int)
#     cumulative_regret = np.zeros(n_steps)
#
#     running_regret = 0.0
#
#     for t in range(n_steps):
#         load, features = bandit.sample_context()
#         loads[t] = load
#
#         arm = policy.select_arm(features)
#         chosen_arms[t] = arm
#
#         reward = bandit.pull(arm, features)
#         opt_arm = bandit.optimal_arm(features)
#         opt_reward = bandit.optimal_reward(features)
#
#         x = features.reshape(-1, 1)
#         expected_latency = (bandit.latency_params[arm].T @ x).item()
#         expected_reward = -expected_latency
#
#         r_t = opt_reward - expected_reward
#         inst_regret[t] = r_t
#
#         if arm == opt_arm:
#             optimal_ind[t] = 1
#
#         policy.update(arm, features, reward)
#
#         running_regret += r_t
#         cumulative_regret[t] = running_regret
#
#     return {
#         "loads": loads,
#         "chosen_arms": chosen_arms,
#         "inst_regret": inst_regret,
#         "optimal_ind": optimal_ind,
#         "cum_regret": cumulative_regret,
#     }
#
#
# def rolling_mean(x, window):
#     if window <= 1:
#         return x.copy()
#     w = np.ones(window) / window
#     return np.convolve(x, w, mode="valid")
#
#
# def plot_avg_regret(results_by_algo, title="Average regret $R_t/t$"):
#     plt.figure(figsize=(10, 6))
#     for name, data in results_by_algo.items():
#         R = data["mean_curve"]
#         t = np.arange(1, len(R) + 1)
#         avg = R / t
#         plt.plot(t, avg, label=name, linewidth=2)
#
#     plt.xlabel("t")
#     plt.ylabel("Average Regret  $R_t/t$")
#     plt.title(title)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_rolling_opt_rate(diagnostics_by_algo, window= 2000,
#                           title="Rolling optimal-action rate"):
#     plt.figure(figsize=(10, 6))
#     for name, diag in diagnostics_by_algo.items():
#         opt = diag["optimal_ind"].astype(float)
#         opt_roll = rolling_mean(opt, window)
#         t = np.arange(len(opt_roll))
#         plt.plot(t, opt_roll, label=name, linewidth=2)
#
#     plt.xlabel("t")
#     plt.ylabel(f"Rolling mean for 1[a_t = a*(x_t)] (window={window})")
#     plt.title(title)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
#
# def run_diagnostics(bandit_factory, algorithms, n_steps=10000, feature_dim=2, n_runs=20, seed=42):
#     np.random.seed(seed)
#
#     diagnostics_by_algo = {}
#     for name, PolicyClass, kwargs in algorithms:
#         loads_all = []
#         inst_regret_all = []
#         optimal_ind_all = []
#
#         for _ in range(n_runs):
#             bandit = bandit_factory()
#             policy = PolicyClass(n_arms=bandit.K, n_features=feature_dim, **kwargs)
#
#             diag = run_single_simulation_diagnostics(bandit, policy, n_steps)
#             loads_all.append(diag["loads"])
#             inst_regret_all.append(diag["inst_regret"])
#             optimal_ind_all.append(diag["optimal_ind"])
#
#         diagnostics_by_algo[name] = {
#             "loads": np.mean(np.stack(loads_all, axis=0), axis=0),
#             "inst_regret": np.mean(np.stack(inst_regret_all, axis=0), axis=0),
#             "optimal_ind": np.mean(np.stack(optimal_ind_all, axis=0), axis=0),
#         }
#
#     return diagnostics_by_algo
#
#
# if __name__ == "__main__":
#     np.random.seed(42)
#
#     algorithms = [
#         ("LinUCB", LinUCB, {"alpha": 0.1}),
#         ("Linear TS", LinearThompsonSampling, {"noise_std": 0.05, "lambda_prior": 2.0}),
#     ]
#
#     bandit_factory = lambda: ServerRoutingBandit(complex_mode=True)
#
#     results = run_experiments(
#         bandit_factory,
#         n_steps=10000,
#         n_runs=20,
#         feature_dim=2
#     )
#
#     # Plot 1: Average regret R_t / t
#     plot_avg_regret(results, title="Average regret $R_t/t$ (mean over runs)")
#
#     diagnostics = run_diagnostics(
#         bandit_factory,
#         algorithms=algorithms,
#         n_steps=10000,
#         feature_dim=2,
#         n_runs=20
#     )
#
#     # Plot 2: Rolling optimal-action rate
#     plot_rolling_opt_rate(diagnostics, window=2000,
#                           title="Rolling optimal-action rate")
# #
