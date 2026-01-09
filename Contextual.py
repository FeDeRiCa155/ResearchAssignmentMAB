# "Server Routing" contextual scenario

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Protocol


# environment
class ServerRoutingBandit:
    """
    Context: Traffic load s_t in [0, 1]
    Linear model: Latency_i(s_t) = \theta_{i,0} + \theta_{i,1} * s_t + noise
    Reward = -Latency
    """
    def __init__(self, complex_mode=False):
        self.complex_mode = complex_mode # modes: simple, complex

        if not complex_mode: # simple
            self.latency_params = [
                np.array([[0.1], [1.5]]), # "lightweight"
                np.array([[0.5], [0.5]]), # "balanced"
                np.array([[0.9], [0.1]]) # "heavy-duty"
            ]
            self.noise_stds = [0.05, 0.05, 0.05]
        else: # complex
            self.latency_params = [
                np.array([[0.10], [3.5]]),
                np.array([[0.35], [2.3]]),
                np.array([[0.58], [1.4]]),
                np.array([[0.79], [0.9]]),
                np.array([[0.98], [0.4]]),
                np.array([[1.15], [0.1]])
            ]
            self.noise_stds = [0.03, 0.35, 0.08, 0.45, 0.12, 0.30]

        self.K = len(self.latency_params) # number of servers (arms)

    def sample_context(self):
        load = np.random.rand()
        features = np.array([1.0, load])
        return load, features

    def pull(self, arm, features):
        x = features.reshape(-1, 1) # x = [1, s_t]^T
        true_latency = (self.latency_params[arm].T @ x).item() # \theta^T @ x to scalar
        obs_latency = true_latency + np.random.normal(0, self.noise_stds[arm])
        return -obs_latency # reward = - latency

    def optimal_arm(self, features):
        x = features.reshape(-1, 1)
        latencies = [(theta.T @ x).item() for theta in self.latency_params] # all servers' latencies
        return int(np.argmin(latencies)) # server with lowest latency

    def optimal_reward(self, features):
        x = features.reshape(-1, 1)
        latencies = [(theta.T @ x).item() for theta in self.latency_params]
        return -min(latencies) # best achievable reward

# policies
class ContextualPolicy(Protocol):
    def select_arm(self, features):
        ...
    def update(self, arm, features, reward):
        ...


@dataclass
class ContextFreeUCB: # standard UCB1: ignores context (baseline)
    n_arms: int
    n_features: int = 0 # features are ignored

    def __post_init__(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.estimates = np.zeros(self.n_arms)
        self.t = 0

    def select_arm(self, features):
        self.t += 1

        # play each arm once initially
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # UCB index
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
    alpha: float = 0.1 # exploration parameter

    def __post_init__(self):
        # for each arm: A = X^T X + I, b = X^T y
        self.A = [np.identity(self.n_features) for _ in range(self.n_arms)]
        self.b = [np.zeros((self.n_features, 1)) for _ in range(self.n_arms)]

    def select_arm(self, features):
        x = features.reshape(-1, 1)
        ucb_values = np.zeros(self.n_arms)

        for a in range(self.n_arms): # for each arm compute UCB index
            theta_hat = np.linalg.solve(self.A[a], self.b[a]) # A @ \hat{\theta} = b
            mean = (theta_hat.T @ x).item() # estimated mean given context x
            A_inv_x = np.linalg.solve(self.A[a], x)
            bonus = self.alpha * np.sqrt((x.T @ A_inv_x).item()) # uncertainty bonus

            ucb_values[a] = mean + bonus
        return int(np.argmax(ucb_values))

    def update(self, arm, features, reward): # update Ridge regression
        x = features.reshape(-1, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x


@dataclass
class LinearThompsonSampling:
    # Thompson Sampling for linear contextual bandits
    # Bayesian approach with Gaussian posterior

    n_arms: int
    n_features: int
    lambda_prior: float = 1.0 # prior parameter
    noise_std: float = 0.1 # assumed noise parameter

    def __post_init__(self):
        self.B = [self.lambda_prior * np.identity(self.n_features)
                  for _ in range(self.n_arms)]
        self.f = [np.zeros((self.n_features, 1)) for _ in range(self.n_arms)]
        self.mu_hat = [np.zeros((self.n_features, 1)) for _ in range(self.n_arms)]

    def select_arm(self, features):
        x = features.reshape(-1, 1)
        samples = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            B_inv = np.linalg.inv(self.B[a])
            cov = (self.noise_std ** 2) * B_inv # cov = sigma^2 * B^{-1}

            theta_sample = np.random.multivariate_normal(self.mu_hat[a].ravel(),cov) # \theta = N(\hat{\mu}, cov)
            samples[a] = theta_sample @ x.ravel()

        return int(np.argmax(samples)) # arm with highest sampled value

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
        ctx_raw, features = bandit.sample_context() # sample new context each round
        arm = policy.select_arm(features)

        if isinstance(bandit, ServerRoutingBandit):
            reward = bandit.pull(arm, features)
            optimal_arm = bandit.optimal_arm(features)
            opt_reward = bandit.optimal_reward(features)
            x = features.reshape(-1, 1)
            expected_latency = (bandit.latency_params[arm].T @ x).item()
            expected_reward = -expected_latency # for decision quality

        policy.update(arm, features, reward)

        if arm == optimal_arm:
            optimal_choices += 1 # optimal arm choice tracker

        inst_regret = opt_reward - expected_reward # deterministic: E(reward) used
        running_regret += inst_regret
        cumulative_regret[t] = running_regret

    optimal_fraction = optimal_choices / n_steps
    return cumulative_regret, optimal_fraction


def run_experiments(bandit_factory, n_steps = 10000, n_runs = 20, feature_dim=2):

    algorithms = [
        # ("UCB (Context-Free)", ContextFreeUCB, {}),
        ("LinUCB", LinUCB, {'alpha': 0.1}),
        ("Linear TS", LinearThompsonSampling, {'noise_std': 0.05, 'lambda_prior': 2.0})
    ]

    bandit_name = bandit_factory().__class__.__name__
    print("\n") # result table
    print(f"Scenario: {bandit_name}")
    print(f"{'Algorithm':<25} | {'Final Regret':<20} | {'Avg Regret':<12} | {'Opt %':<8}")

    results = {}

    for name, PolicyClass, kwargs in algorithms:
        regret_curves = []
        opt_fractions = []

        for _ in range(n_runs): # independent runs
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
        plt.fill_between(t, lower, upper, alpha=0.2) # bands

    plt.xlabel("t", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.title(title)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


#main
# if __name__ == "__main__":
#     np.random.seed(42)
#
#     N_STEPS = 10000
#     N_RUNS = 20
#
#     server = ServerRoutingBandit()
#
#     server_results = run_experiments(
#         lambda: ServerRoutingBandit(complex_mode=True),
#         n_steps=N_STEPS,
#         n_runs=N_RUNS,
#         feature_dim=2
#     )
#
#     plot_results(
#         server_results,
#         "Cumulative Rgeret - Multiple Runs"
#     )


# SWEEP PARAMETERS
def evaluate_policy_on_bandit(bandit_factory, PolicyClass, policy_kwargs, n_steps=10000, n_runs=20, feature_dim=2, seed=42):
    regret_curves = []
    opt_fractions = []

    for r in range(n_runs):

        bandit = bandit_factory()
        policy = PolicyClass(n_arms=bandit.K, n_features=feature_dim, **policy_kwargs)
        regret_curve, opt_frac = run_single_simulation(bandit, policy, n_steps)

        regret_curves.append(regret_curve)
        opt_fractions.append(opt_frac)

    regret_curves = np.array(regret_curves)
    final_regrets = regret_curves[:, -1]

    summary = {
        "final_regret_mean": float(final_regrets.mean()),
        "final_regret_std": float(final_regrets.std(ddof=1)),
        "avg_regret_mean": float(final_regrets.mean() / n_steps),
        "avg_regret_std": float(final_regrets.std(ddof=1) / n_steps),
        "opt_frac_mean": float(np.mean(opt_fractions)),
        "opt_frac_std": float(np.std(opt_fractions, ddof=1)),
        "n_runs": int(n_runs),
        "n_steps": int(n_steps),
    }
    return summary


def sweep_linucb_alpha(bandit_factory,alphas, n_steps=10000, n_runs=20, feature_dim=2,base_seed=42):
    # sweep over different alpha values for LinUCB
    rows = []
    for i, a in enumerate(alphas):
        summary = evaluate_policy_on_bandit(
            bandit_factory=bandit_factory,
            PolicyClass=LinUCB,
            policy_kwargs={"alpha": float(a)},
            n_steps=n_steps,
            n_runs=n_runs,
            feature_dim=feature_dim,
            seed=base_seed + i,
        )
        rows.append({
            "algo": "LinUCB",
            "alpha": float(a),
            "lambda_prior": None,
            "noise_std": None,
            **summary
        })
    return rows


def sweep_linear_ts(bandit_factory,lambda_priors, noise_stds, n_steps=10000,n_runs=20,feature_dim=2,base_seed=42):
    # gird searchover lambda and sigma values for Linear TS
    rows = []
    idx = 0
    for lam in lambda_priors:
        for sig in noise_stds:
            summary = evaluate_policy_on_bandit(
                bandit_factory=bandit_factory,
                PolicyClass=LinearThompsonSampling,
                policy_kwargs={"lambda_prior": float(lam), "noise_std": float(sig)},
                n_steps=n_steps,
                n_runs=n_runs,
                feature_dim=feature_dim,
                seed=base_seed + idx,
            )
            rows.append({
                "algo": "Linear TS",
                "alpha": None,
                "lambda_prior": float(lam),
                "noise_std": float(sig),
                **summary
            })
            idx += 1
    return rows


def print_sweep_table(rows, sort_by="final_regret_mean"):
    rows_sorted = sorted(rows, key=lambda r: r.get(sort_by, float("inf")))
    print("\n") # resulting table
    header = (f"{'Algo':<10} | {'alpha':<7} | {'lambda':<7} | {'sigma':<7} | "
        f"{'Final Regret (mean±std)':<26} | {'Avg Regret':<16} | {'Opt% (mean±std)':<18}")
    print(header)

    for r in rows_sorted:
        algo = r["algo"]
        alpha = "-" if r["alpha"] is None else f"{r['alpha']:.3g}"
        lam = "-" if r["lambda_prior"] is None else f"{r['lambda_prior']:.3g}"
        sig = "-" if r["noise_std"] is None else f"{r['noise_std']:.3g}"

        final_str = f"{r['final_regret_mean']:.1f}±{r['final_regret_std']:.1f}"
        avg_str = f"{r['avg_regret_mean']:.4f}±{r['avg_regret_std']:.4f}"
        opt_str = f"{(100*r['opt_frac_mean']):.1f}±{(100*r['opt_frac_std']):.1f}"

        print(
            f"{algo:<10} | {alpha:<7} | {lam:<7} | {sig:<7} | "
            f"{final_str:<26} | {avg_str:<16} | {opt_str:<18}"
        )

if __name__ == "__main__":
    np.random.seed(42)

    N_STEPS = 10000
    N_RUNS = 20

    # simple environment
    bandit_factory_simple = lambda: ServerRoutingBandit(complex_mode=False)

    # LinUCB sweep (alpha)
    alphas = [0.01, 0.03, 0.1, 0.3, 1.0]
    lin_rows = sweep_linucb_alpha(
        bandit_factory_simple,
        alphas=alphas,
        n_steps=N_STEPS,
        n_runs=N_RUNS,
        feature_dim=2,
    )

    # Linear TS sweep (lambda, sigma)
    lambda_priors = [0.5, 1.0, 2.0, 5.0]
    noise_stds = [0.02, 0.05, 0.1, 0.2]
    ts_rows = sweep_linear_ts(
        bandit_factory_simple,
        lambda_priors=lambda_priors,
        noise_stds=noise_stds,
        n_steps=N_STEPS,
        n_runs=N_RUNS,
        feature_dim=2,
    )

    all_rows = lin_rows + ts_rows
    print_sweep_table(all_rows, sort_by="final_regret_mean")

    # server_results = run_experiments(
    #         lambda: ServerRoutingBandit(complex_mode=False),
    #         n_steps=N_STEPS,
    #         n_runs=N_RUNS,
    #         feature_dim=2
    #     )
    #
    # plot_results(
    #     server_results,
    #     "Cumulative Rgeret - Multiple Runs"
    # )



