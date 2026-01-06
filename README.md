# ResearchAssignmentMAB
Repository for the simulations of UCB and TS in both traditional and contextual settings.

Four files are present, each for a different simulation topic and goal:

(1) bandits.py:
 contains Bernoulli bandits simulations for 3 scenarios of increasing complexity,

(2) gaussianBandits.py:
 contains Gaussian bandits simulations for 3 scenarios of increasing complexity,

(3) energyDis.py:
 contains the simulation of an energy dispatch network scenario, modelled as Gaussian bandit,

(4) Contextual.py:
 contains the simulation of a server routing scenario, modelled as a contextual bandit.

---
Here the discussion of the simulation findings, as taken from the report, is included for completeness:

"Across all experiments, algorithmic performance is primarily governed by an effective signal-to-noise ratio: large sub-optimality gaps and moderate noise allow for rapid identification and exploitation, whereas small gaps, many competing arms, or heterogeneous uncertainty require prolonged exploration and lead to higher regret over the finite horizons considered.

In the stationary Bernoulli and Gaussian benchmarks, Thompson Sampling typically achieves lower regret than UCB in easy and moderately difficult settings, consistent with its probability-matching behavior and with empirical observations reported in practical applications. The non-stationary experiments highlight a limitation of classical Thompson Sampling: posterior concentration induces inertia after a distributional shift, causing delayed adaptation and regret spikes. This observation is critical for future research: it motivates the development of an adaptive Hierarchical Bayesian framework. By modeling parameters hierarchically and incorporating dynamic update mechanisms, the system can support continuous online parameter identification, allowing the execution policy to adapt to evolving market regimes, a necessary feature for financial environments that violate the stationarity assumption.

The energy dispatch scenario illustrates how specific environmental structures can influence this algorithm ranking. While Thompson Sampling usually excels in stochastic settings, UCB demonstrated superior performance in the specific case of clear optimality with low noise: here, its deterministic optimism allowed for immediate commitment, avoiding the residual sampling variance of Bayesian approaches. However, under closely competing options with heterogeneous noise, Thompson Sampling was more robust, yielding smoother learning dynamics and lower average regret. This scenario serves as a structural proxy for the trade execution problem: as energy dispatch minimizes generation costs under fluctuating demand, the execution problem aims to minimize implementation shortfall under fluctuating liquidity. The success of TS in this complex regime validates its selection for broker routing, where separating signal from noise among closely competing brokers is the main challenge.

Finally, the server routing simulations emphasize the complexity introduced by continuous state spaces, where the agent must learn decision boundaries rather than independent mean rewards. In simple routing scenarios with distinct optimal regions, both LinUCB and Linear Thompson Sampling converge rapidly. However, the complex environment reveals a critical divergence in exploration strategies. LinUCB’s deterministic confidence ellipsoids can shrink too rigidly around suboptimal arms, locking out the optimal action in regions with narrow reward gaps. In contrast, Linear TS’s randomized posterior sampling maintains a small exploration noise, allowing it to refine complex decision boundaries more effectively. This results in superior accuracy and lower cumulative regret, highlighting the robustness of Bayesian methods in navigating high-dimensional and intersecting reward landscapes. This simulation problem is quite similar to contextual broker selection, where the server load maps to market context: the ability of contextual TS to resolve ambiguous decision boundaries confirms its suitability for learning state-dependent trading policies in a continuous market environment.

Furthermore, due to intersecting performance lines and continuous contexts, perfect action selection is generally unattainable in finite time, thus the relevant notion of convergence is a decreasing average regret and increasing consistency in the learned decision boundaries: no flattening can be observed when analyzing the cumulative pseudo-regret curves."

