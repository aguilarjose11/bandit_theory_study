from typing import List, Dict
import scipy
import copy
import numpy as np


class ArmedBanditAlgorithm:
    """Bandit Algorithm Template for Inference and Training

    The class template will ensure the expectations from the
    k-armed testbed.
    """
    def __call__(self, *args, **kwargs) -> int:
        return self.choose()

    def learn(self,
              arm: int,
              reward: float) -> None:
        pass

    def choose(self) -> int:
        pass


class GreedyBandit(ArmedBanditAlgorithm):
    def __init__(self,
                 k_arms: int):
        self.k_arms = k_arms
        self.k_pulls = [0 for arm in range(k_arms)]
        self.value = np.array([0. for arm in range(k_arms)])

    def learn(self,
              arm: int,
              reward: float) -> None:
        """ Update algorithm's values

        The algorithm updates its value over the arms by performing incremental averaging.
        """

        self.k_pulls[arm] += 1
        # TODO: Double check the math
        self.value[arm] = (self.value[arm]
                           + (np.array(reward) - self.value[arm]) / self.k_pulls[arm])

    def choose(self) -> int:
        """ Choose an arm based on the given policy

        The policy that the greedy bandit algorithm follows is to select the arm with the
        largest value found so far (take the argmax.)
        """
        max_inds, = np.where(self.value == self.value.max())
        return int(np.random.choice(max_inds))


class KArmedBanditTestbed:
    def __init__(self,
                 k_arms: int,
                 time_steps: int,
                 episodes: int,
                 algorithms: Dict[str, ArmedBanditAlgorithm]):
        """K-Armed Bandit Testbed

        parameters
        ----------
        k_arms: int
            Number of arms that the bandit will have
        time_steps: int
            Number of timesteps that an epoch lasts
        episodes: int
            Number of episodes to train
        algorithms: Dict[str, ArmedBanditAlgorithm]
            The initial algorithm objects to used for training. During each
            episode, a new dictionary is created, containing copies of the
            algorithm objects. These copies will then be trained, and their
            performance is what will be returned. For correct functioning,
            the number of arms expected by the algorithms must match the
            k_arms of the bandit testbed.
        """
        self.algorithms = algorithms
        self.k_arms = k_arms
        self.episodes = episodes
        self.time_steps = time_steps

    # TODO: Consider tracking the optimal action taken.
    def run(self,
            episodes: int = None,
            time_steps: int = None) -> Dict[str, List[List[float]]]:
        # Allow changing the number of episodes and time_steps before starting
        episodes = episodes if episodes is not None else self.episodes
        time_steps = time_steps if time_steps is not None else self.time_steps

        rewards_over_episodes_per_step = {
            algo_name: [] for algo_name in self.algorithms
        }

        for episode in range(episodes):
            # DeepCopy algorithms and do learning
            algorithms: Dict[str, ArmedBanditAlgorithm] = copy.deepcopy(self.algorithms)
            # Add new list to rewards_over_episodes_per_step
            for value in rewards_over_episodes_per_step.values():
                value.append([])
            # Choose random means for arms
            means = scipy.stats.norm(0, scale=1).rvs(size=self.k_arms)
            variances = [1 for _ in range(self.k_arms)]
            arms = [
                scipy.stats.norm(loc=mean, scale=variance) for (mean, variance) in zip(means, variances)
            ]

            # Perform testbed experiments
            for step in range(time_steps):
                for name, algo in algorithms.items():
                    selection = algo()
                    reward = arms[selection].rvs(size=1)
                    algo.learn(arm=selection,
                               reward=reward)
                    # Add reward collected
                    rewards_over_episodes_per_step[name][episode].append(float(reward))

        return rewards_over_episodes_per_step



