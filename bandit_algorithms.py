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


class EpsilonGreedyBandit(ArmedBanditAlgorithm):
    def __init__(self,
                 k_arms: int,
                 epsilon: float):
        assert 0 <= epsilon <= 1, f"Epsilon ({epsilon}) parameter is not a probability!"

        self.k_arms = k_arms
        self.epsilon = epsilon
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

        In the case of epsilon greedy, there is an epsilon chance that one of the 
        unexplored, non-optimal (w.r.t. value) arms will be chosen. Note that when
        'exploring,' the maximum value arm is still being considered, which gives
        it a probability of (1-e) + e*(1/k), which is larger than 1-e!
        """
        explore = np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon])
        if explore:
            # Choose any arm regardless of value (so max value may be included)
            arm = int(np.random.choice(self.k_arms))
        else:
            max_indices, = np.where(self.value == self.value.max())
            # Randomly choice an arm if there is a tie.
            arm = int(np.random.choice(max_indices))
        return arm


class UCBBandit(ArmedBanditAlgorithm):
    def __init__(self,
                 k_arms: int,
                 confidence: int):
        self.k_arms = k_arms
        self.k_pulls = np.array([0 for arm in range(k_arms)])
        self.value = np.array([0. for arm in range(k_arms)])
        self.c = confidence
        self.t = 0

    def learn(self,
              arm: int,
              reward: float) -> None:
        """ Update algorithm's values

        The algorithm updates its value over the arms by performing incremental averaging.
        """

        self.k_pulls[arm] += 1
        # Incremental Sample Average Value 
        self.value[arm] = (self.value[arm]
                           + (np.array(reward) - self.value[arm]) / self.k_pulls[arm])

    def choose(self) -> int:
        """ Choose an arm based on the given policy

        The policy that the greedy bandit algorithm follows is to select the arm with the
        largest value found so far (take the argmax.)
        """
        self.t += 1
        var_approx = np.log(self.t) / self.k_pulls
        upper_bound = self.value + self.c * np.sqrt(var_approx)
        max_inds, = np.where(upper_bound == upper_bound.max())
        return int(np.random.choice(max_inds))
