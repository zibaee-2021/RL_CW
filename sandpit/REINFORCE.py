class REINFORCE(object):
    def __init__(self, name, number_of_arms, step_size=0.1, baseline=False):
        self.name = name
        self._num_of_arms = number_of_arms
        self._alpha = step_size
        self._use_baseline = baseline
        self._baseline_avg_reward = None  # to reduce variance (This should not depend on actions.)
        self._policy = None  # the objective is to optimise this to give highest probs for best actions (actions that give most reward)
        self._action_prefs = None  # the parameters that get updated
        self.reset()

    def step(self, previous_action, reward):
        r_hat = 0
        # 1. UPDATE:
        if previous_action is None:  # no update
            pass
        else:
            self._policy[previous_action] = self._action_prefs[previous_action] ** 2 / np.sum(self._action_prefs ** 2)  # square-max policy parameterisation, so policy always sums to 1
            if self._use_baseline:
                self._baseline_avg_reward += self._alpha * (reward - self._baseline_avg_reward)  # update rule (2.3)
                r_hat = self._baseline_avg_reward
        step_reward = self._alpha * (reward - r_hat)
        # update action-preferences with gradient of policy (simplified by log-likelihood trick and chain rule)
        for action in self._num_of_arms:
            if action == previous_action:
                 self._action_prefs[previous_action] += step_reward * (1 - self._policy[previous_action])
            else:
                 self._action_prefs[action] -= step_reward * self._policy[action]
            self._policy[previous_action] = self._action_prefs[previous_action] ** 2 / np.sum(self._action_prefs ** 2)

            assert np.sum(self._policy) == 1
            next_action = np.random.choice(a=self._num_of_arms, p=self._policy)
        return next_action

    def reset(self):
        self._baseline_avg_reward = 0
        self._policy = np.zeros(self._num_of_arms)
        self._action_prefs = np.zeros(self._num_of_arms)
