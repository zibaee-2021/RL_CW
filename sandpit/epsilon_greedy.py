class EpsilonGreedy(object):
  """An epsilon-greedy agent.

  This agent returns an action between 0 and 'number_of_arms'; with probability
  `(1-epsilon)` it chooses the action with the highest estimated value, while
  with probability `epsilon` it samples an action uniformly at random.
  """

  def __init__(self, name, number_of_arms, epsilon=0.1):
    self.name = name
    self._epsilon = epsilon
    self._num_of_arms = number_of_arms
    self._current_timestep, self._Qta, self._Nta = None, None, None
    self.reset()

  def step(self, previous_action, reward):
    """Update the learnt statistics and return an action.

    A single call to step uses the provided reward to update the value of the
    taken action (which is also provided as an input), and returns an action.
    The action is either uniformly random (with probability epsilon), or greedy
    (with probability 1 - epsilon).

    If the input action is None (typically on the first call to step), then no
    statistics are updated, but an action is still returned.
    """
    self._current_timestep += 1

    # 1. UPDATE:

    if previous_action is None:
      pass  # no update

    else:  # update action-value-estimate of previous_action towards the observed reward for having taken that previous action
           # (via updating step-size, via updating previous action count).
        self._Nta[previous_action] += 1
        alpha = 1 / self._Nta[previous_action]
        self._Qta[previous_action] += alpha * (reward - self._Qta[previous_action])

    # 2. SELECT NEXT ACTION:
    if callable(self._epsilon):
        eps = self._epsilon(self._current_timestep)
    else:
        eps = self._epsilon

    random = np.random.rand()  # random number in [0, 1)

    if random <= eps:  # EXPLORE
        next_action = np.random.choice(self._num_of_arms)

    else:  # EXPLOIT
        best_actions = np.nonzero(self._Qta == np.max(self._Qta))[0]
        next_action = np.random.choice(best_actions)
    return next_action

  def reset(self):
    self._Qta = np.zeros(self._num_of_arms)
    self._Nta = np.zeros(self._num_of_arms)
    self._current_timestep = 0