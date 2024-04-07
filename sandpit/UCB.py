class UCB(object):
  def __init__(self, name, number_of_arms, bonus_multiplier):
    self._number_of_arms = number_of_arms  # number of actions available to the agent
    self._bonus_multiplier = bonus_multiplier  # parameter c from slides. (c=0 makes algo greedy).
    self.name = name
    self._Nta = None  #  N_t(a) The number of times each action taken
    self._Qta = None  #  Q_t(a) The action-value-estimate
    self._current_timestep = None # t
    self.reset()

  def step(self, previous_action, reward):
    self._current_timestep += 1

    # 1. UPDATE:

    if previous_action is None:
      pass  # no update, cos no previous action

    else:  # update action-value-estimate of previous_action towards the observed reward for having taken that previous action
           # (via updating step-size, via updating previous action count).
      self._Nta[previous_action] += 1
      alpha = 1 / self._Nta[previous_action]
      self._Qta[previous_action] = self._Qta[previous_action] + alpha * (reward - self._Qta[previous_action])

    # 2. SELECT NEXT ACTION:

    if (self._Nta == 0).any: # try something new!
        zeros = np.where(self._Nta == 0)[0]
        if zeros.size != 0:
            action_not_yet_taken = np.random.choice(zeros)
            return action_not_yet_taken
    else:
        tiny_number = np.finfo(np.float64).eps
        self._Nta = self._Nta + tiny_number # to prevent divide-by-zero
        u = np.log(self._current_timestep) / self._Nta  # uncertainty
        Uta = self._bonus_multiplier * np.sqrt(u)
        sum_of_exploit_and_explore = self._Qta + Uta
        # if more than one has exact same value, so tie-breaking needed:
        max_value = sum_of_exploit_and_explore.max()
        indices_of_all_with_max = np.flatnonzero(sum_of_exploit_and_explore == max_value)
        next_action = np.random.choice(indices_of_all_with_max)
        return next_action

  def reset(self):
    self._Nta = np.zeros(number_of_arms)  #  N_t(a) is number of times each action taken
    self._Qta = np.zeros(number_of_arms)  #  Q_t(a)
    self._current_timestep = 0 # t