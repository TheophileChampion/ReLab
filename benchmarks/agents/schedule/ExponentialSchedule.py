import math


class ExponentialSchedule:
    """
    A class implementing an exponential schedule.
    """

    def __init__(self, schedule):
        """
        Create an exponential schedule.
        :param schedule: a tuple of the form (maximum_value, exponential_decay)
        """
        self.maximum_value = schedule[0]
        self.decay = schedule[1]
        assert self.decay <= 0

    def __call__(self, current_step):
        """
        Compute the current scheduled value at a given step.
        :param current_step: the step for which the scheduled value must be computed
        :return: the current scheduled value
        """
        return max(self.maximum_value, math.exp(self.decay * current_step))
