import math
from typing import Tuple


class ExponentialSchedule:
    """!
    A class implementing an exponential schedule.
    """

    def __init__(self, schedule: Tuple[float, float]) -> None:
        """!
        Create an exponential schedule.
        @param schedule: a tuple of the form (maximum_value, exponential_decay)
        """

        # @var maximum_value
        # The maximum/initial value taken by the scheduled value.
        self.maximum_value = schedule[0]

        # @var decay
        # The exponential decay rate (must be negative). Controls how quickly
        # the value decreases.
        self.decay = schedule[1]
        assert self.decay <= 0

    def __call__(self, current_step: int) -> float:
        """!
        Compute the current scheduled value at a given step.
        @param current_step: the step for which the scheduled value must be computed
        @return the current scheduled value
        """
        return max(self.maximum_value, math.exp(self.decay * current_step))
