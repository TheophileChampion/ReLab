import math
from typing import Tuple


class ExponentialSchedule:
    """!
    A class implementing an exponential schedule.
    """

    def __init__(self, schedule: Tuple[float, float, float]) -> None:
        """!
        Create an exponential schedule.
        @param schedule: a tuple of the form (maximum_value, minimum_value, exponential_decay)
        """

        # @var maximum_value
        # The maximum/initial value taken by the scheduled value.
        self.maximum_value = schedule[0]

        # @var minimum_value
        # The minimum/asymptotic value taken by the scheduled value.
        self.minimum_value = schedule[1]

        # @var decay
        # The exponential decay rate (must be positive). Controls how quickly
        # the value decreases.
        self.decay = schedule[2]
        assert self.decay >= 0

    def __call__(self, current_step: int) -> float:
        """!
        Compute the current scheduled value at a given step.
        @param current_step: the step for which the scheduled value must be computed
        @return the current scheduled value
        """
        return self.minimum_value + (
            self.maximum_value - self.minimum_value
        ) * math.exp(-1.0 * current_step / self.decay)
