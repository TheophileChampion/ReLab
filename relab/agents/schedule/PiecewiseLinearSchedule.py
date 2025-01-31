from typing import Union, List, Tuple


class PiecewiseLinearSchedule:
    """!
    A class implementing a piecewise linear schedule.
    """

    def __init__(
        self, schedule: Union[List[Tuple[int, float]], Tuple[int, float]]
    ) -> None:
        """!
        Create the piecewise linear schedule.
        @param schedule: a list of tuples of the form (value, time_step)
        """

        # @var schedule
        # List of tuples defining the schedule breakpoints. Each tuple contains (time_step, value).
        # The schedule linearly interpolates between these points.
        self.schedule = schedule if isinstance(schedule, list) else [schedule]

    def __call__(self, current_step: int) -> float:
        """!
        Compute the current scheduled value at a given step.
        @param current_step: the step for which the scheduled value must be computed
        @return the current scheduled value
        """
        for i, (next_step, next_epsilon) in enumerate(self.schedule):
            if next_step > current_step:
                prev_step, prev_epsilon = self.schedule[i - 1]
                progress = (current_step - prev_step) / (next_step - prev_step)
                return progress * next_epsilon + (1 - progress) * prev_epsilon
        return self.schedule[-1][1]
