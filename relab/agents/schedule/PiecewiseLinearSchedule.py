class PiecewiseLinearSchedule:
    """!
    A class implementing a piecewise linear schedule.
    """

    def __init__(self, schedule):
        """!
        Create the piecewise linear schedule.
        @param schedule: a list of tuples of the form (value, time_step)
        """
        
        ## @var schedule
        # List of tuples defining the schedule breakpoints. Each tuple contains (time_step, value).
        # The schedule linearly interpolates between these points.
        self.schedule = schedule if isinstance(schedule, list) else [schedule]

    def __call__(self, current_step):
        """!
        Compute the current scheduled value at a given step.
        @param current_step: the step for which the scheduled value must be computed
        @return the current scheduled value
        """
        for i, (next_step, next_epsilon) in enumerate(self.schedule):
            if next_step > current_step:
                previous_step, previous_epsilon = self.schedule[i - 1]
                progress = (current_step - previous_step) / (next_step - previous_step)
                return progress * next_epsilon + (1 - progress) * previous_epsilon
        return self.schedule[-1][1]
