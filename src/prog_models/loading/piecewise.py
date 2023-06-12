# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

class Piecewise():
    """
    This is a simple piecewise future loading class. It takes a list of times and values and returns the value that corresponds to the current time. The object replaces the future_loading_eqn for simulate_to and simulate_to_threshold.

    Args
    -----
        InputContainer : class
            The InputContainer class for the model (model.InputContainer)
        times : list
            A list of times (s)
        values : list
            A list of values

    Example
    -------
    >>> from prog_models.loading import Piecewise
    >>> m = SomeModel()
    >>> future_load = Piecewise(m.InputContainer, [0, 10, 20], [0, 1, 0])
    >>> m.simulate_to_threshold(future_load)
    """
    def __init__(self, InputContainer, times, values):
        self.InputContainer = InputContainer
        self.times = times
        self.values = values

    def __call__(self, t, x=None):
        """
        Return the value that corresponds to the current time

        Args:
            t (float): Time (s)
            x (StateContaienr, optional): Current state. Defaults to None.

        Returns:
            InputContainer: The value that corresponds to the current time
        """
        return self.InputContainer({
            key: next(self.values[key][i] for i in range(len(self.times)) if self.times[i] > t)
            for key in self.values})
