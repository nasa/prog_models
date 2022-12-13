# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np


class MovingAverage():
    """
    This is a simple moving average future loading class. It takes input values and stores them in a window. When called, it returns the average of the values in the window. The object replaces the future_loading_eqn for simulate_to and simulate_to_threshold.

    Parameters
    ----------
    InputContainer : class
        The InputContainer class for the model (model.InputContainer)
    window : int, optional
        The size of the window to use for the moving average, by default 10
    """
    def __init__(self, InputContainer, window = 10):
        self.window = window
        self.InputContainer = InputContainer
        self.values = {}
        self.index = 0

    def add_load(self, input):
        """
        Add a load to the moving average

        Args:
            input (InputContainer): One load value to add to the moving average
        """
        for key, value in input.items():
            if key not in self.values:
                self.values[key] = np.zeros(self.window)
            self.values[key][self.index] = value
        self.index = (self.index + 1) % self.window

    def __call__(self, t, x = None):
        """
        Return the average of the values in the window

        Args:
            t (float): Time (s)
            x (StateContaienr, optional): Current state. Defaults to None.

        Returns:
            InputContainer: The average of the values in the window
        """
        return self.InputContainer({key: np.mean(self.values[key]) for key in self.values})


class GuassianNoiseLoadWrapper():
    """
    This is a simple wrapper for future loading functions that adds gaussian noise to the inputs. It takes a future loading function and a standard deviation and returns a new future loading function that adds gaussian noise to the inputs.

    Parameters
    ----------
    fcn : function
        The future loading function to wrap
    std : float
        The standard deviation of the gaussian noise to add
    """
    def __init__(self, fcn, std):
        self.fcn = fcn
        self.std = std

    def __call__(self, t, x = None):
        """
        Return the load with noise added

        Args:
            t (float): Time (s)
            x (StateContaienr, optional): Current state. Defaults to None.

        Returns:
            InputContainer: The load with noise added
        """
        input = self.fcn(t, x)
        for key, value in input.items():
            input[key] = np.random.normal(value, self.std)
        return input
