# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
from collections import UserList
from .visualize import plot_timeseries
from copy import deepcopy


class SimResult(UserList):
    """
    `SimResult` is a data structure for the results of a simulation, with time. It is returned from the `simulate_to*` methods for inputs, outputs, states, and event_states for the beginning and ending time step of the simulation, plus any save points indicated by the `savepts` and `save_freq` configuration arguments. The class includes methods for analyzing, manipulating, and visualizing the results of the simulation.

    Args:
            times (array[float]): Times for each data point where times[n] corresponds to data[n]
            data (array[Dict[str, float]]): Data points where data[n] corresponds to times[n]
    """

    __slots__ = ['times', 'data']  # Optimization 
    
    def __init__(self, times = [], data = []):
        self.times = deepcopy(times)
        self.data = deepcopy(data)

    def __eq__(self, other) -> bool:
        """Compare 2 SimResults

        Args:
            other (SimResult)

        Returns:
            bool: If the two SimResults are equal
        """
        return self.times == other.times and self.data == other.data

    def index(self, other : dict, *args, **kwargs) -> int:
        """
        Get the index of the first sample where other occurs

        Args: 
            other (dict)
    
        Returns:
            int: Index of first sample where other occurs
        """
        return self.data.index(other, *args, **kwargs)

    def extend(self, other) -> None:
        """
        Extend the SimResult with another SimResult

        Args:
            other (SimResult)

        """
        self.times.extend(other.times)
        self.data.extend(other.data)

    def pop(self, index : int = -1) -> dict:
        """Remove and return an element

        Args:
            index (int, optional): Index of element to be removed. Defaults to -1.

        Returns:
            dict: Element Removed
        """
        self.times.pop(index)
        return self.data.pop(index)
    
    def remove(self, index : int) -> None:
        """Remove an element

        Args:
            index (int): Index of element to be removed.
        """
        self.times.remove(index)
        self.data.remove(index)

    def clear(self) -> None:
        """Clear the SimResult"""
        self.times = []
        self.data = []

    def time(self, index : int) -> float:
        """Get time for data point at index `index`

        Args:
            index (int)

        Returns:
            float: Time for which the data point at index `index` corresponds
        """
        return self.times[index]

    def plot(self, **kwargs):
        """
        Plot the simresult as a line plot

        Args: 
            kwargs: Configuration parameters for plot_timeseries

        Returns:
            Figure
        """
        return plot_timeseries(self.times, self.data, options=kwargs)  

    def __not_implemented(self):  # lgtm [py/inheritance/signature-mismatch]
        raise NotImplementedError("Not Implemented")

    # Functions of list not implemented
    # Specified here to stop users from accidentally trying to use them (due to this classes similarity to list)
    append = __not_implemented 
    count = __not_implemented 
    insert = __not_implemented
    reverse = __not_implemented 
    # lgtm [py/missing-equals]


class LazySimResult(SimResult):  # lgtm [py/missing-equals]
    """
    Used to store the result of a simulation, which is only calculated on first request
    """
    def __init__(self, fcn, times = [], states = []):
        """
        Args:
            fcn (callable): function (x) -> z where x is the state and z is the data
            times (array(float)): Times for each data point where times[n] corresponds to data[n]
            data (array(dict)): Data points where data[n] corresponds to times[n]
        """
        self.fcn = fcn
        self.times = deepcopy(times)
        self.states = deepcopy(states)
        self.__data = None

    def is_cached(self):
        """
        Returns:
            bool: If the value has been calculated
        """
        return self.__data is not None

    def clear(self):
        self.times = []
        self.__data = None
        self.states = []

    def extend(self, other):
        self.times.extend(deepcopy(other.times))  # lgtm [py/modification-of-default-value]
        self.__data = None
        self.states.extend(deepcopy(other.states))  # lgtm [py/modification-of-default-value]

    def pop(self, index = -1):
        """Remove an element. If data hasn't been cached, remove the state - so it wont be calculated

        Args:
            index (int, optional): Index of element to be removed. Defaults to -1.

        Returns:
            dict: Element Removed
        """
        self.times.pop(index)
        x = self.states.pop(index)
        if self.__data is not None:
            return self.__data.pop(index)
        return self.fcn(x)

    @property
    def data(self):
        """
        Get the data (elements of list). Only calculated on first request

        Returns:
            array(dict): data
        """
        if self.__data is None:
            self.__data = [self.fcn(x) for x in self.states]
        return self.__data
    