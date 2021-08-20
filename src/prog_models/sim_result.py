from collections import UserList
from .visualize import plot_timeseries


class SimResult(UserList):
    __slots__ = ['times', 'data']  # Optimization 

    """
    Used to store the result of a simulation, with time 
    """
    def __init__(self, times, data):
        """
        Args:
            times (array(float)): Times for each data point where times[n] corresponds to data[n]
            data (array(dict)): Data points where data[n] corresponds to times[n]
        """
        self.times = times
        self.data = data

    def __eq__(self, other):
        """Compare 2 SimResults

        Args:
            other (SimResult)

        Returns:
            bool: If the two SimResults are equal
        """
        return self.times == other.times and self.data == other.data

    def time(self, index):
        """Get time for data point at index `index`

        Args:
            index (int)

        Returns:
            float: Time for which the data point at index `index` corresponds
        """
        return self.times[index]

    def plot(self, **kwargs):
        plot_timeseries(self.times, self.data, options=kwargs)  
    # lgtm [py/missing-equals]

class LazySimResult(SimResult):  # lgtm [py/missing-equals]
    """
    Used to store the result of a simulation, which is only calculated on first request
    """
    def __init__(self, fcn, times, states):
        """
        Args:
            fcn (callable): function (x) -> z where x is the state and z is the data
            times (array(float)): Times for each data point where times[n] corresponds to data[n]
            data (array(dict)): Data points where data[n] corresponds to times[n]
        """
        self.fcn = fcn
        self.times = times
        self.states = states
        self.__data = None

    def is_cached(self):
        """
        Returns:
            bool: If the value has been calculated
        """
        return self.__data is not None

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
    