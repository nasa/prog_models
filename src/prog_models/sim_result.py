# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from collections import UserList, defaultdict
from copy import deepcopy
from matplotlib.pyplot import figure
import numpy as np
from typing import Callable, Dict, List

from .utils.containers import DictLikeMatrixWrapper
from .visualize import plot_timeseries


class SimResult(UserList):
    """
    `SimResult` is a data structure for the results of a simulation, with time. It is returned from the `simulate_to*` methods for :term:`inputs<input>`, :term:`outputs<output>`, :term:`states<state>`, and :term:`event_states<event state>` for the beginning and ending time step of the simulation, plus any save points indicated by the `savepts` and `save_freq` configuration arguments. The class includes methods for analyzing, manipulating, and visualizing the results of the simulation.

    Args:
            times (array[float]): Times for each data point where times[n] corresponds to data[n]
            data (array[Dict[str, float]]): Data points where data[n] corresponds to times[n]
    """

    __slots__ = ['times', 'data']  # Optimization 
    
    def __init__(self, times : list = None, data : list = None, _copy = True):
        if times is None or data is None:
            self.times = [] 
            self.data = []
        else:
            self.times = times.copy()
            if _copy:
                self.data = deepcopy(data)
            else:
                self.data = data

    def __eq__(self, other : "SimResult") -> bool:
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

    def extend(self, other : "SimResult") -> None:
        """
        Extend the SimResult with another SimResult or LazySimResult object

        Args:
            other (SimResult/LazySimResult)

        """
        if isinstance(other, SimResult):
            self.times.extend(other.times)
            self.data.extend(other.data)
        else:
            raise ValueError(f"ValueError: Argument must be of type {self.__class__}")

    def pop(self, index : int = -1) -> dict:
        """Remove and return an element

        Args:
            index (int, optional): Index of element to be removed. Defaults to -1.

        Returns:
            dict: Element Removed
        """
        self.times.pop(index)
        return self.data.pop(index)

    def remove(self, d : float = None, t : float = None) -> None:
        """Remove an element

        Args:
            d: Data value to be removed.
            t: Time value to be removed.
        """
        if sum([i is None for i in (d, t)]) != 1:
            raise ValueError("ValueError: Only one named argument (d, t) can be specified.")

        if (t is not None):
            self.data.pop(self.times.index(t))
            self.times.remove(t)
        else:
            self.times.pop(self.data.index(d))
            self.data.remove(d)

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

    def to_numpy(self, keys = None) -> np.ndarray:
        """
        Convert from simresult to numpy array

        Args:
            keys: Subset of keys to return as part of numpy array (by default, all)

        Returns:
            np.ndarray: numpy array representing simresult
        """
        if len(self.data) == 0:
            return np.array([[]], dtype=np.float64)
        if len(self.data[0]) == 0:
            return np.array([[] for _ in self.data], dtype=np.float64)
        if isinstance(self.data[0], DictLikeMatrixWrapper) and keys is None:
            return np.array([u_i.matrix[:, 0] for u_i in self.data], dtype=np.float64)
        if keys is None:
            keys = self.data[0].keys()
        return np.array([[u_i[key] for key in keys] for u_i in self.data], dtype=np.float64)

    def plot(self, **kwargs) -> figure:
        """
        Plot the simresult as a line plot

        Keyword Args:
            keys (list[str]): list of keys to plot. If not provided, all keys in the series are plotted.
            figsize (tuple[float, float]): width and height of the figure
            compact (bool): If true, all timeseries are displayed in one plot (multiple colored lines)
            xlabel (str) : label for the x-axis. Default is 'time'
            ylabel (str) : label for the y-axis. Default is 'state'
            title (str) : plot title. Default is no title
            title_fontsize (str or float): plot title fontsize. Default is 'x-large'
            suptitle (str) : plot suptitle. Default is no suptitle
            ticklabel_fontsize (str or float): tick label font sizes. Default is 'small'
            tight_layout (bool): whether to use tight layout (minimize figure blank space around the graph)
            display_labels (str): whether to display x and y-labels in the figure (['no', 'minimal', 'all'])

        Returns:
            Figure
        """
        return plot_timeseries(self.times, self.data, legend = {'display': True}, options=kwargs)

    def monotonicity(self) -> Dict[str, float]:
        """
        Calculate monotonicty for a single prediction.
        Given a single simulation result, for each event: go through all predicted states and compare those to the next one.
        Calculates monotonicity for each event key using its associated mean value in UncertainData.

        Where N is number of measurements and sign indicates sign of calculation.

        Coble, J., et. al. (2021). Identifying Optimal Prognostic Parameters from Data: A Genetic Algorithms Approach. Annual Conference of the PHM Society.
        http://www.papers.phmsociety.org/index.php/phmconf/article/view/1404
        Baptistia, M., et. al. (2022). Relation between prognostics predictor evaluation metrics and local interpretability SHAP values. Aritifical Intelligence, Volume 306.
        https://www.sciencedirect.com/science/article/pii/S0004370222000078

        Args:
            None

        Returns:
            float: Value between [0, 1] indicating monotonicity of a given event for the Prediction.
        """
        # Collect and organize mean values for each event
        by_event = defaultdict(list)
        for uncertaindata in self.data:
            for key, value in uncertaindata.items():
                by_event[key].append(value)

        # For each event, calculate monotonicity using formula
        result = {}
        for key, l in by_event.items():
            mono_sum = 0
            for i in range(len(l)-1): 
                mono_sum += np.sign(l[i+1] - l[i])
            result[key] = abs(mono_sum / (len(l)-1))
        return result

    def __not_implemented(self):  # lgtm [py/inheritance/signature-mismatch]
        raise NotImplementedError("Not Implemented")

    # Functions of list not implemented
    # Specified here to stop users from accidentally trying to use them 
    # (due to this classes similarity to list)
    append = __not_implemented
    count = __not_implemented
    insert = __not_implemented
    reverse = __not_implemented
    # lgtm [py/missing-equals]


class LazySimResult(SimResult):  # lgtm [py/missing-equals]
    """
    Used to store the result of a simulation, which is only calculated on first request
    """
    def __init__(self, fcn : Callable, times : list = None, states : list = None, _copy = True) -> None:
        """
        Args:
            fcn (callable): function (x) -> z where x is the state and z is the data
            times (array(float)): Times for each data point where times[n] corresponds to data[n]
            data (array(dict)): Data points where data[n] corresponds to times[n]
        """
        self.fcn = fcn
        self.__data = None
        if times is None or states is None:
            self.times = []
            self.states = []
        else:
            self.times = times.copy()
            if _copy:
                self.states = deepcopy(states)
            else:
                self.states = states

    def __reduce__(self):
        return (self.__class__.__base__, (self.times, self.data))

    def is_cached(self) -> bool:
        """
        Returns:
            bool: If the value has been calculated
        """
        return self.__data is not None

    def clear(self) -> None:
        """
        Clears the times, states, and data cache for a LazySimResult object
        """
        self.times = []
        self.__data = None
        self.states = []

    def extend(self, other : "LazySimResult", _copy=True) -> None:
        """
        Extend the LazySimResult with another LazySimResult object
        Raise ValueError if SimResult is passed
        Function fcn of other LazySimResult MUST match function fcn of LazySimResult object to be extended

        Args:
            other (LazySimResult)

        """
        if (isinstance(other, self.__class__)):
            self.times.extend(other.times)
            if _copy:
                self.states.extend(deepcopy(other.states))
            else:
                self.states.extend(other.states)
            if self.__data is None or not other.is_cached():
                self.__data = None
            else:
                self.__data.extend(other.data)
        elif (isinstance(other, SimResult)):
            raise ValueError(f"ValueError: {self.__class__} cannot be extended by SimResult. First convert to SimResult using to_simresult() method.")
        else:
            raise ValueError(f"ValueError: Argument must be of type {self.__class__}.")

    def pop(self, index : int = -1) -> dict:
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

    def remove(self, d : float = None, t : float = None, s = None) -> None:
        """Remove an element

        Args:
            d: Data value to be removed
            t: Time value to be removed
            s: State value to be removed
        """
        if sum([i is None for i in (d, t, s)]) != 2:
            raise ValueError("ValueError: Only one named argument (d, t, s) can be specified.")

        if (t is not None):
            target_index = self.times.index(t)
            self.times.pop(target_index)
            self.states.pop(target_index)
            if self.__data is not None:
                self.__data.pop(target_index)
        elif (s is not None):
            target_index = self.states.index(s)
            self.times.pop(target_index)
            self.states.pop(target_index)
            if self.__data is not None:
                self.__data.pop(target_index)
        else:
            target_index = self.data.index(d)
            self.times.pop(target_index)
            self.states.pop(target_index)
            self.__data.pop(target_index)

    def to_simresult(self) -> SimResult:
        return SimResult(self.times, self.data)

    @property
    def data(self) -> List[dict]:
        """
        Get the data (elements of list). Only calculated on first request

        Returns:
            array(dict): data
        """
        if self.__data is None:
            self.__data = [self.fcn(x) for x in self.states]
        return self.__data
