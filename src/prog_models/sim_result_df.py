# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from collections import UserList, defaultdict
from copy import deepcopy
from matplotlib.pyplot import figure
import numpy as np
from typing import Callable, Dict, List
import pandas as pd

from .utils.containers import DictLikeMatrixWrapper
from .visualize import plot_timeseries


class SimResultDF(UserList):
    """
    SimResultDF` is a Class creating pandas dataframes shortcuts for the results of a simulation, with time.
    It is returned from the `simulate_to*` methods for :term:`inputs<input>`, :term:`outputs<output>`, :term:`states<state>`, and :term:`event_states<event state>` for the beginning and ending time step of the simulation, plus any save points indicated by the `savepts` and `save_freq` configuration arguments. The class includes methods for analyzing, manipulating, and visualizing the results of the simulation.

    Args:
            times (array[float]): Times for each data point where times[n] corresponds to data[n]
            data (array[Dict[str, float]]): Data points where data[n] corresponds to times[n]
    """

    __slots__ = ['times', 'data']  # Optimization

    def __init__(self, times: list, data: list, _copy=True):
        """initializes

                Args:
                    times (a list of timestamps)
                    data (a list of data, it corresponds to the timestamps)

                (theory) if times or data has nothing in the list then the error is that there is no output.
                    then self variables are set to empty data frames

                Else: the data frames are set depending on whether the data is a copy.
                    index_ul: a list of integers is creates for the dataframes vertical indexing
                    A dataframe for self.times_df is initialized with the column label of 'times' since the array of timestamps doesn't have a label.
                    temp_df_list: a list for the dataframes created from all the dictionaries in data

                    list comprehension is used to create a list of dataframes for each dictionary array of values

                    If it is a copy, deepcopy is used.
                    Else: data is sufficient.
                the final dataframe, self.data_df contains all the data, the first column being the timestamps.
                """

        if times is None or data is None:   # in case the error is that there is no output (theory)
            self.data_df = pd.DataFrame()
        else:
            temp_df_list = []
            if _copy:
                [temp_df_list.append(pd.DataFrame(x, index=times.copy())) for x in deepcopy(data)]
            else:
                [temp_df_list.append(pd.DataFrame(x, index=times.copy())) for x in data]
            self.data_df = pd.concat(temp_df_list, axis=1)

    def __eq__(self, other: "SimResultDF") -> bool:
        """Compare 2 SimResultDFs

                Args:
                    other (SimResultDF)

                Returns:
                    bool: If the two SimResultDFs are equal
                """
        compare_dfs_bool = self.data_df.equals(other.data_df)
        return compare_dfs_bool

    #def index(self, other: dict, *args, **kwargs) -> int:   # UNFINISHED!!!! ask Chris about type of return
        """
        Get the location index of the first sample where other occurs

        Args:
            other (dict)

        Returns:
            int: location Index of first sample where other occurs
        """
        #return self.data.index(other, *args, **kwargs)

    def extend(self, other : "SimResultDF") -> None:
        """
        Extend the SimResult with another SimResult or LazySimResult object

        Args:
            other (SimResult/LazySimResult)

        """
        if isinstance(other, SimResultDF) and isinstance(other.data_df, pd.DataFrame):
            self.data_df = pd.concat([self.data_df, other.data_df], axis=0)
        else:
            raise ValueError(f"ValueError: Argument must be of type {self.__class__} with a {self.data_df.__class__}")

    def dfpop(self, label: str, index: int = -1) -> dict:
        """Remove and return an element

        Args:
            label (string): label for column in dataframe
            index (int, optional): Index(timestamp) of element to be removed. Defaults to last item.

        Returns:
            dict: Element Removed
        """
        if index == -1:
            index = self.data_df.index[-1]
        ret_dict = {label: self.data_df.at[index, label]}
        self.data_df.at[index, label] = -0.0
        return ret_dict
