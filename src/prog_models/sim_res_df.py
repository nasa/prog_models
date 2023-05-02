# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from collections import UserList, defaultdict
from copy import deepcopy
from matplotlib.pyplot import figure
import numpy as np
from typing import Callable, Dict, List, Union
import pandas as pd

from prog_models.utils.containers import DictLikeMatrixWrapper
from prog_models.visualize import plot_timeseries


class SimResultDF(UserList):
    """
    SimResultDF` is a Class creating pandas dataframes shortcuts for the results of a simulation, with time.
    It is returned from the `simulate_to*` methods for :term:`inputs<input>`, :term:`outputs<output>`, :term:`states<state>`, and :term:`event_states<event state>` for the beginning and ending time step of the simulation, plus any save points indicated by the `savepts` and `save_freq` configuration arguments. The class includes methods for analyzing, manipulating, and visualizing the results of the simulation.

    Args:
            times (array[float]): Times for each data point where times[n] corresponds to data[n]
            data (array[Dict[str, float]]): Data points where data[n] corresponds to times[n]
    """

    # __slots__ = ['times', 'data']  # Optimization

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

        if times is None or data is None:  # in case the error is that there is no output (theory)
            self.data_df = pd.DataFrame()
        else:
            print(type(data))
            print(data)

    def __eq__(self, other: "SimResultDF") -> bool:
        """Compare 2 SimResultDFs

                Args:
                    other (SimResultDF)

                Returns:
                    bool: If the two SimResultDFs are equal
                """
        compare_dfs_bool = self.data_df.equals(other.data_df)
        return compare_dfs_bool

    # def index(self, other: dict, *args, **kwargs) -> int:   # UNFINISHED!!!! ask Chris about type of return

    def extend(self, other: Union["SimResultDF", "LazySimResultDF"]) -> None:
        """
        Extend the SimResult with another SimResult or LazySimResultDF object

        Args:
            other (SimResultDF/LazySimResultDF)

        """
        if isinstance(other, SimResultDF) and isinstance(other.data_df, pd.DataFrame):
            self.data_df = pd.concat([self.data_df, other.data_df], ignore_index=True, axis=0)
            self.data_df.reindex()
        else:
            raise ValueError(f"ValueError: Argument must be of type {self.__class__} with a {self.data_df.__class__}")

    def dfpop(self, d: dict = None, t: float = None, index: int = -1) -> pd.Series:

        if d is not None:
            for i in d:
                for x in d[i]:
                    if self.data_df[i].where(self.data_df[i] == x).count() != 1:
                        raise ValueError("ValueError: Only one named argument (d, t) can be specified.")
                    else:
                        self.data_df = self.data_df.replace(d, value=None)
                        return pd.DataFrame(d)

        if t is not None:
            row_num = self.data_df[self.data_df['time'] == t].index[0]
            print('drop time')
            self.data_df = self.data_df.drop([row_num])
            return self.data_df
        elif index == -1:
            num_rows = len(self.data_df.index) - 1
            transpose_df = self.data_df.T
            popped = transpose_df.pop(num_rows)
            self.data_df = transpose_df.T
            return popped

    def remove(self, d: dict = None, t: float = None, index: int = -1) -> None:
        """Remove an element/row/column of data

        Args:
            d (string): label for column in dataframe
            t (float, optional): timestamp(seconds) of element to be removed. Defaults to last item.
            index (int, optional): index using integer positions

        """
        self.dfpop(d=d, t=t, index=index)

    def clear(self) -> None:
        """Clear the SimResultDF"""
        self.data_df = pd.DataFrame()

    def time(self, index: int) -> float:
        """Get time for data point at index `index`

        Args:
            index (int)

        Returns:
            float: Time for which the data point at index `index` corresponds
        """
        return self.data_df.loc[index, 'time']

    def monotonicity(self) -> pd.DataFrame:
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

        result = []  # list for dataframes of monotonicity values
        for label in self.data_df.columns:  # iterates through each column label
            mono_sum = 0
            for i in [*range(0, len(self.data_df.index) - 1)]:  # iterates through for calculating monotonocity
                # print(test_df[label].iloc[i+1], ' - ', test_df[label].iloc[i])
                mono_sum += np.sign(self.data_df[label].iloc[i + 1] - self.data_df[label].iloc[i])
            result.append(pd.DataFrame({label: abs(mono_sum / (len(self.data_df.index) - 1))},
                                       index=['monotonicity']))  # adds each dataframe to a list
        temp_pd = pd.concat(result, axis=1)
        return temp_pd.drop(columns=['time'])


class LazySimResultDF(SimResultDF):  # lgtm [py/missing-equals]
    """
    Used to store the result of a simulation, which is only calculated on first request
    """

    def __init__(self, fcn: Callable, times: list = None, states: dict = None, _copy=True) -> None:
        """
        Args:
            fcn (callable): function (x) -> z where x is the state and z is the data
            times (array(float)): Times for each data point where times[n] corresponds to data[n]
            data (array(dict)): Data points where data[n] corresponds to times[n]
        """
        self.fcn = fcn
        self.__data = None
        if times is None or states is None:  # in case the error is that there is no output (theory)
            self.data_df = pd.DataFrame()
        else:
            # Lists that will be used to create the DataFrame with all the data
            temp_df_list = [pd.DataFrame(times.copy(), columns=['time'])]  # created with the column of time data
            if _copy:
                for x in deepcopy(states):
                    fcn_temp = []
                    #temp_df_list.append(pd.DataFrame(x))
                    print(x)
                    [fcn_temp.append(fcn(y)) for y in x]
                    temp_df_list.append(pd.DataFrame(fcn_temp))
            else:
                for x in states:
                    print(x)
                    fcn_temp = []
                    #temp_df_list.append(pd.DataFrame(x))
                    [fcn_temp.append(fcn(y)) for y in x]
                    temp_df_list.append(pd.DataFrame(fcn_temp))
            self.data_df = pd.concat(temp_df_list, axis=1)

    def __reduce__(self):
        return self.__class__.__base__, self.data_df

    def is_cached(self) -> bool:
        """
        Returns:
            bool: If the value has been calculated
        """
        return self.__data_df is not None

    def clear(self) -> None:
        """
        Clears the times, states, and data cache for a LazySimResultDF object
        """
        self.data_df = pd.DataFrame()

    def extend(self, other: "LazySimResultDF", _copy=True) -> None:
        """
        Extend the LazySimResultDF with another LazySimResultDF object
        Raise ValueError if SimResult is passed
        Function fcn of other LazySimResultDF MUST match function fcn of LazySimResultDF object to be extended

        Args:
            other (LazySimResultDF)
        """
        if isinstance(other, self.__class__) and isinstance(other.data_df, self.data_df.__class__):
            if _copy:
                self.data_df = pd.concat([self.data_df, deepcopy(other.data_df)], ignore_index=True, axis=0)
                self.data_df.reindex()
            else:
                self.data_df = pd.concat([self.data_df, other.data_df], ignore_index=True, axis=0)
                self.data_df.reindex()
        elif isinstance(other, SimResultDF):
            raise ValueError(
                f"ValueError: {self.__class__} cannot be extended by SimResult. First convert to SimResult using to_simresult() method.")
        else:
            raise ValueError(f"ValueError: Argument must be of type {self.__class__}.")
