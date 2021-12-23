# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from . import PrognosticsModel
from abc import ABC, abstractmethod
import numpy as np


class LinearModel(PrognosticsModel, ABC):
    """
    A linear prognostics model. Used when behavior can be described using a simple linear time-series model defined by the following equations:
        * dx/dt = Ax + Bu + C
        * z = Dx + E
        * es = Fx + G
    where x is state, u is input, z is output and es is event state

    Linear Models must inherit from this class and define the following properties:
        * A: 2-d numpy.array[float] 
        * B: 2-d numpy.array[float], optional (zeros by default)
        * C: 2-d numpy.array[float], optional (zeros by default)
        * D: 2-d numpy.array[float]
        * E: 2-d numpy.array[float], optional (zeros by default)
        * F: 2-d numpy.array[float]
        * G: 2-d numpy.array[float], optional (zeros by default)
        * inputs:  list[str] - input keys
        * states:  list[str] - state keys
        * outputs: list[str] - output keys
        * events:  list[str] - event keys
    """

    @property
    @abstractmethod
    def A(self):
        pass

    @property
    def B(self):
        n_inputs = len(self.inputs)
        n_states = len(self.states)
        return np.zeros((n_states, n_inputs))

    @property
    def C(self):
        n_states = len(self.states)
        return np.zeros((n_states, 1))

    @property
    @abstractmethod
    def D(self):
        pass

    @property
    def E(self):
        n_outputs = len(self.outputs)
        return np.zeros((n_outputs, 1))

    @property
    @abstractmethod
    def F(self):
        pass

    @property
    def G(self):
        n_events = len(self.events)
        return np.zeros((n_events, 1))

    def dx(self, x, u):
        x_array = np.array([list(x.values())]).T
        u_array = np.array([list(u.values())]).T

        dx_array = np.matmul(self.A, x_array) + np.matmul(self.B, u_array) + self.C
        return {key: value[0] for key, value in zip(self.states, dx_array)}
        
    def output(self, x):
        x_array = np.array([list(x.values())]).T

        z_array = np.matmul(self.D, x_array) + self.E
        return {key: value[0] for key, value in zip(self.outputs, z_array)}

    def event_state(self, x):
        x_array = np.array([list(x.values())]).T

        es_array = np.matmul(self.F, x_array) + self.G
        return {key: value[0] for key, value in zip(self.events, es_array)}
