# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from . import PrognosticsModel
from abc import ABC, abstractmethod
import numpy as np


class LinearModel(PrognosticsModel, ABC):
    """
    A linear prognostics model. Used when behavior can be described using a simple linear time-series model defined by the following equations:
        * dx/dt = Ax + Bu + E
        * z = Cx + D
        * es = Fx + G
    where x is state, u is input, z is output and es is event state

    Linear Models must inherit from this class and define the following properties:
        * A: 2-d numpy.array[float], dimensions: n_states x n_states
        * B: 2-d numpy.array[float], optional (zeros by default), dimensions: n_states x n_inputs
        * C: 2-d numpy.array[float], optional (zeros by default), dimensions: n_outputs x n_states
        * D: 1-d numpy.array[float], dimensions: n_outputs x 1
        * E: 1-d numpy.array[float], optional (zeros by default), dimensions: n_states x 1
        * F: 2-d numpy.array[float], dimensions: n_es x n_states
        * G: 1-d numpy.array[float], optional (zeros by default), dimensions: n_es x 1
        * inputs:  list[str] - input keys
        * states:  list[str] - state keys
        * outputs: list[str] - output keys
        * events:  list[str] - event keys
    """

    def __init__(self):
        super().__init__()
        self.matrixCheck()

    def matrixCheck(self):
        """
        Public class method for checking matrices dimensions across all properties of the model.
        """
        self._propertyCheck(self.A, len(self.states), len(self.states), ["A","states","states"])
        self._propertyCheck(self.B, len(self.states), len(self.inputs), ["B","states","inputs"])
        self._propertyCheck(self.C, len(self.outputs), len(self.states), ["C","outputs","states"])
        self._propertyCheck(self.D, len(self.outputs), 1, ["D","outputs","1"])
        self._propertyCheck(self.E, len(self.states), 1, ["E","states","1"])
        self._propertyCheck(self.G, len(self.events), 1, ["G","events","1"])

        if self.F is not None:
            self._propertyCheck(self.F, len(self.events), len(self.states), ["F","events","states"])

    def _propertyCheck(self, matrix, rowsCount, colsCount, notes):
        """
        matrix: Input matrix to check dimensions of (e.g. self.A, self.B, etc)
        rowsCount: Row count to check matrix against
        colsCount: Column count to check matrix against
        notes: List of strings containing information for exception message debugging
        """
        if isinstance(matrix, list):
            setattr(self, "self."+notes[0], np.array(matrix))
            matrix = getattr(self, "self."+notes[0])
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Matrix type check failed: @property {} dimensions is not of type list or NumPy array.".format(notes[0]))

        matrixShape = matrix.shape
        if matrixShape[0] != rowsCount or # check matrix is 2 dimensional, correspond to rows count
            len(matrixShape) == 1 or # check .shape returns 2-tuple, meaning all rows are of equal length
            matrixShape[1] != colsCount or # check all rows are equal to correct column count
            matrix.ndim != 2: # check matrix is 2 dimensional
            raise AttributeError("Matrix size check failed: @property {} dimensions improperly formed along {} x {}.".format(notes[0],notes[1],notes[2]))
   
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
    def E(self):
        n_states = len(self.states)
        return np.zeros((n_states, 1))

    @property
    @abstractmethod
    def C(self):
        pass

    @property
    def D(self):
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

        dx_array = np.matmul(self.A, x_array) + np.matmul(self.B, u_array) + self.E
        return {key: value[0] for key, value in zip(self.states, dx_array)}
        
    def output(self, x):
        x_array = np.array([list(x.values())]).T

        z_array = np.matmul(self.C, x_array) + self.D
        return {key: value[0] for key, value in zip(self.outputs, z_array)}

    def event_state(self, x):
        x_array = np.array([list(x.values())]).T

        es_array = np.matmul(self.F, x_array) + self.G
        return {key: value[0] for key, value in zip(self.events, es_array)}
