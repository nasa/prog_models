# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from abc import ABC, abstractmethod
import numpy as np

from prog_models.prognostics_model import PrognosticsModel

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class LinearModel(PrognosticsModel, ABC):
    """
    A linear prognostics :term:`model`. Used when behavior can be described using a simple linear time-series model defined by the following equations:

    .. math::
        \dfrac{dx}{dt} = Ax + Bu + E

        z = Cx + D

        es = Fx + G

    where x is :term:`state`, u is :term:`input`, z is :term:`output` and es is :term:`event state`

    Linear Models must inherit from this class and define the following properties:
        * A: 2-d np.array[float], dimensions: n_states x n_states
        * B: 2-d np.array[float], optional (zeros by default), dimensions: n_states x n_inputs
        * C: 2-d np.array[float], optional (zeros by default), dimensions: n_outputs x n_states
        * D: 1-d np.array[float], dimensions: n_outputs x 1
        * E: 1-d np.array[float], optional (zeros by default), dimensions: n_states x 1
        * F: 2-d np.array[float], dimensions: n_es x n_states
        * G: 1-d np.array[float], optional (zeros by default), dimensions: n_es x 1
        * inputs:  list[str] - :term:`input` keys
        * states:  list[str] - :term:`state` keys
        * outputs: list[str] - :term:`output` keys
        * events:  list[str] - :term:`event` keys
    """


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._B = np.zeros((self.n_states, self.n_inputs))
        self._D = np.zeros(self.n_outputs)
        self._G = np.zeros(self.n_events)
        

        # matrixCheck = np.zeros((self.n_states, self.n_inputs))

        # if self.B is not matrixCheck and self.inputs:
        #     raise AssertionError('Attribute should not exist.')

        if self.F is None and type(self).event_state == LinearModel.event_state:
            raise AttributeError('LinearModel must define F if event_state is not defined. Either override event_state or define F.')

        self.matrixCheck()


    def matrixCheck(self) -> None:
        """
        Public class method for checking matrices dimensions across all properties of the model.
        """
        self._propertyCheck(self.n_states, self.n_states, ["A","states","states"])
        self._propertyCheck(self.n_states, self.n_inputs, ["B","states","inputs"])
        self._propertyCheck(self.n_outputs, self.n_states, ["C","outputs","states"])
        self._propertyCheck(self.n_outputs, 1, ["D","outputs","1"])
        self._propertyCheck(self.n_states, 1, ["E","states","1"])
        self._propertyCheck(self.n_events, 1, ["G","events","1"])

        if self.F is not None:
            self._propertyCheck(self.n_events, self.n_states, ["F","events","states"])

    def _propertyCheck(self, rowsCount : int, colsCount : int, notes : list) -> None:
        """
        matrix: Input matrix to check dimensions of (e.g. self.A, self.B, etc)
        rowsCount: Row count to check matrix against
        colsCount: Column count to check matrix against
        notes: List of strings containing information for exception message debugging
        """
        target_property = getattr(self, notes[0])
        if isinstance(target_property, list):
            setattr(self, notes[0], np.array(target_property))
        matrix = getattr(self, notes[0]) 
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Matrix type check failed: @property {} dimensions is not of type list or NumPy array.".format(notes[0]))
        
        matrixShape = matrix.shape
        #INCORRECT? Define Tests that go through the matrix and evaluate based of that.
        if notes[0] == 'G' or notes[0] == 'E' or notes[0] == 'D':
            if (matrixShape == () or
            matrixShape[0] != rowsCount or
            matrix.ndim != 1):
                raise AttributeError("Matrix size check failed: @property {} dimensions improperly formed along {} x {}.".format(notes[0],notes[1],notes[2]))
        # PART 1: Gives specific comment about information on error, this would be a run-time check
        elif (matrixShape == () or
            matrixShape[0] != rowsCount or # check matrix is 2 dimensional,
            matrix.ndim != 2 or
            matrixShape[1] != colsCount): # check all rows are equal to correct column count
                raise AttributeError("Matrix size check failed: @property {} dimensions improperly formed along {} x {}.".format(notes[0],notes[1],notes[2]))

    @property
    @abstractmethod
    def A(self):
        pass

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, value):
        if (value == 'setDefault'):
            self._B = np.zeros((self.n_states, self.n_inputs))
        else:
            self._B = value

    @property
    @abstractmethod
    def C(self):
        pass

    @property
    def D(self):
        return self._D

    @D.setter
    def D(self, value):
        if (value == 'setDefault'):
            self._D = np.zeros(self.n_outputs)
        else:
            self._D = value

    @property
    def E(self):
        return self._E
    
    @E.setter
    def E(self, value):
        if (value == 'setDefault'):
            self._E = np.zeros(self.n_states)
        else:
            self._E = value

    @property
    @abstractmethod
    def F(self):
        pass

    @property
    def G(self):
        return self._G
    
    @G.setter
    def G(self, value):
        if (value == 'setDefault'):
            self._G = np.zeros(self.n_events)
        else:
            self._G = value

    def dx(self, x, u):
        dx_array = np.matmul(self.A, x.matrix) + self.E
        if len(u.matrix) > 0:
            dx_array += np.matmul(self.B, u.matrix)
        return self.StateContainer(dx_array)
        
    def output(self, x):
        z_array = np.matmul(self.C, x.matrix) + self.D
        return self.OutputContainer(z_array)

    def event_state(self, x):
        es_array = np.matmul(self.F, x.matrix) + self.G
        return {key: value[0] for key, value in zip(self.events, es_array)}
    
    def threshold_met(self, x):
        es_array = np.matmul(self.F, x.matrix) + self.G
        return {key: value[0] <= 0 for key, value in zip(self.events, es_array)}
