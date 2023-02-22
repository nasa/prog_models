# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from abc import ABC, abstractmethod
import numpy as np

from prog_models import PrognosticsModel

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

    # Set these to None
    default_parameters = {
        '_B' : None,
        '_C' : None,
        '_D' : None,
        '_E' : None,
        '_G' : None
    }

# Set it to None instead of setDefault
# Look into creating __copy__ function that 
    # __deepcopy__ 
    def __init__(self, **kwargs):
        params = LinearModel.default_parameters.copy()
        params.update(self.default_parameters)
        params.update(kwargs)
        super().__init__(**params)

        self.A = self.A
        self.B = self.B
        self.C = self.C
        self.D = self.D
        self.E = self.E
        self.G = self.G

        # Add Error when wrong parameter
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
        
        if (matrix.ndim != 2 or # Checks too see if matrix is two-dimensional
            matrixShape[0] != rowsCount or # checks if matrix has correct row count 
            matrixShape[1] != colsCount): # check all rows are equal to correct column count
                raise AttributeError("Matrix size check failed: @property {} dimensions improperly formed along {} x {}.".format(notes[0],notes[1],notes[2]))

    @property
    def A(self):
        return self.parameters['_A']

    @A.setter
    def A(self, value):
        self.parameters['_A'] = value

    @property
    def B(self):
        return self.parameters['_B']

    @B.setter
    def B(self, value):
        if (value is None):
            self.parameters['_B'] = np.zeros((self.n_states, self.n_inputs))
        else:
            self.parameters['_B'] = value

    @property
    def C(self):
        return self.parameters['_C']

    @C.setter
    def C(self, value):
        if (value is None):
            self.parameters['_C'] = np.zeros((self.n_outputs, self.n_states))
        else:
            self.parameters['_C'] = value

    @property
    def D(self):
        return self.parameters['_D']

    @D.setter
    def D(self, value):
        if (value is None):
            self.parameters['_D'] = np.zeros((self.n_outputs, 1))
        else:
            self.parameters['_D'] = value

    @property
    def E(self):
        return self.parameters['_E']
    
    @E.setter
    def E(self, value):
        if (value is None):
            self.parameters['_E'] = np.zeros((self.n_states, 1))
        else:
            self.parameters['_E'] = value

    @property
    @abstractmethod
    def F(self):
        pass

    @property
    def G(self):
        return self.parameters['_G']
    
    @G.setter
    def G(self, value):
        if (value is None):
            self.parameters['_G'] = np.zeros((self.n_events, 1))
        else:
            self.parameters['_G'] = value

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
