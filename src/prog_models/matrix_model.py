# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from .exceptions import ProgModelInputException, ProgModelTypeError
from .prognostics_model import PrognosticsModel, PrognosticsModelParameters
from .sim_result import SimResult, LazySimResult
from .utils import ProgressBar
from warnings import warn

from abc import ABC, abstractmethod
from collections import abc
from numbers import Number
import numpy as np
import types


class MatrixModelParameters(PrognosticsModelParameters):
    """
    MatrixModelParameters extends PrognosticsModelParameters to provide the logic specific to MatrixModel. Specifically, it provides the logic for updating the noise parameters specific to the MatrixModel.

    Args:
        model (Matrix): The model with member parameters of type PrognosticsModelParameters.

    Use:
        # Upgrade model parameters
        model.parameters = MatrixModelParameters(model)
    """
    def __init__(self, model):
        self.__m = model
        super().__init__(model, model.parameters, model.parameters.callbacks)

    def __setitem__(self, key, value):
        # TODO(CT): Link to process_noise/process_noise_mat option

        super().__setitem__(key, value)

        # TODO(CT): Fix process_noise/process_noise_mat connection
        # I'm thinking hidden variable _process_noise_mat- if process_noise_matrix set directly, set flag- otherwise link it to process_noise

        # Logic specific to matrix 
        if key == "process_noise":
            # if updating process_noise- update process_noise_matrix to equivilant matrix
            if callable(value):
                def process_noise_adapt(self, x, dt=1):
                    x_dict = {state: x_i for (state, x_i) in zip(self.states, x)}
                    x_dict = self.apply_process_noise(x_dict, dt)
                    return np.array([[x_dict[state]] for state in self.states])
                
                self.__m.apply_process_noise_matrix = types.MethodType(process_noise_adapt, self.__m)
                warn('For maximum efficiency when using custom process noise functions with Matrix-type models, set process_noise_matrix as well as process_noise.')
            else:
                self['process_noise_matrix'] = np.array(
                    [[value[key]] for key in self.__m.states])
        elif key == "measurement_noise":
            # if updating measurement_noise- update measurement_noise_matrix to equivilant matrix
            if callable(value):
                def measurement_noise_adapt(self, z):
                    z_dict = {output: z_i for (output, z_i) in zip(self.outputs, z)}
                    z_dict = self.apply_measurement_noise(z_dict)
                    return np.array([[z_dict[output]] for output in self.outputs])
                
                self.__m.apply_measurement_noise_matrix = types.MethodType(measurement_noise_adapt, self.__m)
                warn('For maximum efficiency when using custom measurement noise functions with Matrix-type models, set measurement_noise_matrix as well as measurement_noise.')
            else:
                self['measurement_noise_matrix'] = np.array(
                    [[value[key]] for key in self.__m.outputs])
        elif key == 'process_noise_matrix':
            if callable(self['process_noise_matrix']):  # Provided a function
                self.__m.apply_process_noise_matrix = types.MethodType(self['process_noise_matrix'], self.__m)
            else:  # Not a function
                # Process noise is single number - convert to dict
                if isinstance(self['process_noise_matrix'], Number):
                    self['process_noise_matrix'] = {key: self['process_noise_matrix'] for key in self.__m.states}
                
                # Process distribution type
                if 'process_noise_dist_matrix' in self and self['process_noise_dist_matrix'].lower() not in ["gaussian", "normal"]:
                    # Update process noise distribution to custom
                    if self['process_noise_dist_matrix'].lower() == "uniform":
                        def uniform_process_noise(self, x, dt=1):
                            return {key: x[key] + \
                                dt*np.random.uniform(-self.parameters['process_noise_matrix'][key], self.parameters['processprocess_noise_matrix_noise'][key], size=None if np.isscalar(x[key]) else len(x[key])) \
                                    for key in self.states}
                        self.__m.apply_process_noise_matrix = types.MethodType(uniform_process_noise, self.__m)
                    elif self['process_noise_dist_matrix'].lower() == "triangular":
                        def triangular_process_noise(self, x, dt=1):
                            return {key: x[key] + \
                                dt*np.random.triangular(-self.parameters['process_noise_matrix'][key], 0, self.parameters['process_noise_matrix'][key], size=None if np.isscalar(x[key]) else len(x[key])) \
                                    for key in self.states}
                        self.__m.apply_process_noise_matrix = types.MethodType(triangular_process_noise, self.__m)
                    else:
                        raise ProgModelTypeError("Unsupported process noise distribution")
                
                # Make sure every key is present (single value already handled above)
                if self['process_noise_matrix'].shape[0] != len(self.__m.states):
                    raise ProgModelTypeError("Process noise must have every key in model.states")
        elif key == 'measurement_noise_matrix':
            if callable(self['measurement_noise_matrix']):
                self.__m.apply_measurement_noise_matrix = types.MethodType(self['measurement_noise_matrix'], self.__m)
            else:
                # Process noise is single number - convert to dict
                if isinstance(self['measurement_noise_matrix'], Number):
                    self['measurement_noise_matrix'] = {key: self['measurement_noise_matrix'] for key in self.__m.outputs}
                
                # Process distribution type
                if 'measurement_noise_dist_matrix' in self and self['measurement_noise_dist_matrix'].lower() not in ["gaussian", "normal"]:
                    # Update measurement noise distribution to custom
                    if self['measurement_noise_dist_matrix'].lower() == "uniform":
                        def uniform_noise(self, x):
                            return {key: x[key] + \
                                np.random.uniform(-self.parameters['measurement_noise_matrix'][key], self.parameters['measurement_noise_matrix'][key], size=None if np.isscalar(x[key]) else len(x[key])) \
                                    for key in self.outputs}
                        self.__m.apply_measurement_noise_matrix = types.MethodType(uniform_noise, self.__m)
                    elif self['measurement_noise_dist_matrix'].lower() == "triangular":
                        def triangular_noise(self, x):
                            return {key: x[key] + \
                                np.random.triangular(-self.parameters['measurement_noise_matrix'][key], 0, self.parameters['measurement_noise_matrix'][key], size=None if np.isscalar(x[key]) else len(x[key])) \
                                    for key in self.outputs}
                        self.__m.apply_measurement_noise_matrix = types.MethodType(triangular_noise, self.__m)
                    else:
                        raise ProgModelTypeError("Unsupported measurement noise distribution")
                
                # Make sure every key is present (single value already handled above)
                if self['measurement_noise_matrix'].shape[0] != len(self.__m.outputs):
                    raise ProgModelTypeError("Measurement noise must have ever key in model.states")


class MatrixModel(PrognosticsModel, ABC):
    """
    A time-variant state space model of system degradation behavior, where states, inputs, outputs, event_states, and thresholds_met are represented as matricies instead of dictionaries. This is important for some applications like surrogate and machine learned models where the state is represented by a tensor, and operations by matrix operations. Simulation functions propogate the state using the matrix form, preventing the inefficiency of having to convert to and from dictionaries.

    The MatrixModel class is an extension of the basic PrognosticsModel class. Like PrognosticsModel, MatrixModel a wrapper around a mathematical model of a system. The model is represented by a state_matrix, output_matrix, input_matrix, event_state_matrix, and threshold_matrix equations.

    A Model also has a parameters structure, which contains fields for various model parameters.

    Keyword Args
    ------------
        process_noise : Optional, float or Dict[Srt, float]
          Process noise (applied at dx/next_state). 
          Can be number (e.g., .2) applied to every state, a dictionary of values for each 
          Process noise provided in this form will be converted into its matrix form for computation
          state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x, dt) -> x
        process_noise_matrix : Optional, float or Array
          Process noise (applied at dx/next_state). 
          Can be number (e.g., .2) applied to every state, a dictionary of values for each 
          state (e.g., np.array([[0.2], [0.3]])), or a function (x, dt) -> x
        process_noise_dist : Optional, String
          distribution for process noise (e.g., normal, uniform, triangular)
        process_noise_dist_matrix : Optional, String
          distribution for process noise (e.g., normal, uniform, triangular) specific to the matrix model
        measurement_noise : Optional, float or Dict[Srt, float]
          Measurement noise (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of values for each
          output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        measurement_noise_matrix : Optional, float or Array
          Measurement noise (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of values for each
          output (e.g., np.array([[0.2], [0.3]]), or a function (z) -> z
        measurement_noise_dist : Optional, String
          distribution for measurement noise (e.g., normal, uniform, triangular)
        measurement_noise_dist_matrix : Optional, String
          distribution for measurement noise (e.g., normal, uniform, triangular) specific to the matrix model
        Additional parameters specific to the model

    Raises
    ------
        ProgModelTypeError, ProgModelInputException, ProgModelException

    Example
    -------
        m = MatrixModel(process_noise = 3.2)

    Attributes
    ----------
        is_vectorized : bool, optional
            True if the model is vectorized, False otherwise. Default is False
        default_parameters : dict[str, float], optional
            Default parameters for the model class
        parameters : dict[str, float]
            Parameters for the specific model object. This is created automatically from the default_parameters and kwargs
        state_limits: dict[str, tuple[float, float]], optional
            Limits on the state variables format {'state_name': (lower_limit, upper_limit)}
        state_limits_mat: tuple[np.array, np.array], optional
            Limits on the state variables format (lower_limit, upper_limit)
            Note: will be generated from state_limits if not provided
        param_callbacks : dict[str, list[function]], optional
            Callbacks for derived parameters
        inputs: List[str]
            Identifiers for each input
        states: List[str]
            Identifiers for each state
        outputs: List[str]
            Identifiers for each output
        performance_metric_keys: List[str], optional
            Identifiers for each performance metric
        events: List[str], optional
            Identifiers for each event predicted 
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Upgrade parameters to include logic for matrix model
        self.parameters = MatrixModelParameters(self)

        # TODO: only if mat not a member
        self.state_limits_mat = (
            [[self.state_limits[key][0]] if key in self.state_limits else [-float('inf')] for key in self.states],
            [[self.state_limits[key][1]] if key in self.state_limits else [float('inf')] for key in self.states])
        # TODO(CT): Issue- this doesn't support changing state_limits after construction

    def initialize_matrix(self, u = None, z = None):
        """
        Calculate initial state given inputs and outputs in matrix form. This method or initialize() is overwritten by children classes

        Parameters
        ----------
        u : np.array as column vector
            Inputs, in order of model.inputs \n
            e.g., u = np.array([[3.2]]) given inputs = ['i']
        z : np.array as column vector
            Outputs, in order of model.outputs \n
            e.g., z = np.array([[12.4], [3.3]]) given inputs = ['t', 'v']

        Returns
        -------
        x : np.array as column vector
            First state, in order of model.states \n
            e.g., x = np.array([[332.1], [221.003]]) given states = ['abc', 'def']

        Example
        -------
        | m = MatrixModel() # Replace with specific model being simulated
        | u = np.array([[3.2], [1.2]])
        | z = np.array([[2.2], [3.3], [4.4]])
        | x = m.initialize_matrix(u, z) # Initialize first state

        See Also
        --------
        initialize
        """
        pass

    def initialize(self, u = {}, z = {}):
        """
        Calculate initial state given inputs and outputs in dictionary form. calls initialize_matrix

        Parameters
        ----------
        u : dict
            Inputs, with keys defined by model.inputs \n
            e.g., u = {'i':3.2} given inputs = ['i']
        z : dict
            Outputs, with keys defined by model.outputs \n
            e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']

        Returns
        -------
        x : dict
            First state, with keys defined by model.states \n
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = {'u1': 3.2}
        | z = {'z1': 2.2}
        | x = m.initialize(u, z) # Initialize first state

        See Also
        --------
        initialize_matrix
        """

        # Convert to matrix
        z_mat = np.array([[z[key]] if key in z else [None] for key in self.outputs])
        u_mat = np.array([[u[key]] if key in u else [None] for key in self.inputs])

        # Initialize
        x = self.initialize_matrix(u_mat, z_mat)

        # Convert to dictionary
        return {key: x_i for key, x_i in zip(self.states, x)}

    def dx_matrix(self, x, u):
        """
        Calculate the first derivative of state `x` at a specific time `t`, given state and input, in matrix form. This method is overwritten by children classes

        Parameters
        ----------
        x : np.array as column vector
            state, in order of model.states \n
            e.g., x = np.array([[332.1], [221.003]]) given states = ['abc', 'def']
        u : np.array as column vector
            Inputs, in order of model.inputs \n
            e.g., u = np.array([[3.2]]) given inputs = ['i']

        Returns
        -------
        dx : np.array as column vector
            first derivative of state, in order of model.states \n
            e.g., x = np.array([[-3.1], [1.7]]) given states = ['abc', 'def']
        
        Example
        -------
        | m = DerivMatrixModel() # Replace with specific model being simulated
        | u = np.array([[3.2], [1.2]])
        | z = np.array([[2.2], [3.3], [4.4]])
        | x = m.initialize(u, z) # Initialize first state
        | dx = m.dx_matrix(x, u) # Returns first derivative of state given input u
        
        See Also
        --------
        dx
        next_state
        next_state_matrix

        Note
        ----
        A model should overwrite either `next_state_matrix` or `dx_matrix`. Override `dx_matrix` for continuous models,
        and `next_state_matrix` for discrete, where the behavior cannot be described by the first derivative
        """
        pass

    def dx(self, x, u):
        """
        Calculate the first derivative of state `x` at a specific time `t`, given state and input in dictionary form. Uses dx_matrix function

        Parameters
        ----------
        x : dict
            state, with keys defined by model.states \n
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        u : dict
            Inputs, with keys defined by model.inputs \n
            e.g., u = {'i':3.2} given inputs = ['i']

        Returns
        -------
        dx : dict
            First derivitive of state, with keys defined by model.states \n
            e.g., dx = {'abc': 3.1, 'def': -2.003} given states = ['abc', 'def']
        
        Example
        -------
        | m = DerivProgModel() # Replace with specific model being simulated
        | u = {'u1': 3.2}
        | z = {'z1': 2.2}
        | x = m.initialize(u, z) # Initialize first state
        | dx = m.dx(x, u) # Returns first derivative of state given input u
        
        See Also
        --------
        dx_matrix
        next_state
        next_state_matrix
        """
        # Convert to matrix
        x_mat = np.array([[x[key]] for key in self.states])
        u_mat = np.array([[u[key]] for key in self.inputs])

        # Calculate
        x_next = self.dx(x_mat, u_mat)

        # Convert to dictionary
        return {state: x_next_i for (state, x_next_i) in zip(self.states, x_next)}

    def next_state_matrix(self, x, u, dt):
        """
        State transition equation: Calculate next state, where state and input are represented in matrix format. 

        Parameters
        ----------
        x : dict
            state, with keys defined by model.states \n
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        u : dict
            Inputs, with keys defined by model.inputs \n
            e.g., u = {'i':3.2} given inputs = ['i']
        dt : number
            Timestep size in seconds (≥ 0) \n
            e.g., dt = 0.1
        

        Returns
        -------
        x : dict
            next state at t0 + dt, with keys defined by model.states \n
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Example
        -------
        | m = MatrixModel() # Replace with specific model being simulated
        | u = np.array([[3.2], [1.2]])
        | z = np.array([[2.2], [3.3], [4.4]])
        | x = m.initialize(u, z) # Initialize first state
        | dx = m.next_state_matrix(x, u, 0.1) # Returns first derivative of state given input u
        
        See Also
        --------
        dx_matrix
        dx
        next_state

        Note
        ----
        A model should overwrite either `next_state_matrix` or `dx_matrix`. Override `dx_matrix` for continuous models, and `next_state_matrix` for discrete, where the behavior cannot be described by the first derivative
        """
        return x + self.dx_matrix(x, u) * dt

    def next_state(self, x, u, dt):
        """
        State transition equation: Calculate next state using state and input, represented by a dictionary. Uses next_state_matrix method

        Parameters
        ----------
        x : dict
            state, with keys defined by model.states \n
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        u : dict
            Inputs, with keys defined by model.inputs \n
            e.g., u = {'i':3.2} given inputs = ['i']
        dt : number
            Timestep size in seconds (≥ 0) \n
            e.g., dt = 0.1
        

        Returns
        -------
        x : dict
            Next state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = {'u1': 3.2}
        | z = {'z1': 2.2}
        | x = m.initialize(u, z) # Initialize first state
        | x = m.next_state(x, u, 0.1) # Returns state at 3.1 seconds given input u
        
        See Also
        --------
        dx_matrix
        dx
        next_state_matrix
        """
        # Convert to matrix
        x_mat = np.array([[x[key]] for key in self.states])
        u_mat = np.array([[u[key]] for key in self.inputs])

        # Calculate
        x_next = self.next_state_matrix(x_mat, u_mat, dt)

        # Convert to dictionary
        return {state: x_next_i for (state, x_next_i) in zip(self.states, x_next)}

    @abstractmethod
    def output_matrix(self, x):
        """
        Calculate outputs given state in matrix form. Overwritten by children classes

        Parameters
        ----------
        x : np.array as column vector
            state, in order of model.states \n
            e.g., x = np.array([[332.1], [221.003]]) given states = ['abc', 'def']
        
        Returns
        -------
        z : np.array as column vector
            outputs, in order of model.outputs \n
            e.g., z = np.array([[12.4], [3.3]]) given outputs = ['t', 'v']

        Example
        -------
        | m = MatrixModel() # Replace with specific model being simulated
        | u = np.array([[3.2], [1.2]])
        | z = np.array([[2.2], [3.3], [4.4]])
        | x = m.initialize(u, z) # Initialize first state
        | z = m.output(x) # Returns {'o1': 1.2}

        See Also
        --------
        output
        """
        pass

    def output(self, x):
        """
        Calculate outputs given state in dictionary form. Uses output_matrix function

        Parameters
        ----------
        x : dict
            state, with keys defined by model.states \n
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        z : dict
            Outputs, with keys defined by model.outputs. \n
            e.g., z = {'t':12.4, 'v':3.3} given outputs = ['t', 'v']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = {'u1': 3.2}
        | z = {'z1': 2.2}
        | x = m.initialize(u, z) # Initialize first state
        | z = m.output(x) # Returns {'o1': 1.2}

        See Also
        --------
        output_matrix
        """
        # Convert to matrix
        x_mat = np.array([[x[key]] for key in self.states])

        # Calculate
        z = self.output(x_mat)

        # Convert to dictionary
        return {output: z_i for (output, z_i) in zip(self.outputs, z)}

    def event_state_matrix(self, x):
        """
        Calculate event states (i.e., measures of progress towards event (0-1, where 0 means event has occured)), given state represented by a matrix

        Parameters
        ----------
        x : np.array as column vector
            state, in order of model.states \n
            e.g., x = np.array([[332.1], [221.003]]) given states = ['abc', 'def']
        
        Returns
        -------
        event_state : np.array as column vector
            event states, in order of model.events \n
            e.g., es = np.array([[0.32],[0.75]]) given events = ['eol', 'eod']

        Example
        -------
        | m = MatrixModel() # Replace with specific model being simulated
        | u = np.array([[3.2], [1.2]])
        | z = np.array([[2.2], [3.3], [4.4]])
        | x = m.initialize(u, z) # Initialize first state
        | event_state = m.event_state(x) # Returns {'e1': 0.8, 'e2': 0.6}

        Note
        ----
        Default is to return an empty array (for system models that do not include any events)
        
        See Also
        --------
        threshold_met_matrix
        threshold_met
        event_state
        """
        return np.array([[]])

    def event_state(self, x):
        """
        Calculate event states (i.e., measures of progress towards event (0-1, where 0 means event has occured)), given a state, represented by a dictionary

        Parameters
        ----------
        x : dict
            state, with keys defined by model.states\n
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        event_state : dict
            Event States, with keys defined by prognostics_model.events.\n
            e.g., event_state = {'EOL':0.32} given events = ['EOL']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = {'u1': 3.2}
        | z = {'z1': 2.2}
        | x = m.initialize(u, z) # Initialize first state
        | event_state = m.event_state(x) # Returns {'e1': 0.8, 'e2': 0.6}

        Note
        ----
        Default is to return an empty array (for system models that do not include any events)
        
        See Also
        --------
        event_state_matrix
        threshold_met_matrix
        threshold_met
        """

        # Convert to matrix
        x_mat = np.array([[x[key]] for key in self.states])

        # Calculate
        es = self.event_state_matrix(x_mat)

        # Convert to dictionary
        return {event: es_i for (event, es_i) in zip(self.events, es)}

    def threshold_matrix(self, x):
        """
        For each event threshold, calculate if it has been met given the state, represented by a matrix

        Parameters
        ----------
        x : np.array as column vector
            state, in order of model.states \n
            e.g., x = np.array([[332.1], [221.003]]) given states = ['abc', 'def']
        
        Returns
        -------
        thresholds_met : np.array as column vector
            state, in order of model.states \n
            e.g., x = np.array([[True], [False]]) given events = ['eod', 'eol']

        Example
        -------
        | m = MatrixModel() # Replace with specific model being simulated
        | u = np.array([[3.2], [1.2]])
        | z = np.array([[2.2], [3.3], [4.4]])
        | x = m.initialize(u, z) # Initialize first state
        | threshold_met = m.threshold_met(x) # returns np.array([[False], [False]])

        Note
        ----
        If not overwritten, the default behavior is to say the threshold is met if the event state is <= 0
        
        See Also
        --------
        event_state_matrix
        event_state
        threshold
        """
        es = self.event_state_matrix(x)
        return  es[es <= 0]

    def threshold(self, x):
        """
        For each event threshold, calculate if it has been met given state, represented by a dictionary

        Parameters
        ----------
        x : dict
            state, with keys defined by model.states\n
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        thresholds_met : dict
            If each threshold has been met (bool), with keys defined by prognostics_model.events\n
            e.g., thresholds_met = {'EOL': False} given events = ['EOL']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = {'u1': 3.2}
        | z = {'z1': 2.2}
        | x = m.initialize(u, z) # Initialize first state
        | threshold_met = m.threshold_met(x) # returns {'e1': False, 'e2': False}

        Note
        ----
        If not overwritten, the default behavior is to say the threshold is met if the event state is <= 0
        
        See Also
        --------
        event_state_matrix
        event_state
        threshold_met_matrix
        """
        # Convert to matrix
        x_mat = np.array([[x[key]] for key in self.states])

        # Calculate
        t_met = self.threshold_matrix(x_mat)

        # Convert to dictionary
        return {event: t_met_i for (event, t_met_i) in zip(self.events, t_met)}

    def apply_process_noise_matrix(self, x, dt=1):
        """
        Apply process noise to the state, represted by a matrix

        Parameters
        ----------
        x : np.array as column vector
            state, in order of model.states \n
            e.g., x = np.array([[332.1], [221.003]]) given states = ['abc', 'def']
        dt : Number, optional
            Time step (e.g., dt = 0.1)

        Returns
        -------
        x : np.array as column vector
            state, with applied noise in order of model.states \n
            e.g., x = np.array([[332.15], [221.023]]) given states = ['abc', 'def']
        
        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = np.array([[3.2], [1.2]])
        | z = np.array([[2.2], [3.3], [4.4]])
        | x = m.initialize(u, z) # Initialize first state
        | x = m.apply_process_noise(x) 

        Note
        ----
        Configured using parameters `process_noise_matrix` and `process_noise_dist_matrix`
        """
        x + np.random.normal(
                    0, self.parameters['process_noise_matrix'],
                    size=x.shape) * dt
        return x

    def apply_limits_matrix(self, x):
        """
        Apply state bound limits given state represented by a matrix. Any state outside of limits will be set to the closest limit.

        Parameters
        ----------
        x : np.array as column vector
            state, in order of model.states \n
            e.g., x = np.array([[332.1], [221.003]]) given states = ['abc', 'def']

        Returns
        -------
        x : np.array as column vector
            bounded state, in order of model.states \n
            e.g., x = np.array([[332.1], [221.003]]) given states = ['abc', 'def']
        """
        return x.clip(*self.state_limits_mat)

    def __next_state_matrix(self, x, u, dt):
        """
        State transition equation: Calls next_state_matrix(), calculating the next state, and then adds noise and limits

        Parameters
        ----------
        x : np.array as column vector
            state, in order of model.states \n
            e.g., x = np.array([[332.1], [221.003]]) given states = ['abc', 'def']
        u : dict
            Inputs, with keys defined by model.inputs \n
            e.g., u = {'i':3.2} given inputs = ['i']
        dt : number
            Timestep size in seconds (≥ 0) \n
            e.g., dt = 0.1

        Returns
        -------
        x : np.array as column vector
            next state, in order of model.states \n
            e.g., x = np.array([[332.1], [221.003]]) given states = ['abc', 'def']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = np.array([[3.2], [1.2]])
        | z = np.array([[2.2], [3.3], [4.4]])
        | x = m.initialize(u, z) # Initialize first state
        | x = m.__next_state(x, u, 0.1) # Returns state, with noise, at 3.1 seconds given input u
        
        See Also
        --------
        next_state_matrix
        dx_matrix


        Note
        ----
        A model should not overwrite '__next_state_matrix'
        A model should overwrite either `next_state_matrix` or `dx_matrix`. Override `dx_matrix` for continuous models, and `next_state_matrix` for discrete, where the behavior cannot be described by the first derivative.
        """
        # Calculate next state and add process noise
        next_state = self.apply_process_noise_matrix(self.next_state_matrix(x, u, dt))

        # Apply Limits
        return self.apply_limits_matrix(next_state)

    def apply_measurement_noise_matrix(self, z):
        """
        Apply measurement noise to the measurement, represented by a matrix

        Parameters
        ----------
        z : np.array as column vector
            Outputs, in order of model.outputs \n
            e.g., z = np.array([[12.4], [3.3]]) given inputs = ['t', 'v']
 
        Returns
        -------
        z : np.array as column vector
            Outputs, with applied noise in order of model.outputs \n
            e.g., z = np.array([[12.45], [3.26]]) given inputs = ['t', 'v']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | z = np.array([[12.45], [3.26]])
        | z = m.apply_measurement_noise(z)

        Note
        ----
        Configured using parameters `measurement_noise_matrix` and `measurement_noise_dist_matrix`
        """
        z + np.random.normal(
                    0, self.parameters['measurement_noise_matrix'],
                    size=z.shape)
        return z
    
    def __output_matrix(self, x):
        """
        Calls output, which calculates next state forward one timestep, and then adds noise

        Parameters
        ----------
        x : dict
            state, with keys defined by model.states \n
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        z : dict
            Outputs, with keys defined by model.outputs. \n
            e.g., z = {'t':12.4, 'v':3.3} given outputs = ['t', 'v']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | z = np.array([[12.45], [3.26]])
        | z = m.__output_matrix(z)
        """
        # Calculate next state, forward one timestep
        z = self.output_matrix(x)

        # Add measurement noise
        return self.apply_measurement_noise(z)

    def simulate_to_threshold(self, future_loading_eqn, first_output = None, threshold_keys = None, **kwargs):
        """
        Simulate prognostics model until any or specified threshold(s) have been met. Uses the special matrix functions (e.g., dx_matrix, output_matrix) in simulation, eliminating any need to convert to and from dictionaries. 

        Parameters
        ----------
        future_loading_eqn : callable
            Function of (t) -> z used to predict future loading (output) at a given time (t). z is returned in matrix format as a column vector.

        Keyword Arguments
        -----------------
        t0 : Number, optional
            Starting time for simulation in seconds (default: 0.0) \n
        dt : Number or function, optional
            time step (s), e.g. dt = 0.1 or function (t) -> dt\n
        save_freq : Number, optional
            Frequency at which output is saved (s), e.g., save_freq = 10 \n
        save_pts : List[Number], optional
            Additional ordered list of custom times where output is saved (s), e.g., save_pts= [50, 75] \n
        horizon : Number, optional
            maximum time that the model will be simulated forward (s), e.g., horizon = 1000 \n
        first_output : dict, optional
            First measured output, needed to initialize state for some classes. Can be omitted for classes that dont use this
        threshold_keys: List[str] or str, optional
            Keys for events that will trigger the end of simulation.
            If blank, simulation will occur if any event will be met ()
        x : dict, optional
            initial state dict, e.g., x= {'x1': 10, 'x2': -5.3}\n
        thresholds_met_eqn : function/lambda, optional
            custom equation to indicate logic for when to stop sim f(thresholds_met) -> bool\n
        print : bool, optional
            toggle intermediate printing, e.g., print = True\n
            e.g., m.simulate_to_threshold(eqn, z, dt=0.1, save_pts=[1, 2])
        progress : bool, optional
            toggle progress bar printing, e.g., progress = True\n
    
        Returns
        -------
        times: Array[number]
            Times for each simulated point
        inputs: SimResult
            Future input (from future_loading_eqn) for each time in times
        states: SimResult
            Estimated states for each time in times
        outputs: SimResult
            Estimated outputs for each time in times
        event_states: SimResult
            Estimated event state (e.g., SOH), between 1-0 where 0 is event occurance, for each time in times
        
        Raises
        ------
        ProgModelInputException

        See Also
        --------
        simulate_to

        Example
        -------
        | def future_load_eqn(t):
        |     if t< 5.0: # Load is 3.0 for first 5 seconds
        |         return np.array([3.0])
        |     else:
        |         return np.array([5.0])
        | first_output = {'o1': 3.2, 'o2': 1.2}
        | m = PrognosticsModel() # Replace with specific model being simulated
        | (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load_eqn, first_output)

        Note
        ----
        configuration of the model is set through model.parameters.\n
        """
        # Input Validation
        if first_output and not all(key in first_output for key in self.outputs):
            raise ProgModelInputException("Missing key in 'first_output', must have every key in model.outputs")

        if not (callable(future_loading_eqn)):
            raise ProgModelInputException("'future_loading_eqn' must be callable f(t)")
        
        if isinstance(threshold_keys, str):
            # A single threshold key
            threshold_keys = [threshold_keys]

        if threshold_keys and not all([key in self.events for key in threshold_keys]):
            raise ProgModelInputException("threshold_keys must be event names")

        # Configure
        config = { # Defaults
            't0': 0.0,
            'dt': 1.0,
            'save_pts': [],
            'save_freq': 10.0,
            'horizon': 1e100, # Default horizon (in s), essentially inf
            'print': False,
            'progress': False
        }
        config.update(kwargs)
        
        # Configuration validation
        if not isinstance(config['dt'], Number) and not callable(config['dt']):
            raise ProgModelInputException("'dt' must be a number or function, was a {}".format(type(config['dt'])))
        if isinstance(config['dt'], Number) and config['dt'] < 0:
            raise ProgModelInputException("'dt' must be positive, was {}".format(config['dt']))
        if not isinstance(config['save_freq'], Number):
            raise ProgModelInputException("'save_freq' must be a number, was a {}".format(type(config['save_freq'])))
        if config['save_freq'] <= 0:
            raise ProgModelInputException("'save_freq' must be positive, was {}".format(config['save_freq']))
        if not isinstance(config['save_pts'], abc.Iterable):
            raise ProgModelInputException("'save_pts' must be list or array, was a {}".format(type(config['save_pts'])))
        if not isinstance(config['horizon'], Number):
            raise ProgModelInputException("'horizon' must be a number, was a {}".format(type(config['horizon'])))
        if config['horizon'] < 0:
            raise ProgModelInputException("'save_freq' must be positive, was {}".format(config['horizon']))
        if 'x' in config and not all([state in config['x'] for state in self.states]):
            raise ProgModelInputException("'x' must contain every state in model.states")
        if 'thresholds_met_eqn' in config and not callable(config['thresholds_met_eqn']):
            raise ProgModelInputException("'thresholds_met_eqn' must be callable (e.g., function or lambda)")
        if 'thresholds_met_eqn' in config and config['thresholds_met_eqn'].__code__.co_argcount != 1:
            raise ProgModelInputException("'thresholds_met_eqn' must accept one argument (thresholds)-> bool")
        if not isinstance(config['print'], bool):
            raise ProgModelInputException("'print' must be a bool, was a {}".format(type(config['print'])))

        # Setup
        t = config['t0']
        u = future_loading_eqn(t)
        if 'x' in config:
            x = config['x']
        else:
            x = self.initialize(u, first_output)
        
        # Optimization
        next_state = self.__next_state_matrix
        output = self.__output_matrix
        thresthold_met_eqn = self.threshold_met_matrix
        event_state = self.event_state_matrix # TODO: Get this working

        progress = config['progress']
        def check_thresholds(thresholds_met):
            t_met = [thresholds_met[key] for key in threshold_keys]
            if len(t_met) > 0 and not np.isscalar(list(t_met)[0]):
                return np.any(t_met)
            return any(t_met)
        if 'thresholds_met_eqn' in config:
            check_thresholds = config['thresholds_met_eqn']
            threshold_keys = []
        elif threshold_keys is None: 
            # Note: Setting threshold_keys to be all events if it is None
            threshold_keys = self.events
        
        # Convert to indexes
        threshold_keys = [self.events.index(event) for event in threshold_keys]

        # Initialization of save arrays
        saved_times = []
        saved_inputs = []
        saved_states = []  
        saved_outputs = []
        saved_event_states = []
        save_freq = config['save_freq']
        horizon = t+config['horizon']
        next_save = t+save_freq
        save_pt_index = 0
        save_pts = config['save_pts']
        save_pts.append(1e99)  # Add last endpoint

        # confgure optional intermediate printing
        if config['print']:
            def update_all():
                saved_times.append(t)
                saved_inputs.append({key: u_i[0] for key, u_i in zip(self.inputs, u)})
                saved_states.append({key: x_i[0] for key, x_i in zip(self.states, x_mat)})
                saved_outputs.append({key: z_i[0] for key, z_i in zip(self.outputs, self.output_matrix(x_mat))})
                saved_event_states.append({key: es_i[0] for key, es_i in zip(self.events, self.event_state_matrix(x_mat))})
                print("Time: {}\n\tInput: {}\n\tState: {}\n\tOutput: {}\n\tEvent State: {}\n"\
                    .format(
                        saved_times[-1],
                        saved_inputs[-1],
                        saved_states[-1],
                        saved_outputs[-1],
                        saved_event_states[-1]))  
        else:
            def update_all():
                saved_times.append(t)
                saved_inputs.append({key: u_i[0] for key, u_i in zip(self.inputs, u)})
                saved_states.append({key: x_i[0] for key, x_i in zip(self.states, x_mat)})

        # configuring next_time function to define prediction time step, default is constant dt
        if callable(config['dt']):
            next_time = config['dt']
        else:
            dt = config['dt']  # saving to optimize access in while loop
            def next_time(t, x):
                return dt

        # Convert to mat
        x_mat = np.vstack(tuple(x[key] for key in self.states))
        
        # Simulate
        update_all()
        if progress:
            simulate_progress = ProgressBar(100, "Progress")
            last_percentage = 0
       
        while t < horizon:
            dt = next_time(t, x_mat) 
            # TODO(CT): next_time for non matrix model could be a dict- figure out how to handle this
            t = t + dt
            u = future_loading_eqn(t)
            # TODO(CT): Future load with state
            x_mat = next_state(x_mat, u, dt)
            if (t >= next_save):
                next_save += save_freq
                update_all()
            if (t >= save_pts[save_pt_index]):
                save_pt_index += 1
                update_all()
            if config['progress']:
                percentages = [1-val for val in event_state(x_mat).values()]
                percentages.append((t/horizon))
                converted_iteration = int(max(min(100, max(percentages)*100), 0))
                if converted_iteration - last_percentage > 1:
                    simulate_progress(converted_iteration)
                    last_percentage = converted_iteration

            if check_thresholds(thresthold_met_eqn(x_mat)):
                break
        
        # Save final state
        if saved_times[-1] != t:
            # This check prevents double recording when the last state was a savepoint
            update_all()
        
        if not saved_outputs:
            # saved_outputs is empty, so it wasn't calculated in simulation - used cached result
            saved_outputs = LazySimResult(self.output, saved_times, saved_states) 
            saved_event_states = LazySimResult(self.event_state, saved_times, saved_states)
        else:
            saved_outputs = SimResult(saved_times, saved_outputs)
            saved_event_states = SimResult(saved_times, saved_event_states)
        
        return self.SimulationResults(
            saved_times, 
            SimResult(saved_times, saved_inputs), 
            SimResult(saved_times, saved_states), 
            saved_outputs, 
            saved_event_states
        )
