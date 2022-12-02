# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from abc import abstractmethod, ABC
from collections import abc, namedtuple
from copy import deepcopy
import itertools
import json
from numbers import Number
import numpy as np
from typing import Callable, Iterable, List
from warnings import warn

from .exceptions import ProgModelInputException, ProgModelTypeError, ProgModelException, ProgModelStateLimitWarning
from .sim_result import SimResult, LazySimResult
from .utils import ProgressBar
from .utils.containers import DictLikeMatrixWrapper
from .utils.parameters import PrognosticsModelParameters
from .utils.serialization import *


class PrognosticsModel(ABC):
    """
    A general time-variant state space :term:`model` of system degradation behavior.

    The PrognosticsModel class is a wrapper around a mathematical model of a system as represented by a state, output, input, event_state and threshold equation.

    A Model also has a parameters structure, which contains fields for various model parameters.

    Keyword Args
    ------------
        process_noise : Optional, float or dict[str, float]
          :term:`Process noise<process noise>` (applied at dx/next_state).
          Can be number (e.g., .2) applied to every state, a dictionary of values for each
          state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        process_noise_dist : Optional, str
          distribution for :term:`process noise` (e.g., normal, uniform, triangular)
        measurement_noise : Optional, float or dict[str, float]
          :term:`Measurement noise<measurement noise>` (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of values for each
          output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        measurement_noise_dist : Optional, str
          distribution for :term:`measurement noise` (e.g., normal, uniform, triangular)

    Additional parameters specific to the model

    Raises
    ------
        ProgModelTypeError, ProgModelInputException, ProgModelException 

    Example
    -------
        m = PrognosticsModel(process_noise = 3.2)

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
        param_callbacks : dict[str, list[function]], optional
            Callbacks for derived parameters
        inputs: list[str]
            Identifiers for each :term:`input`
        states: list[str]
            Identifiers for each :term:`state`
        outputs: list[str]
            Identifiers for each :term:`output`
        performance_metric_keys: list[str], optional
            Identifiers for each performance metric
        events: list[str], optional
            Identifiers for each :term:`event` predicted
        StateContainer : DictLikeMatrixWrapper
            Class for state container - used for representing :term:`state`
        OutputContainer : DictLikeMatrixWrapper
            Class for output container - used for representing :term:`output`
        InputContainer : DictLikeMatrixWrapper
            Class for input container - used for representing :term:`input`
    """
    is_vectorized = False

    # Configuration Parameters for model
    default_parameters = {
        'process_noise': 0.1,
        'measurement_noise': 0.0
    }

    # Configurable state range limit
    state_limits = {
        # 'state': (lower_limit, upper_limit)
    }

    # inputs = []     # Identifiers for each input
    # states = []     # Identifiers for each state
    # outputs = []    # Identifiers for each output
    # performance_metric_keys = []  # Identifies for each performance metric
    # events = []       # Identifiers for each event
    param_callbacks = {}  # Callbacks for derived parameters

    SimulationResults = namedtuple('SimulationResults', ['times', 'inputs', 'states', 'outputs', 'event_states'])

    def __init__(self, **kwargs):
        # Default params for any model
        params = PrognosticsModel.default_parameters.copy()

        # Add params specific to the model
        params.update(self.__class__.default_parameters) 

        # Add params specific passed via command line arguments
        try:
            params.update(kwargs)
        except TypeError:
            raise ProgModelTypeError("couldn't update parameters. Check that all parameters are valid")

        PrognosticsModel.__setstate__(self, params)

    def __eq__(self, other : "PrognosticsModel") -> bool:
        """
        Check if two models are equal
        """
        return self.__class__ == other.__class__ and self.parameters == other.parameters

    def __str__(self) -> str:
        return "{} Prognostics Model (Events: {})".format(type(self).__name__, self.events)

    def __getstate__(self) -> dict:
        return self.parameters.data

    def __setstate__(self, params : dict) -> None:
        # This method is called when depickling and in construction. It builds the model from the parameters

        if not hasattr(self, 'inputs'):
            self.inputs = []
        self.n_inputs = len(self.inputs)

        if not hasattr(self, 'states'):
            raise ProgModelTypeError('Must have `states` attribute')
        if len(self.states) <= 0:
            raise ProgModelTypeError('`states` attribute must have at least one state key')
        try:
            iter(self.states)
        except TypeError:
            raise ProgModelTypeError('model.states must be iterable')
        self.n_states = len(self.states)

        if not hasattr(self, 'events'):
            self.events = []
        self.n_events = len(self.events)

        if not hasattr(self, 'outputs'):
            raise ProgModelTypeError('Must have `outputs` attribute')
        try:
            iter(self.outputs)
        except TypeError:
            raise ProgModelTypeError('model.outputs must be iterable')
        self.n_outputs = len(self.outputs)

        if not hasattr(self, 'performance_metric_keys'):
            self.performance_metric_keys = []
        self.n_performance = len(self.performance_metric_keys)

        # Setup Containers
        # These containers should be used instead of dictionaries for models that use the internal matrix state
        states = self.states

        class StateContainer(DictLikeMatrixWrapper):
            def __init__(self, data):
                super().__init__(states, data)
        self.StateContainer = StateContainer

        inputs = self.inputs

        class InputContainer(DictLikeMatrixWrapper):
            def __init__(self, data):
                super().__init__(inputs, data)
        self.InputContainer = InputContainer

        outputs = self.outputs

        class OutputContainer(DictLikeMatrixWrapper):
            def __init__(self, data):
                super().__init__(outputs, data)
        self.OutputContainer = OutputContainer

        self.parameters = PrognosticsModelParameters(self, params, self.param_callbacks)

    def initialize(self, u = None, z = None):
        """
        Calculate initial state given inputs and outputs. If not defined for a model, it will return parameters['x0']

        Parameters
        ----------
        u : InputContainer
            Inputs, with keys defined by model.inputs \n
            e.g., u = m.InputContainer({'i':3.2}) given inputs = ['i']
        z : OutputContainer
            Outputs, with keys defined by model.outputs \n
            e.g., z = m.OutputContainer({'t':12.4, 'v':3.3}) given outputs = ['t', 'v']

        Returns
        -------
        x : StateContainer
            First state, with keys defined by model.states \n
            e.g., x = StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Example
        -------
            :
                m = PrognosticsModel() # Replace with specific model being simulated
                u = {'u1': 3.2}
                z = {'z1': 2.2}
                x = m.initialize(u, z) # Initialize first state
        """
        return self.StateContainer(self.parameters['x0'])

    def apply_measurement_noise(self, z):
        """
        Apply measurement noise to the measurement

        Parameters
        ----------
        z : OutputContainer
            output, with keys defined by model.outputs \n
            e.g., z = m.OutputContainer({'abc': 332.1, 'def': 221.003}) given outputs = ['abc', 'def']

        Returns
        -------
        z : OutputContainer
            output, with applied noise, with keys defined by model.outputs \n
            e.g., z = m.OutputContainer({'abc': 332.2, 'def': 221.043}) given outputs = ['abc', 'def']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | z = m.OutputContainer({'z1': 2.2})
        | z = m.apply_measurement_noise(z)

        Note
        ----
        Configured using parameters `measurement_noise` and `measurement_noise_dist`
        """
        z.matrix += np.random.normal(0, self.parameters['measurement_noise'].matrix, size=z.matrix.shape)
        return z

    def apply_process_noise(self, x, dt : int = 1):
        """
        Apply process noise to the state

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
        dt : float, optional
            Time step (e.g., dt = 0.1)

        Returns
        -------
        x : StateContainer
            state, with applied noise, with keys defined by model.states
            e.g., x = m.StateContainer({'abc': 332.2, 'def': 221.043}) given states = ['abc', 'def']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = m.InputContainer({'u1': 3.2})
        | z = m.OutputContainer({'z1': 2.2})
        | x = m.initialize(u, z) # Initialize first state
        | x = m.apply_process_noise(x)

        Note
        ----
        Configured using parameters `process_noise` and `process_noise_dist`
        """
        x.matrix += dt*np.random.normal(0, self.parameters['process_noise'].matrix, size=x.matrix.shape)
        return x

    def dx(self, x, u):
        """
        Calculate the first derivative of state `x` at a specific time `t`, given state and input

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
        u : InputContainer
            Inputs, with keys defined by model.inputs \n
            e.g., u = m.InputContainer({'i':3.2}) given inputs = ['i']

        Returns
        -------
        dx : StateContainer
            First derivitive of state, with keys defined by model.states \n
            e.g., dx = m.StateContainer({'abc': 3.1, 'def': -2.003}) given states = ['abc', 'def']

        Example
        -------
        | m = DerivProgModel() # Replace with specific model being simulated
        | u = m.InputContainer({'u1': 3.2})
        | z = m.OutputContainer({'z1': 2.2})
        | x = m.initialize(u, z) # Initialize first state
        | dx = m.dx(x, u) # Returns first derivative of state given input u

        See Also
        --------
        next_state

        Note
        ----
        A model should overwrite either `next_state` or `dx`. Override `dx` for continuous models,
        and `next_state` for discrete, where the behavior cannot be described by the first derivative
        """
        raise ProgModelException('dx not defined - please use next_state()')

    def next_state(self, x, u, dt : float):
        """
        State transition equation: Calculate next state

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
        u : InputContainer
            Inputs, with keys defined by model.inputs \n
            e.g., u = m.InputContainer({'i':3.2}) given inputs = ['i']
        dt : float
            Timestep size in seconds (≥ 0) \n
            e.g., dt = 0.1

        Returns
        -------
        x : StateContainer
            Next state, with keys defined by model.states
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = m.InputContainer({'u1': 3.2})
        | z = m.OutputContainer({'z1': 2.2})
        | x = m.initialize(u, z) # Initialize first state
        | x = m.next_state(x, u, 0.1) # Returns state at 3.1 seconds given input u

        See Also
        --------
        dx

        Note
        ----
        A model should overwrite either `next_state` or `dx`. Override `dx` for continuous models, and `next_state` for discrete, where the behavior cannot be described by the first derivative
        """
        # Note: Default is to use the dx method (continuous model) - overwrite next_state for continuous
        dx = self.dx(x, u)
        if isinstance(x, DictLikeMatrixWrapper) and isinstance(dx, DictLikeMatrixWrapper):
            return self.StateContainer(x.matrix + dx.matrix * dt)
        elif isinstance(dx, dict) or isinstance(x, dict):
            return self.StateContainer({key: x[key] + dx[key]*dt for key in dx.keys()})
        else:
            raise ValueError(f"ValueError: dx return must be of type StateContainer, was {type(dx)}")

    def apply_limits(self, x):
        """
        Apply state bound limits. Any state outside of limits will be set to the closest limit.

        Parameters
        ----------
        x : StateContainer or dict
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Returns
        -------
        x : StateContainer or dict
            Bounded state, with keys defined by model.states
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
        """
        for (key, limit) in self.state_limits.items():
            if x[key] < limit[0]:
                warn("State {} limited to {} (was {})".format(key, limit[0], x[key]), ProgModelStateLimitWarning)
                x[key] = limit[0]
            elif x[key] > limit[1]:
                warn("State {} limited to {} (was {})".format(key, limit[1], x[key]), ProgModelStateLimitWarning)
                x[key] = limit[1]
        return x

    def __next_state(self, x, u, dt : float):
        """
        State transition equation: Calls next_state(), calculating the next state, and then adds noise

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
        u : InputContainer
            Inputs, with keys defined by model.inputs \n
            e.g., u = m.InputContainer({'i':3.2}) given inputs = ['i']
        dt : float
            Timestep size in seconds (≥ 0) \n
            e.g., dt = 0.1

        Returns
        -------
        x : StateContainer
            Next state, with keys defined by model.states
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = m.InputContainer({'u1': 3.2})
        | z = m.OutputContainer({'z1': 2.2})
        | x = m.initialize(u, z) # Initialize first state
        | x = m.__next_state(x, u, 0.1) # Returns state, with noise, at 3.1 seconds given input u

        See Also
        --------
        next_state

        Note
        ----
        A model should not overwrite '__next_state'
        A model should overwrite either `next_state` or `dx`. Override `dx` for continuous models, and `next_state` for discrete, where the behavior cannot be described by the first derivative.
        """
        # Calculate next state and add process noise
        next_state = self.apply_process_noise(self.next_state(x, u, dt), dt)

        # Apply Limits
        return self.apply_limits(next_state)

    def performance_metrics(self, x) -> dict:
        """
        Calculate performance metrics where

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
        
        Returns
        -------
        pm : dict
            Performance Metrics, with keys defined by model.performance_metric_keys. \n
            e.g., pm = {'tMax':33, 'iMax':19} given performance_metric_keys = ['tMax', 'iMax']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = m.InputContainer({'u1': 3.2})
        | z = m.OutputContainer({'z1': 2.2})
        | x = m.initialize(u, z) # Initialize first state
        | pm = m.performance_metrics(x) # Returns {'tMax':33, 'iMax':19}
        """
        return {}

    observables = performance_metrics  # For backwards compatibility

    @abstractmethod
    def output(self, x):
        """
        Calculate outputs given state

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Returns
        -------
        z : OutputContainer
            Outputs, with keys defined by model.outputs. \n
            e.g., z = m.OutputContainer({'t':12.4, 'v':3.3}) given outputs = ['t', 'v']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = m.InputContainer({'u1': 3.2})
        | z = m.OutputContainer({'z1': 2.2})
        | x = m.initialize(u, z) # Initialize first state
        | z = m.output(x) # Returns m.OutputContainer({'z1': 2.2})
        """
        return {}

    def __output(self, x):
        """
        Calls output, which calculates next state forward one timestep, and then adds noise

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Returns
        -------
        z : OutputContainer
            Outputs, with keys defined by model.outputs. \n
            e.g., z = m.OutputContainer({'t':12.4, 'v':3.3} )given outputs = ['t', 'v']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = m.InputContainer({'u1': 3.2})
        | z = m.OutputContainer({'z1': 2.2})
        | x = m.initialize(u, z) # Initialize first state
        | z = m.__output(3.0, x) # Returns {'o1': 1.2} with noise added
        """

        # Calculate next state, forward one timestep
        z = self.output(x)

        # Add measurement noise
        return self.apply_measurement_noise(z)

    def event_state(self, x) -> dict:
        """
        Calculate event states (i.e., measures of progress towards event (0-1, where 0 means event has occurred))

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states\n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Returns
        -------
        event_state : dict
            Event States, with keys defined by prognostics_model.events.\n
            e.g., event_state = {'EOL':0.32} given events = ['EOL']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = m.InputContainer({'u1': 3.2})
        | z = m.OutputContainer({'z1': 2.2})
        | x = m.initialize(u, z) # Initialize first state
        | event_state = m.event_state(x) # Returns {'e1': 0.8, 'e2': 0.6}

        Note
        ----
        If not overridden, will return 0.0 if threshold_met returns True, otherwise 1.0. If neither threshold_met or event_state is overridden, will return an empty dictionary (i.e., no events)

        See Also
        --------
        threshold_met
        """
        if type(self).threshold_met == PrognosticsModel.threshold_met:
            # Neither Threshold Met nor Event States are overridden
            return {}

        return {key: 1.0-float(t_met) \
            for (key, t_met) in self.threshold_met(x).items()} 
    
    def threshold_met(self, x) -> dict:
        """
        For each event threshold, calculate if it has been met

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states\n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
        
        Returns
        -------
        thresholds_met : dict
            If each threshold has been met (bool), with keys defined by prognostics_model.events\n
            e.g., thresholds_met = {'EOL': False} given events = ['EOL']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = m.InputContainer({'u1': 3.2})
        | z = m.OutputContainer({'z1': 2.2})
        | x = m.initialize(u, z) # Initialize first state
        | threshold_met = m.threshold_met(x) # returns {'e1': False, 'e2': False}

        Note
        ----
        If not overridden, will return True if event_state is <= 0, otherwise False. If neither threshold_met or event_state is overridden, will return an empty dictionary (i.e., no events)
        
        See Also
        --------
        event_state
        """
        if type(self).event_state == PrognosticsModel.event_state:
            # Neither Threshold Met nor Event States are overridden
            return {}

        return {key: event_state <= 0 \
            for (key, event_state) in self.event_state(x).items()} 

    def simulate_to(self, time : float, future_loading_eqn : Callable, first_output = None, **kwargs) -> namedtuple:
        """
        Simulate prognostics model for a given number of seconds

        Parameters
        ----------
        time : float
            Time to which the model will be simulated in seconds (≥ 0.0) \n
            e.g., time = 200
        future_loading_eqn : callable
            Function of (t) -> z used to predict future loading (output) at a given time (t)
        first_output : OutputContainer, optional
            First measured output, needed to initialize state for some classes. Can be omitted for classes that don't use this
        options: kwargs, optional
            Configuration options for the simulation \n
            Note: configuration of the model is set through model.parameters \n
            Supported parameters: see `simulate_to_threshold`
        
        Returns
        -------
        times: list[float]
            Times for each simulated point
        inputs: SimResult
            Future input (from future_loading_eqn) for each time in times
        states: SimResult
            Estimated states for each time in times
        outputs: SimResult
            Estimated outputs for each time in times
        event_states: SimResult
            Estimated event state (e.g., SOH), between 1-0 where 0 is event occurrence, for each time in times
        
        Raises
        ------
        ProgModelInputException

        See Also
        --------
        simulate_to_threshold

        Example
        -------
        | def future_load_eqn(t):
        |     if t< 5.0: # Load is 3.0 for first 5 seconds
        |         return 3.0
        |     else:
        |         return 5.0
        | first_output = m.OutputContainer({'o1': 3.2, 'o2': 1.2})
        | m = PrognosticsModel() # Replace with specific model being simulated
        | (times, inputs, states, outputs, event_states) = m.simulate_to(200, future_load_eqn, first_output)
        """
        # Input Validation
        if not isinstance(time, Number) or time < 0:
            raise ProgModelInputException("'time' must be positive, was {} (type: {})".format(time, type(time)))

        # Configure
        config = { # Defaults
            'thresholds_met_eqn': (lambda x: False), # Override threshold
            'horizon': time
        }
        kwargs.update(config) # Config should override kwargs

        return self.simulate_to_threshold(future_loading_eqn, first_output, **kwargs)
 
    def simulate_to_threshold(self, future_loading_eqn : Callable, first_output = None, threshold_keys : list = None, **kwargs) -> namedtuple:
        """
        Simulate prognostics model until any or specified threshold(s) have been met

        Parameters
        ----------
        future_loading_eqn : callable
            Function of (t) -> z used to predict future loading (output) at a given time (t)

        Keyword Arguments
        -----------------
        t0 : float, optional
            Starting time for simulation in seconds (default: 0.0) \n
        dt : float, tuple, str, or function, optional
            float: constant time step (s), e.g. dt = 0.1\n
            function (t, x) -> dt\n
            tuple: (mode, dt), where modes could be constant or auto. If auto, dt is maximum step size\n
            str: mode - 'auto' or 'constant'\n
        integration_method: str, optional
            Integration method, e.g. 'rk4' or 'euler' (default: 'euler')
        save_freq : float, optional
            Frequency at which output is saved (s), e.g., save_freq = 10 \n
        save_pts : list[float], optional
            Additional ordered list of custom times where output is saved (s), e.g., save_pts= [50, 75] \n
        horizon : float, optional
            maximum time that the model will be simulated forward (s), e.g., horizon = 1000 \n
        first_output : OutputContainer, optional
            First measured output, needed to initialize state for some classes. Can be omitted for classes that don't use this
        threshold_keys: list[str] or str, optional
            Keys for events that will trigger the end of simulation.
            If blank, simulation will occur if any event will be met ()
        x : StateContainer, optional
            initial state dict, e.g., x= m.StateContainer({'x1': 10, 'x2': -5.3})\n
        thresholds_met_eqn : function/lambda, optional
            custom equation to indicate logic for when to stop sim f(thresholds_met) -> bool\n
        print : bool, optional
            toggle intermediate printing, e.g., print = True\n
            e.g., m.simulate_to_threshold(eqn, z, dt=0.1, save_pts=[1, 2])
        progress : bool, optional
            toggle progress bar printing, e.g., progress = True\n
    
        Returns
        -------
        times: list[float]
            Times for each simulated point
        inputs: SimResult
            Future input (from future_loading_eqn) for each time in times
        states: SimResult
            Estimated states for each time in times
        outputs: SimResult
            Estimated outputs for each time in times
        event_states: SimResult
            Estimated event state (e.g., SOH), between 1-0 where 0 is event occurrence, for each time in times
        
        Raises
        ------
        ProgModelInputException

        See Also
        --------
        simulate_to

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | def future_load_eqn(t):
        |     if t< 5.0: # Load is 3.0 for first 5 seconds
        |         return m.InputContainer({'load': 3.0})
        |     else:
        |         return m.InputContainer({'load': 5.0})
        | first_output = m.OutputContainer({'o1': 3.2, 'o2': 1.2})
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
            'dt': ('auto', 1.0),
            'integration_method': 'euler',
            'save_pts': [],
            'save_freq': 10.0,
            'horizon': 1e100, # Default horizon (in s), essentially inf
            'print': False,
            'progress': False
        }
        config.update(kwargs)
        
        # Configuration validation
        if not isinstance(config['dt'], (Number, tuple, str)) and not callable(config['dt']):
            raise ProgModelInputException("'dt' must be a number or function, was a {}".format(type(config['dt'])))
        if isinstance(config['dt'], Number) and config['dt'] < 0:
            raise ProgModelInputException("'dt' must be positive, was {}".format(config['dt']))
        if not isinstance(config['save_freq'], Number) and not isinstance(config['save_freq'], tuple):
            raise ProgModelInputException("'save_freq' must be a number, was a {}".format(type(config['save_freq'])))
        if (isinstance(config['save_freq'], Number) and config['save_freq'] <= 0) or \
            (isinstance(config['save_freq'], tuple) and config['save_freq'][1] <= 0):
            raise ProgModelInputException("'save_freq' must be positive, was {}".format(config['save_freq']))
        if not isinstance(config['save_pts'], abc.Iterable):
            raise ProgModelInputException("'save_pts' must be list or array, was a {}".format(type(config['save_pts'])))
        if not isinstance(config['horizon'], Number):
            raise ProgModelInputException("'horizon' must be a number, was a {}".format(type(config['horizon'])))
        if config['horizon'] < 0:
            raise ProgModelInputException("'horizon' must be positive, was {}".format(config['horizon']))
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
            x = deepcopy(config['x'])
        else:
            x = self.initialize(u, first_output)

        if not isinstance(x, self.StateContainer):
            x = self.StateContainer(x)
        
        # Optimization
        next_state = self.__next_state
        output = self.__output
        threshold_met_eqn = self.threshold_met
        event_state = self.event_state
        load_eqn = future_loading_eqn
        progress = config['progress']

        # Threshold Met Equations
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
        elif len(threshold_keys) == 0:
            check_thresholds = lambda _: False

        # Initialization of save arrays
        saved_times = []
        saved_inputs = []
        saved_states = []  
        saved_outputs = []
        saved_event_states = []
        horizon = t+config['horizon']
        if isinstance(config['save_freq'], tuple):
            # Tuple used to specify start and frequency
            t_step = config['save_freq'][1]
            # Use starting time or the next multiple
            t_start = config['save_freq'][0]
            start = max(t_start, t - (t-t_start)%t_step)
            iterator = itertools.count(start, t_step)
        else:
            # Otherwise - start is t0
            t_step = config['save_freq']
            iterator = itertools.count(t, t_step)
        next(iterator) # Skip current time
        next_save = next(iterator)
        save_pt_index = 0
        save_pts = config['save_pts']
        save_pts.append(1e99)  # Add last endpoint

        # confgure optional intermediate printing
        if config['print']:
            def update_all():
                saved_times.append(t)
                saved_inputs.append(u)
                saved_states.append(deepcopy(x))  # Avoid optimization where x is not copied
                saved_outputs.append(output(x))
                saved_event_states.append(event_state(x))
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
                saved_inputs.append(u)
                saved_states.append(deepcopy(x))  # Avoid optimization where x is not copied

        # configuring next_time function to define prediction time step, default is constant dt
        if callable(config['dt']):
            next_time = config['dt']
            dt_mode = 'function'
        elif isinstance(config['dt'], tuple):
            dt_mode = config['dt'][0]
            dt = config['dt'][1]                
        elif isinstance(config['dt'], str):
            dt_mode = config['dt']
            if dt_mode == 'constant':
                dt = 1.0  # Default
            else:
                dt = np.inf
        else:
            dt_mode = 'constant'
            dt = config['dt']  # saving to optimize access in while loop

        if dt_mode == 'constant':
            def next_time(t, x):
                return dt
        elif dt_mode == 'auto':
            def next_time(t, x):
                return min(dt, next_save-t, save_pts[save_pt_index]-t)
        elif dt_mode != 'function':
            raise ProgModelInputException(f"'dt' mode {dt_mode} not supported. Must be 'constant', 'auto', or a function")
        
        # Auto Container wrapping
        dt0 = next_time(t, x) - t
        if not isinstance(u, DictLikeMatrixWrapper):
            # Wrapper around the future loading equation
            def load_eqn(t, x):
                u = future_loading_eqn(t, x)
                return self.InputContainer(u)

        if not isinstance(self.next_state(x.copy(), u, dt0), DictLikeMatrixWrapper):
            # Wrapper around next_state
            def next_state(x, u, dt):
                # Calculate next state, and convert
                x_new = self.next_state(x, u, dt)
                x_new = self.StateContainer(x_new)

                # Calculate next state and add process noise
                next_state = self.apply_process_noise(x_new, dt)

                # Apply Limits
                return self.apply_limits(next_state)

        if not isinstance(self.output(x), DictLikeMatrixWrapper):
            # Wrapper around the output equation
            def output(x):
                # Calculate output, convert to outputcontainer
                z = self.output(x)
                z = self.OutputContainer(z)

                # Add measurement noise
                return self.apply_measurement_noise(z)

        # Simulate
        update_all()
        if progress:
            simulate_progress = ProgressBar(100, "Progress")
            last_percentage = 0

        if config['integration_method'].lower() == 'rk4':
            # Using RK4 Method
            dx = self.dx

            try:
                dx(x, load_eqn(t, x))
            except ProgModelException:
                raise ProgModelException("dx(x, u) must be defined to use RK4 method")

            apply_limits = self.apply_limits
            apply_process_noise = self.apply_process_noise
            StateContainer = self.StateContainer
            def next_state(x, u, dt):
                dx1 = StateContainer(dx(x, u))
                
                x2 = StateContainer({key: x[key] + dt*dx_i/2 for key, dx_i in dx1.items()})
                dx2 = dx(x2, u)

                x3 = StateContainer({key: x[key] + dt*dx_i/2 for key, dx_i in dx2.items()})
                dx3 = dx(x3, u)
                
                x4 = StateContainer({key: x[key] + dt*dx_i for key, dx_i in dx3.items()})   
                dx4 = dx(x4, u)

                x = StateContainer({key: x[key]+ dt/3*(dx1[key]/2 + dx2[key] + dx3[key] + dx4[key]/2) for key in dx1.keys()})
                return apply_limits(apply_process_noise(x))
        elif config['integration_method'].lower() != 'euler':
            raise ProgModelInputException(f"'integration_method' mode {config['integration_method']} not supported. Must be 'euler' or 'rk4'")
       
        while t < horizon:
            dt = next_time(t, x)
            t = t + dt/2
            # Use state at midpoint of step to best represent the load during the duration of the step
            # This is sometimes referred to as 'leapfrog integration'
            u = load_eqn(t, x)
            t = t + dt/2
            x = next_state(x, u, dt)

            # Save if at appropriate time
            if (t >= next_save):
                next_save = next(iterator)
                update_all()
                if (t >= save_pts[save_pt_index]):
                    # Prevent double saving when save_pt and save_freq align
                    save_pt_index += 1
            elif (t >= save_pts[save_pt_index]):
                save_pt_index += 1
                update_all()

            # Update progress bar
            if config['progress']:
                percentages = [1-val for val in event_state(x).values()]
                percentages.append((t/horizon))
                converted_iteration = int(max(min(100, max(percentages)*100), 0))
                if converted_iteration - last_percentage > 1:
                    simulate_progress(converted_iteration)
                    last_percentage = converted_iteration

            # Check thresholds
            if check_thresholds(threshold_met_eqn(x)):
                break
        
        # Save final state
        if saved_times[-1] != t:
            # This check prevents double recording when the last state was a savepoint
            update_all()
        
        if not saved_outputs:
            # saved_outputs is empty, so it wasn't calculated in simulation - used cached result
            saved_outputs = LazySimResult(self.__output, saved_times, saved_states) 
            saved_event_states = LazySimResult(self.event_state, saved_times, saved_states)
        else:
            saved_outputs = SimResult(saved_times, saved_outputs, _copy=False)
            saved_event_states = SimResult(saved_times, saved_event_states, _copy=False)
        
        return self.SimulationResults(
            saved_times, 
            SimResult(saved_times, saved_inputs, _copy=False), 
            SimResult(saved_times, saved_states, _copy=False), 
            saved_outputs, 
            saved_event_states
        )
    
    @staticmethod
    def generate_model(keys : dict, initialize_eqn : Callable, output_eqn : Callable, next_state_eqn : Callable = None, dx_eqn : Callable = None, event_state_eqn : Callable = None, threshold_eqn : Callable = None, config : dict = {'process_noise': 0.1}) -> "PrognosticsModel":
        """
        [DEPRECIATED] Generate a new prognostics model from individual model functions

        Parameters
        ----------
        keys : dict
            Dictionary containing keys required by model. Must include `inputs`, `outputs`, and `states`. Can also include `events`
        initialize_eqn : callable
            Equation to initialize first state of the model. See `initialize`
        output_eqn : callable
            Equation to calculate the outputs (measurements) for the model. See `output`
        next_state_eqn : callable
            Equation to calculate next_state from current state. See `next_state`.\n
            Use this for discrete functions
        dx_eqn : callable
            Equation to calculate dx from current state. See `dx`. \n
            Use this for continuous functions
        event_state_eqn : callable, optional
            Equation to calculate the state for each event of the model. See `event_state`
        threshold_eqn : callable, optional
            Equation to calculate if the threshold has been met for each event in model. See `threshold_met`
        config : dict, optional
            Any configuration parameters for the model

        Returns
        -------
        model : PrognosticsModel
            A callable PrognosticsModel

        Raises
        ------
        ProgModelInputException

        Example
        -------
        | keys = {
        |     'inputs': ['u1', 'u2'],
        |     'states': ['x1', 'x2', 'x3'],
        |     'outputs': ['z1'],
        |     'events': ['e1', 'e2']
        | }
        |
        | m = PrognosticsModel.generate_model(keys, initialize_eqn, next_state_eqn, output_eqn, event_state_eqn, threshold_eqn)
        """
        warn("PrognosticsModel.generate_model is deprecated and will be removed in v1.5. Subclass PrognosticsModel instead", DeprecationWarning)

        # Input validation
        if not callable(initialize_eqn):
            raise ProgModelTypeError("Initialize Function must be callable")

        if not callable(output_eqn):
            raise ProgModelTypeError("Output Function must be callable")

        if next_state_eqn and not callable(next_state_eqn):
            raise ProgModelTypeError("Next_State Function must be callable")

        if dx_eqn and not callable(dx_eqn):
            raise ProgModelTypeError("dx Function must be callable")

        if not next_state_eqn and not dx_eqn:
            raise ProgModelTypeError("Either next_state or dx must be defined (but not both)")

        if next_state_eqn and dx_eqn:
            raise ProgModelTypeError("Either next_state or dx must be defined (but not both)")

        if event_state_eqn and not callable(event_state_eqn):
            raise ProgModelTypeError("Event State Function must be callable")

        if threshold_eqn and not callable(threshold_eqn):
            raise ProgModelTypeError("Threshold Function must be callable")

        if 'inputs' not in keys:
            raise ProgModelTypeError("Keys must include 'inputs'")
        
        if 'states' not in keys:
            raise ProgModelTypeError("Keys must include 'states'")
        
        if 'outputs' not in keys:
            raise ProgModelTypeError("Keys must include 'outputs'")

        # Construct model
        class NewProgModel(PrognosticsModel):
            inputs = keys['inputs']
            states = keys['states']
            outputs = keys['outputs']
            
            def initialize():
                pass

            def output():
                pass

        m = NewProgModel(**config)
        m.initialize = initialize_eqn
        m.output = output_eqn

        if next_state_eqn:
            m.next_state = next_state_eqn
        if dx_eqn:
            m.dx = dx_eqn
        if 'events' in keys:
            m.events = keys['events']
        if event_state_eqn:
            m.event_state = event_state_eqn
        if threshold_eqn:
            m.threshold_met = threshold_eqn

        return m

    def calc_error(self, times : List[float], inputs : List[dict], outputs : List[dict], **kwargs) -> float:
        """Calculate Mean Squared Error (MSE) between simulated and observed

        Args:
            times (list[float]): array of times for each sample
            inputs (list[dict]): array of input dictionaries where input[x] corresponds to time[x]
            outputs (list[dict]): array of output dictionaries where output[x] corresponds to time[x]
        
        Keyword Args:
            x0 (dict, optional): Initial state
            dt (double, optional): time step

        Returns:
            double: Total error
        """
        if isinstance(times[0], Iterable):
            # Calculate error for each
            error = [self.calc_error(t, i, z, **kwargs) for (t, i, z) in zip(times, inputs, outputs)]
            return sum(error)/len(error)

        x = kwargs.get('x0', self.initialize(inputs[0], outputs[0]))
        dt = kwargs.get('dt', 1e99)

        if not isinstance(x, self.StateContainer):
            x = [self.StateContainer(x_i) for x_i in x]

        if not isinstance(inputs[0], self.InputContainer):
            inputs = [self.InputContainer(u_i) for u_i in inputs]
        
        if not isinstance(outputs[0], self.OutputContainer):
            outputs = [self.OutputContainer(z_i) for z_i in outputs]

        counter = 0  # Needed to account for skipped (i.e., none) values
        t_last = times[0]
        err_total = 0
        z_obs = self.output(x)  # Initialize
        for t, u, z in zip(times, inputs, outputs):
            while t_last < t:
                t_new = min(t_last + dt, t)
                x = self.next_state(x, u, t_new-t_last)
                t_last = t
                if t >= t_last:
                    # Only recalculate if required
                    z_obs = self.output(x)
            if not (None in z_obs.matrix or None in z.matrix):
                if any(np.isnan(z_obs.matrix)):
                    warn("Model unstable- NaN reached in simulation (t={})".format(t))
                    break
                err_total += np.sum(np.square(z.matrix - z_obs.matrix), where= ~np.isnan(z.matrix))
                counter += 1

        return err_total/counter
    
    def estimate_params(self, runs : List[tuple], keys : List[str], **kwargs) -> None:
        """Estimate the model parameters given data. Overrides model parameters

        Args:
            runs (list[tuple]): data from all runs, where runs[0] is the data from run 0. Each run consists of a tuple of arrays of times, input dicts, and output dicts
            keys (list[str]): Parameter keys to optimize
        
        Keyword Args: 
            method (str, optional): Optimization method- see scipy.optimize.minimize for options
            bounds (tuple): Bounds for optimization in format ((lower1, upper1), (lower2, upper2), ...)
            options (dict): Options passed to optimizer. see scipy.optimize.minimize for options

        See: examples.param_est
        """
        from scipy.optimize import minimize

        config = {
            'method': 'nelder-mead',  # Optimization method
            'bounds': ((-np.inf, np.inf) for _ in keys),
            'options': {'xatol': 1e-8}  # Options passed to optimizer
        }
        config.update(kwargs)

        if 'x0' in kwargs and not isinstance(kwargs['x0'], self.StateContainer):
            # Convert here so it isn't done every call of calc_error
            kwargs['x0'] = [self.StateContainer(x_i) for x_i in kwargs['x0']]

        # Set noise to 0
        m_noise, self.parameters['measurement_noise'] = self.parameters['measurement_noise'], 0
        p_noise, self.parameters['process_noise'] = self.parameters['process_noise'], 0

        for i, (times, inputs, outputs) in enumerate(runs):
            has_changed = False
            if not isinstance(inputs[0], self.InputContainer):
                inputs = [self.InputContainer(u_i) for u_i in inputs]
                has_changed = True

            if isinstance(outputs, np.ndarray):
                outputs = [self.OutputContainer(u_i) for u_i in outputs]
                has_changed = True

            if has_changed:
                runs[i] = (times, inputs, outputs)

        def optimization_fcn(params):
            for key, param in zip(keys, params):
                self.parameters[key] = param
            err = 0
            for run in runs:
                try:
                    err += self.calc_error(run[0], run[1], run[2], **kwargs)
                except Exception:
                    return 1e99 
                    # If it doesn't work (i.e., throws an error), don't use it
            return err
        
        params = np.array([self.parameters[key] for key in keys])

        res = minimize(optimization_fcn, params, method=config['method'], bounds = config['bounds'], options=config['options'])
        for x, key in zip(res.x, keys):
            self.parameters[key] = x

        # Reset noise
        self.parameters['measurement_noise'] = m_noise
        self.parameters['process_noise'] = p_noise   

    def generate_surrogate(self, load_functions, method = 'dmd', **kwargs):
        """
        Generate a surrogate model to approximate the higher-fidelity model 

        Parameters
        ----------
        load_functions : list of callable functions
            Each index is a callable loading function of (t, x = None) -> z used to predict future loading (output) at a given time (t) and state (x)
        method : str, optional
            list[ indicating surrogate modeling method to be used 

        Keyword Arguments
        -----------------
        dt : float or function, optional
            Same as in simulate_to_threshold; for DMD, this value is the time step of the training data\n
        save_freq : float, optional
            Same as in simulate_to_threshold; for DMD, this value is the time step with which the surrogate model is generated  \n
        state_keys: list, optional
            List of state keys to be included in the surrogate model generation. keys must be a subset of those defined in the PrognosticsModel  \n
        input_keys: list, optional
            List of input keys to be included in the surrogate model generation. keys must be a subset of those defined in the PrognosticsModel  \n
        output_keys: list, optional
            List of output keys to be included in the surrogate model generation. keys must be a subset of those defined in the PrognosticsModel  \n
        event_keys: list, optional
            List of event_state keys to be included in the surrogate model generation. keys must be a subset of those defined in the PrognosticsModel  \n   
        ...: optional
            Keyword arguments from simulate_to_threshold (except save_pts)

        Returns
        -------
        SurrogateModel(): class
            Instance of SurrogateModel class

        Example
        -------
        See examples/generate_surrogate
        """

        from .data_models import SURROAGATE_METHOD_LOOKUP

        if method not in SURROAGATE_METHOD_LOOKUP.keys():
            raise ProgModelInputException("Method {} not supported. Supported methods: {}".format(method, SURROAGATE_METHOD_LOOKUP.keys()))

        # Configure
        config = { # Defaults
            'save_freq': 1.0, 
            'state_keys': self.states.copy(),
            'input_keys': self.inputs.copy(),
            'output_keys': self.outputs.copy(),
            'event_keys': self.events.copy(),
        }
        config.update(kwargs)

        if 'inputs' in config:
            warn("Use 'input_keys' instead of 'inputs'. 'inputs' will be deprecated in v1.5")
            config['input_keys'] = config['inputs']
            del config['inputs']
        if 'states' in config:
            warn("Use 'state_keys' instead of 'states'. 'states' will be deprecated in v1.5")
            config['state_keys'] = config['states']
            del config['states']
        if 'outputs' in config:
            warn("Use 'output_keys' instead of 'outputs'. 'outputs' will be deprecated in v1.5")
            config['output_keys'] = config['outputs']
            del config['outputs']
        if 'events' in config:
            warn("Use 'event_keys' instead of 'events'. 'events' will be deprecated in v1.5")
            config['event_keys'] = config['events']
            del config['events']

        # Validate user inputs 
        try:
            # Check if load_functions is list-like (i.e., iterable)
            iter(load_functions)
        except TypeError:
            raise ProgModelInputException(f"load_functions must be a list or list-like object, was {type(load_functions)}")
        if len(load_functions) <= 0:
            raise ProgModelInputException("load_functions must contain at least one element")
        if 'save_pts' in config.keys():
            raise ProgModelInputException("'save_pts' is not a valid input for DMD Surrogate Model.")

        if isinstance(config['input_keys'], str):
            config['input_keys'] = [config['input_keys']]
        if not all([x in self.inputs for x in config['input_keys']]):
            raise ProgModelInputException(f"Invalid 'input_keys' value ({config['input_keys']}), must be a subset of the model's inputs ({self.inputs}).")
        
        if isinstance(config['state_keys'], str):
            config['state_keys'] = [config['state_keys']]
        if not all([x in self.states for x in config['state_keys']]):
            raise ProgModelInputException(f"Invalid 'state_keys' input value ({config['state_keys']}), must be a subset of the model's states ({self.states}).")

        if isinstance(config['output_keys'], str):
            config['output_keys'] = [config['output_keys']]
        if not all([x in self.outputs for x in config['output_keys']]):
            raise ProgModelInputException(f"Invalid 'output_keys' input value ({config['output_keys']}), must be a subset of the model's outputs ({self.outputs}).")

        if isinstance(config['event_keys'], str):
            config['event_keys'] = [config['event_keys']]
        if not all([x in self.events for x in config['event_keys']]):
            raise ProgModelInputException(f"Invalid 'event_keys' input value ({config['event_keys']}), must be a subset of the model's events ({self.events}).")

        return SURROAGATE_METHOD_LOOKUP[method](self, load_functions, **config)
    
    def to_json(self):
        """
        Serialize parameters as JSON objects 

        Returns:
            JSON: Serialized PrognosticsModel parameters as JSON object

        See Also
        --------
        from_json

        Note
        ----
        This method only serializes the values of the prognostics model parameters (model.parameters)
        """
        return json.dumps(self.parameters.data, cls=CustomEncoder)
    
    @classmethod
    def from_json(cls, data):
        """
        Create a new prognostics model from a previously generated model that was serialized as a JSON object

        Args:
            data: 
                JSON serialized parameters necessary to build a model 
                See to_json method 

        Returns:
            PrognosticsModel: Model generated from serialized parameters 

        See Also
        ---------
        to_json

        Note
        ----
        This serialization only works for models that include all parameters necessary to generate the model in model.parameters. 
        """

        extract_parameters = json.loads(data, object_hook = custom_decoder)
 
        return cls(**extract_parameters)
