# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from typing import Callable, List
from .exceptions import ProgModelInputException, ProgModelTypeError, ProgModelException, ProgModelStateLimitWarning
from abc import abstractmethod, ABC
from numbers import Number
import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy
import itertools
from warnings import warn
from collections import abc, namedtuple
from .sim_result import SimResult, LazySimResult
from .utils import ProgressBar
from .utils.containers import DictLikeMatrixWrapper
from .utils.parameters import PrognosticsModelParameters


class PrognosticsModel(ABC):
    """
    A general time-variant state space model of system degradation behavior.

    The PrognosticsModel class is a wrapper around a mathematical model of a system as represented by a state, output, input, event_state and threshold equation.

    A Model also has a parameters structure, which contains fields for various model parameters.

    Keyword Args
    ------------
        process_noise : Optional, float or Dict[Srt, float]
          Process noise (applied at dx/next_state). 
          Can be number (e.g., .2) applied to every state, a dictionary of values for each 
          state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        process_noise_dist : Optional, String
          distribution for process noise (e.g., normal, uniform, triangular)
        measurement_noise : Optional, float or Dict[Srt, float]
          Measurement noise (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of values for each
          output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        measurement_noise_dist : Optional, String
          distribution for measurement noise (e.g., normal, uniform, triangular)
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
        StateContainer : DictLikeMatrixWrapper
            Class for state container - used for representing state
        OutputContainer : DictLikeMatrixWrapper
            Class for output container - used for representing output
        InputContainer : DictLikeMatrixWrapper
            Class for input container - used for representing input
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
    performance_metric_keys = []  # Identifies for each performance metric
    events = []       # Identifiers for each event
    param_callbacks = {}  # Callbacks for derived parameters

    observables_keys = performance_metric_keys # for backwards compatability        
    SimulationResults = namedtuple('SimulationResults', ['times', 'inputs', 'states', 'outputs', 'event_states'])

    def __init__(self, **kwargs):
        if not hasattr(self, 'inputs'):
            raise ProgModelTypeError('Must have `inputs` attribute')
        
        if not hasattr(self, 'states'):
            raise ProgModelTypeError('Must have `states` attribute')
        if len(self.states) <= 0:
            raise ProgModelTypeError('`states` attribute must have at least one state key')
        try:
            iter(self.states)
        except TypeError:
            raise ProgModelTypeError('model.states must be iterable')

        if not hasattr(self, 'outputs'):
            raise ProgModelTypeError('Must have `outputs` attribute')
        try:
            iter(self.outputs)
        except TypeError:
            raise ProgModelTypeError('model.outputs must be iterable')
        
        # Default params for any model
        params = PrognosticsModel.default_parameters.copy()

        # Add params specific to the model
        params.update(self.__class__.default_parameters) 

        # Add params specific passed via command line arguments
        try:
            params.update(kwargs)
        except TypeError:
            raise ProgModelTypeError("couldn't update parameters. Check that all parameters are valid")

        self.__setstate__(params)

    def __eq__(self, other : "PrognosticsModel") -> bool:
        """
        Check if two models are equal
        """
        return self.__class__ == other.__class__ and self.parameters == other.parameters
    
    def __str__(self) -> str:
        return "{} Prognostics Model (Events: {})".format(type(self).__name__, self.events)

    def __getstate__(self) -> dict:
        return self.parameters.data

    def __setstate__(self, state : dict) -> None:
        self.parameters = PrognosticsModelParameters(self, state, self.param_callbacks)

        self.n_inputs = len(self.inputs)
        self.n_states = len(self.states)
        self.n_events = len(self.events)
        self.n_outputs = len(self.outputs)
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
    
    @abstractmethod
    def initialize(self, u : dict = None, z :dict = None) -> dict:
        """
        Calculate initial state given inputs and outputs

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
        """
        return {}

    def apply_measurement_noise(self, z : dict) -> dict:
        """
        Apply measurement noise to the measurement

        Parameters
        ----------
        z : dict
            output, with keys defined by model.outputs \n
            e.g., z = {'abc': 332.1, 'def': 221.003} given outputs = ['abc', 'def']
 
        Returns
        -------
        z : dict
            output, with applied noise, with keys defined by model.outputs \n
            e.g., z = {'abc': 332.2, 'def': 221.043} given outputs = ['abc', 'def']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | z = {'z1': 2.2}
        | z = m.apply_measurement_noise(z)

        Note
        ----
        Configured using parameters `measurement_noise` and `measurement_noise_dist`
        """
        return self.OutputContainer({key: z[key] \
            + np.random.normal(
                0, self.parameters['measurement_noise'][key],
                size=None if np.isscalar(z[key]) else len(z[key]))
                for key in z.keys()})
        
    def apply_process_noise(self, x : dict, dt : int =1) -> dict:
        """
        Apply process noise to the state

        Parameters
        ----------
        x : dict
            state, with keys defined by model.states \n
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        dt : Number, optional
            Time step (e.g., dt = 0.1)

        Returns
        -------
        x : dict
            state, with applied noise, with keys defined by model.states
            e.g., x = {'abc': 332.2, 'def': 221.043} given states = ['abc', 'def']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = {'u1': 3.2}
        | z = {'z1': 2.2}
        | x = m.initialize(u, z) # Initialize first state
        | x = m.apply_process_noise(x) 

        Note
        ----
        Configured using parameters `process_noise` and `process_noise_dist`
        """
        return self.StateContainer({key: x[key] +
                dt*np.random.normal(
                    0, self.parameters['process_noise'][key],
                    size=None if np.isscalar(x[key]) else len(x[key]))
                    for key in x.keys()})

    def dx(self, x : dict, u : dict) -> dict:
        """
        Calculate the first derivative of state `x` at a specific time `t`, given state and input

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
        next_state

        Note
        ----
        A model should overwrite either `next_state` or `dx`. Override `dx` for continuous models,
        and `next_state` for discrete, where the behavior cannot be described by the first derivative
        """
        raise ProgModelException('dx not defined - please use next_state()')
        
    def next_state(self, x : dict, u : dict, dt : int) -> dict: 
        """
        State transition equation: Calculate next state

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
        dx

        Note
        ----
        A model should overwrite either `next_state` or `dx`. Override `dx` for continuous models, and `next_state` for discrete, where the behavior cannot be described by the first derivative
        """
        
        # Note: Default is to use the dx method (continuous model) - overwrite next_state for continuous
        dx = self.dx(x, u)
        return self.StateContainer({key: x[key] + dx[key]*dt for key in dx.keys()})

    def apply_limits(self, x : dict) -> dict:
        """
        Apply state bound limits. Any state outside of limits will be set to the closest limit.

        Parameters
        ----------
        x : dict
            state, with keys defined by model.states \n
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']

        Returns
        -------
        x : dict
            Bounded state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        """
        for (key, limit) in self.state_limits.items():
            if x[key] < limit[0]:
                warn("State {} limited to {} (was {})".format(key, limit[0], x[key]), ProgModelStateLimitWarning)
                x[key] = limit[0]
            elif x[key] > limit[1]:
                warn("State {} limited to {} (was {})".format(key, limit[1], x[key]), ProgModelStateLimitWarning)
                x[key] = limit[1]
        return x

    
    def __next_state(self, x : dict, u : dict, dt : int) -> dict:
        """
        State transition equation: Calls next_state(), calculating the next state, and then adds noise

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

    def performance_metrics(self, x : dict) -> dict:
        """
        Calculate performance metrics where

        Parameters
        ----------
        x : dict
            state, with keys defined by model.states \n
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        pm : dict
            Performance Metrics, with keys defined by model.performance_metric_keys. \n
            e.g., pm = {'tMax':33, 'iMax':19} given performance_metric_keys = ['tMax', 'iMax']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = {'u1': 3.2}
        | z = {'z1': 2.2}
        | x = m.initialize(u, z) # Initialize first state
        | pm = m.performance_metrics(x) # Returns {'tMax':33, 'iMax':19}
        """
        return {}
    
    observables = performance_metrics

    @abstractmethod
    def output(self, x : dict) -> dict:
        """
        Calculate outputs given state

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
        """
        return {}

    def __output(self, x : dict) -> dict:
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
        | z = {'o1': 1.2}
        | z = m.__output(3.0, x) # Returns {'o1': 1.2} with noise added
        """

        # Calculate next state, forward one timestep
        z = self.output(x)

        # Add measurement noise
        return self.apply_measurement_noise(z)

    def event_state(self, x : dict) -> dict:
        """
        Calculate event states (i.e., measures of progress towards event (0-1, where 0 means event has occured))

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
        threshold_met
        """
        return {}
    
    def threshold_met(self, x : dict) -> dict:
        """
        For each event threshold, calculate if it has been met

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
        event_state
        """
        return {key: event_state <= 0 \
            for (key, event_state) in self.event_state(x).items()} 

    def simulate_to(self, time : float, future_loading_eqn : Callable, first_output : dict = None, **kwargs) -> namedtuple:
        """
        Simulate prognostics model for a given number of seconds

        Parameters
        ----------
        time : number
            Time to which the model will be simulated in seconds (≥ 0.0) \n
            e.g., time = 200
        future_loading_eqn : callable
            Function of (t) -> z used to predict future loading (output) at a given time (t)
        first_output : dict, optional
            First measured output, needed to initialize state for some classes. Can be omitted for classes that dont use this
        options: kwargs, optional
            Configuration options for the simulation \n
            Note: configuration of the model is set through model.parameters \n
            Supported parameters: see `simulate_to_threshold`
        
        Returns
        -------
        times: number
            Times for each simulated point
        inputs: [dict]
            Future input (from future_loading_eqn) for each time in times
        states: [dict]
            Estimated states for each time in times
        outputs: [dict]
            Estimated outputs for each time in times
        event_states: [dict]
            Estimated event state (e.g., SOH), between 1-0 where 0 is event occurance, for each time in times
        
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
        | first_output = {'o1': 3.2, 'o2': 1.2}
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
 
    def simulate_to_threshold(self, future_loading_eqn : Callable, first_output : dict = None, threshold_keys : list = None, **kwargs) -> namedtuple:
        """
        Simulate prognostics model until any or specified threshold(s) have been met

        Parameters
        ----------
        future_loading_eqn : callable
            Function of (t) -> z used to predict future loading (output) at a given time (t)

        Keyword Arguments
        -----------------
        t0 : Number, optional
            Starting time for simulation in seconds (default: 0.0) \n
        dt : Number or function, optional
            time step (s), e.g. dt = 0.1 or function (t, x) -> dt\n
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
        |         return {'load': 3.0}
        |     else:
        |         return {'load': 5.0}
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
            x = deepcopy(config['x'])
        else:
            x = self.initialize(u, first_output)
        
        # Optimization
        next_state = self.__next_state
        output = self.__output
        thresthold_met_eqn = self.threshold_met
        event_state = self.event_state
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
        else:
            dt = config['dt']  # saving to optimize access in while loop
            def next_time(t, x):
                return dt
        
        # Simulate
        update_all()
        if progress:
            simulate_progress = ProgressBar(100, "Progress")
            last_percentage = 0
       
        while t < horizon:
            dt = next_time(t, x)
            t = t + dt/2
            # Use state at midpoint of step to best represent the load during the duration of the step
            u = future_loading_eqn(t, x)
            t = t + dt/2
            x = next_state(x, u, dt)

            # Save if at appropriate time
            if (t >= next_save):
                next_save = next(iterator)
                update_all()
            if (t >= save_pts[save_pt_index]):
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
            if check_thresholds(thresthold_met_eqn(x)):
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
    
    @staticmethod
    def generate_model(keys : dict, initialize_eqn : Callable, output_eqn : Callable, next_state_eqn : Callable = None, dx_eqn : Callable = None, event_state_eqn : Callable = None, threshold_eqn : Callable = None, config : dict = {'process_noise': 0.1}) -> "PrognosticsModel":
        """
        Generate a new prognostics model from individual model functions

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
            times ([double]): array of times for each sample
            inputs ([dict]): array of input dictionaries where input[x] corresponds to time[x]
            outputs ([dict]): array of output dictionaries where output[x] corresponds to time[x]
            kwargs: Configuration parameters, such as:\n
             | x0 [dict]: Initial state
             | dt [double] : time step

        Returns:
            double: Total error
        """
        params = {
            'x0': self.initialize(inputs[0], outputs[0]),
            'dt': 1e99
        }
        params.update(kwargs)
        x = params['x0']
        t_last = times[0]
        err_total = 0

        for t, u, z in zip(times, inputs, outputs):
            while t_last < t:
                t_new = min(t_last + params['dt'], t)
                x = self.next_state(x, u, t_new-t_last)
                t_last = t_new
            z_obs = self.output(x)
            if any([np.isnan(z_i) for z_i in z_obs.values()]):
                warn("Model unstable- NaN reached in simulation (t={})".format(t))
                break
            err_total += sum([(z[key] - z_obs[key])**2 for key in z.keys()])

        return err_total/len(times)
    
    def estimate_params(self, runs : List[tuple], keys : List[str], **kwargs) -> None:
        """Estimate the model parameters given data. Overrides model parameters

        Args:
            runs (array[tuple]): data from all runs, where runs[0] is the data from run 0. Each run consists of a tuple of arrays of times, input dicts, and output dicts
            keys ([string]): Parameter keys to optimize
            kwargs: Configuration parameters. Supported parameters include: \n
             | method: Optimization method- see scikit.optimize.minimize 
             | options: Options passed to optimizer

        See: examples.param_est
        """
        from scipy.optimize import minimize

        config = {
            'method': 'nelder-mead',  # Optimization method
            'options': {'xatol': 1e-8}  # Options passed to optimizer
        }
        config.update(kwargs)

        # Set noise to 0
        m_noise, self.parameters['measurement_noise'] = self.parameters['measurement_noise'], 0
        p_noise, self.parameters['process_noise'] = self.parameters['process_noise'], 0

        def optimization_fcn(params):
            for key, param in zip(keys, params):
                self.parameters[key] = param
            err = 0
            for run in runs:
                try:
                    err += self.calc_error(run[0], run[1], run[2], **kwargs)
                except Exception:
                    return 1e99 
                    # If it doesn't work (i.e., throws an error), dont use it
            return err
        
        params = np.array([self.parameters[key] for key in keys])

        res = minimize(optimization_fcn, params, method=config['method'], options=config['options'])
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
            String indicating surrogate modeling method to be used 

        Keyword Arguments
        -----------------
        Includes all keyword arguments from simulate_to_threshold (except save_pts), and the following additional keywords: 

        dt : Number or function, optional
            Same as in simulate_to_threshold; for DMD, this value is the time step of the training data\n
        save_freq : Number, optional
            Same as in simulate_to_threshold; for DMD, this value is the time step with which the surrogate model is generated  \n
        trim_data_to: int, optional
            Value between 0 and 1 that determines fraction of data resulting from simulate_to_threshold that is used to train DMD surrogate model
            e.g. if trim_data_to = 0.7 and the simulated data spans from t=0 to t=100, the surrogate model is trained on the data from t=0 to t=70 \n   
            Note: To trim data to a set time, use the 'horizon' parameter\n   
        states: list, optional
            List of state keys to be included in the surrogate model generation. keys must be a subset of those defined in the PrognosticsModel  \n
        inputs: list, optional
            List of input keys to be included in the surrogate model generation. keys must be a subset of those defined in the PrognosticsModel  \n
        outputs: list, optional
            List of output keys to be included in the surrogate model generation. keys must be a subset of those defined in the PrognosticsModel  \n
        events: list, optional
            List of event_state keys to be included in the surrogate model generation. keys must be a subset of those defined in the PrognosticsModel  \n      
        stability_tol: int, optional
            Value that determines the tolerance for DMD matrix stability\n

        Returns
        -------
        SurrogateModel(): class
            Instance of SurrogateModel class

        Example
        -------
        See examples/generate_surrogate

        Note
        -------
        This is a first draft of a surrogate model generation using Dynamic Mode Decomposition. 
        DMD does not generate accurate approximations for all models, especially highly non-linear sections, and can be sensitive to the training data time step. 
        In general, the approximation is less accurate if the DMD matrix is unstable. 
        Additionally, this implementation does not yet include all functionalities of DMD (e.g. reducing the system's dimensions through SVD). Further functionalities will be included in future releases. \n
        """

        if method != 'dmd':
            raise ProgModelInputException(f"Method {method} is not supported. DMD is currently the only method available.")

        # Configure
        config = { # Defaults
            'save_freq': 1.0, 
            'trim_data_to': 1,
            'states': self.states,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'events': self.events,
            'stability_tol': 1e-05
        }
        config.update(kwargs)

        # List of user-define values to include in surrogate model: 
        states_dmd = self.states.copy()
        inputs_dmd = self.inputs.copy()
        outputs_dmd = self.outputs.copy()
        events_dmd = self.events.copy()

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
        if not isinstance(config['trim_data_to'], Number) or config['trim_data_to']>1 or config['trim_data_to']<=0:
            raise ProgModelInputException("Invalid 'trim_data_to' input value, must be between 0 and 1.")
        if not isinstance(config['stability_tol'], Number) or  config['stability_tol'] < 0:
            raise ProgModelInputException(f"Invalid 'stability_tol' input value {config['stability_tol']}, must be a positive number.")

        if isinstance(config['inputs'], str):
            config['inputs'] = [config['inputs']]
        if not all([x in self.inputs for x in config['inputs']]):
            raise ProgModelInputException(f"Invalid 'inputs' input value ({config['inputs']}), must be a subset of the model's inputs ({self.inputs}).")
        
        if isinstance(config['states'], str):
            config['states'] = [config['states']]
        if not all([x in self.states for x in config['states']]):
            raise ProgModelInputException(f"Invalid 'states' input value ({config['states']}), must be a subset of the model's states ({self.states}).")

        if isinstance(config['outputs'], str):
            config['outputs'] = [config['outputs']]
        if not all([x in self.outputs for x in config['outputs']]):
            raise ProgModelInputException(f"Invalid 'outputs' input value ({config['outputs']}), must be a subset of the model's states ({self.outputs}).")

        if isinstance(config['events'], str):
            config['events'] = [config['events']]
        if not all([x in self.events for x in config['events']]):
            raise ProgModelInputException(f"Invalid 'events' input value ({config['events']}), must be a subset of the model's states ({self.events}).")

        # Initialize lists to hold individual matrices
        x_list = []
        xprime_list = []
        time_list = []

        # Generate Data to train surrogate model: 
        for iter_load, load_fcn_now in enumerate(load_functions):
            print('Generating training data: loading profile {} of {}'.format(iter_load+1, len(load_functions)))

            # Simulate to threshold 
            (times, inputs, states, outputs, event_states) = self.simulate_to_threshold(load_fcn_now, **config)
        
            # Interpolate results to time step of save_freq
            time_data_interp = np.arange(times[0], times[-1], config['save_freq'])

            states_data_interp = {}
            inputs_data_interp = {}

            for state_name in self.states:
                states_data_temp = [states[iter_data1][state_name] for iter_data1 in range(len(states))]
                states_data_interp[state_name] = interp1d(times,states_data_temp)(time_data_interp)
            for input_name in self.inputs:
                inputs_data_temp = [inputs[iter_data4][input_name] for iter_data4 in range(len(inputs))]
                inputs_data_interp[input_name] = interp1d(times,inputs_data_temp)(time_data_interp)

            states_data = [
                self.StateContainer({
                    key: value[iter_dataT] for key, value in states_data_interp.items()
                }) for iter_dataT in range(len(time_data_interp))
                ]
            inputs_data = [
                self.InputContainer({
                    key: value[iter_dataT] for key, value in inputs_data_interp.items()
                }) for iter_dataT in range(len(time_data_interp))
                ]

            times = time_data_interp.tolist()
            states = SimResult(time_data_interp,states_data)
            inputs = SimResult(time_data_interp,inputs_data)
            outputs = LazySimResult(self.output, time_data_interp, states_data) 
            event_states = LazySimResult(self.event_state, time_data_interp, states_data)
            
            def user_val_set(iter_loop : list, config_key : str, remove_from : dict, del_from) -> None:
                """Sub-function for performing check and removal for user designated values.
            
                Args:
                    iter_loop : list
                        List of keys to iterate through.
                    config_key : str
                        String key to check keys against config
                    remove_from : dict
                        Dictionary dmd to remove key from
                    del_from : list or dict
                        Final data structure to remove key and data from 
                """
                for key in iter_loop:
                    if key not in config[config_key]:
                        if iter_load == 0:
                            remove_from.remove(key)
                        for i in range(len(times)):
                            del del_from[i][key]
                           
            if len(config['states']) != len(self.states):
                user_val_set(self.states,  'states', states_dmd, states)
            if len(config['inputs']) != len(self.inputs):
                user_val_set(self.inputs,  'inputs', inputs_dmd, inputs)
            if len(config['outputs']) != len(self.outputs):
                user_val_set(self.outputs,  'outputs', outputs_dmd, outputs) 
            if len(config['events']) != len(self.events):
                user_val_set(self.events,  'events', events_dmd, event_states)  

            # Initialize DMD matrices
            x_mat_temp = np.zeros((len(states[0])+len(outputs[0])+len(event_states[0])+len(inputs[0]),len(times)-1)) 
            xprime_mat_temp = np.zeros((len(states[0])+len(outputs[0])+len(event_states[0]),len(times)-1)) 

            # Save DMD matrices
            for i, time in enumerate(times[:-1]): 
                time_now = time + np.divide(config['save_freq'],2) 
                load_now = load_fcn_now(time_now) # Evaluate load_function at (t_now + t_next)/2 to be consistent with next_state implementation
                if len(config['inputs']) != len(self.inputs): # Delete any input values not specified by user to be included in surrogate model 
                    for key in self.inputs:
                        if key not in config['inputs']:
                            del load_now[key]

                states_now = states[i].matrix 
                states_next = states[i+1].matrix 
  
                stack = (
                        states_now,
                        outputs[i].matrix,
                        np.array([list(event_states[i].values())]).T,
                        np.array([[load_now[key]] for key in load_now.keys()])
                    )
                x_mat_temp[:,i] = np.vstack(tuple(v for v in stack if v.shape != (0, )))[:,0]  # Filter out empty values (e.g., if there is no input)
                stack2 = (
                    states_next,
                    outputs[i+1].matrix,
                    np.array([list(event_states[i+1].values())]).T
                )
                xprime_mat_temp[:,i] = np.vstack(tuple(v for v in stack2 if v.shape != (1,0)))[:,0]  # Filter out empty values (e.g., if there is no output)
                
            # Save matrices in list, where each index in list corresponds to one of the user-defined loading equations 
            x_list.append(x_mat_temp)
            xprime_list.append(xprime_mat_temp)
            time_list.append(times)

        # Format training data for DMD and solve for matrix A, in the form X' = AX 
        print('Generate DMD Surrogate Model')

        # Cut data to user-defined length 
        if config['trim_data_to'] != 1:
            for iter3 in range(len(load_functions)):
                trim_index = round(len(time_list[iter3])*(config['trim_data_to'])) 
                x_list[iter3] = x_list[iter3][:,0:trim_index]
                xprime_list[iter3] = xprime_list[iter3][:,0:trim_index]
     
        # Convert lists of datasets into arrays, sequentially stacking data in the horizontal direction
        x_mat = np.hstack((x_list[:]))
        xprime_mat = np.hstack((xprime_list[:]))

        # Calculate DMD matrix using the Moore-Penrose pseudo-inverse:
        dmd_matrix = np.dot(xprime_mat,np.linalg.pinv(x_mat))

        # Save size of states, inputs, outputs, event_states, and current instance of PrognosticsModel
        num_states = len(states[0].matrix)
        num_inputs = len(inputs[0].matrix)
        num_outputs = len(outputs[0].matrix)
        num_event_states = len(event_states[0])
        num_total = num_states + num_outputs + num_event_states 
        prog_model = self
        dmd_dt = config['save_freq']
        process_noise_temp = {key: 0 for key in prog_model.events}  # Process noise for event states is zero

        # Check for stability of dmd_matrix
        eig_val, _ = np.linalg.eig(dmd_matrix[:,0:-num_inputs if num_inputs > 0 else None])            
        
        if sum(eig_val>1) != 0:
            for eig_val_i in eig_val:
                if eig_val_i>1 and eig_val_i-1>config['stability_tol']:
                    warn("The DMD matrix is unstable, may result in poor approximation.")

        from .linear_model import LinearModel
        

        class SurrogateModelDMD(LinearModel):
            """
            A subclass of LinearModel that uses Dynamic Mode Decomposition to simulate a system throughout time.
            
            Given an initial state of the system (including internal states, outputs, and event_states), and the expected inputs throuhgout time, this class defines a surrogate model that can approximate the internal states, outputs, and event_states throughout time until threshold is met.

            Keyword Args
            ------------
                process_noise : Optional, float or Dict[Srt, float]

            See Also
            ------
                LinearModel

            Attributes
            ----------
                initialize : 
                    Calculate initial state, augmented with outputs and event_states

                next_state : 
                    State transition equation: Calculate next state with matrix multiplication (overrides 'dx' defined in LinearModel)

                simulate_to_threshold:
                    Simulate prognostics model until defined threshold is met, using simulate_to_threshold defined in PrognosticsModel, then interpolate results to be at user-defined times
            """

            # Default parameters: set process_noise and measurement_noise to be defined based on PrognosticsModel values
            default_parameters = {
                'process_noise': {**prog_model.parameters['process_noise'],**prog_model.parameters['measurement_noise'],**process_noise_temp},
                'measurement_noise': prog_model.parameters['measurement_noise'],
                'process_noise_dist': prog_model.parameters.get('process_noise_dist', 'normal'),
                'measurement_noise_dist': prog_model.parameters.get('measurement_noise_dist', 'normal')
            }

            # Define appropriate matrices for LinearModel
            A = dmd_matrix[:,0:num_total]
            B = np.vstack(dmd_matrix[:,num_total:num_total+num_inputs]) 
            C = np.zeros((num_outputs,num_total))
            for iter1 in range(num_outputs):
                C[iter1,num_states+iter1] = 1 
            F = np.zeros((num_event_states,num_total))
            for iter2 in range(num_event_states):
                F[iter2,num_states+num_outputs+iter2] = 1 

            states = states_dmd + outputs_dmd + events_dmd
            inputs = inputs_dmd
            outputs = outputs_dmd 
            events = events_dmd

            dt = dmd_dt  # Step size (so it can be accessed programmatically)

            def initialize(self, u=None, z=None):
                x = prog_model.initialize(u,z)
                x.update(prog_model.output(x))
                x.update(prog_model.event_state(x))

                return self.StateContainer(x)

            def next_state(self, x, u, _):   
                x.matrix = np.matmul(self.A, x.matrix) + np.matmul(self.B, u.matrix) + self.E
                
                return x   

            def simulate_to_threshold(self, future_loading_eqn, first_output = None, threshold_keys = None, **kwargs):
                # Save keyword arguments same as DMD training for approximation 
                kwargs_sim = kwargs.copy()
                kwargs_sim['save_freq'] = dmd_dt
                kwargs_sim['dt'] = dmd_dt

                # Simulate to threshold at DMD time step
                results = super().simulate_to_threshold(future_loading_eqn,first_output, threshold_keys, **kwargs_sim)
                
                # Interpolate results to be at user-desired time step
                if 'dt' in kwargs:
                    warn("dt is not used in DMD approximation")

                # Default parameters 
                config = {
                    'dt': None,
                    'save_freq': None,
                    'save_pts': []
                }
                config.update(kwargs)

                if (config['save_freq'] == dmd_dt or
                    (isinstance(config['save_freq'], tuple) and
                        config['save_freq'][0]%dmd_dt < 1e-9 and
                        config['save_freq'][1] == dmd_dt)
                    ) and config['save_pts'] == []:
                    # In this case, the user wants what the DMD approximation returns 
                    return results 

                # In this case, the user wants something different than what the DMD approximation retuns, so we must interpolate 
                # Define time vector based on user specifications
                time_basic = [results.times[0], results.times[-1]]
                time_basic.extend(config['save_pts'])                       
                if config['save_freq'] != None:
                    if isinstance(config['save_freq'], tuple):
                        # Tuple used to specify start and frequency
                        t_step = config['save_freq'][1]
                        # Use starting time or the next multiple
                        t_start = config['save_freq'][0]
                        start = max(t_start, results.times[0] - (results.times[0]-t_start)%t_step)
                        time_array = np.arange(start+t_step,results.times[-1],t_step)
                    else: 
                        time_array = np.arange(results.times[0]+config['save_freq'],results.times[-1],config['save_freq'])
                    time_basic.extend(time_array.tolist())
                time_interp = sorted(time_basic)

                # Interpolate States
                states_dict_temp = {}
                for states_name in self.states:
                    states_list_temp = [results.states[iter1a][states_name] for iter1a in range(len(results.states))]
                    states_dict_temp[states_name] = interp1d(results.times,states_list_temp)(time_interp)
                states_interp = [
                    self.StateContainer({key: state[i] for key, state in states_dict_temp.items()})
                    for i in range(len(time_interp))
                ]
                    
                # Interpolate Inputs
                inputs_dict_temp = {}
                for inputs_name in self.inputs:
                    inputs_list_temp = [results.inputs[iter1a][inputs_name] for iter1a in range(len(results.inputs))]
                    inputs_dict_temp[inputs_name] = interp1d(results.times,inputs_list_temp)(time_interp)
                inputs_interp = [
                    self.InputContainer({key: input[i] for key, input in inputs_dict_temp.items()}) for i in range(len(time_interp))
                ]

                states = SimResult(time_interp,states_interp)
                inputs = SimResult(time_interp,inputs_interp)
                outputs = LazySimResult(self.output, time_interp, states_interp)
                event_states = LazySimResult(self.event_state, time_interp, states_interp)

                return self.SimulationResults(
                    time_interp,
                    inputs,
                    states,
                    outputs,
                    event_states
                )
                
        return SurrogateModelDMD()
