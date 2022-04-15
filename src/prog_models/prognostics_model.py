# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from .exceptions import ProgModelInputException, ProgModelTypeError, ProgModelException, ProgModelStateLimitWarning
from abc import abstractmethod, ABC
from numbers import Number
import numpy as np
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

    def __eq__(self, other):
        """
        Check if two models are equal
        """
        return self.__class__ == other.__class__ and self.parameters == other.parameters
    
    def __str__(self):
        return "{} Prognostics Model (Events: {})".format(type(self).__name__, self.events)

    def __getstate__(self):
        return self.parameters.data

    def __setstate__(self, state):
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
    def initialize(self, u = None, z = None) -> dict:
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

    def apply_measurement_noise(self, z) -> dict:
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
        
    def apply_process_noise(self, x, dt=1) -> dict:
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

    def dx(self, x, u):
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
        
    def next_state(self, x, u, dt) -> dict: 
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

    def apply_limits(self, x):
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

    
    def __next_state(self, x, u, dt) -> dict:
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

    def performance_metrics(self, x) -> dict:
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
    def output(self, x) -> dict:
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

    def __output(self, x) -> dict:
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

    def event_state(self, x) -> dict:
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
    
    def threshold_met(self, x) -> dict:
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

    def simulate_to(self, time, future_loading_eqn, first_output = None, **kwargs) -> namedtuple:
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
 
    def simulate_to_threshold(self, future_loading_eqn, first_output = None, threshold_keys = None, **kwargs) -> namedtuple:
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
            x = config['x']
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
    def generate_model(keys, initialize_eqn, output_eqn, next_state_eqn = None, dx_eqn = None, event_state_eqn = None, threshold_eqn = None, config = {'process_noise': 0.1}):
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

    def calc_error(self, times, inputs, outputs, **kwargs):
        """Calculate error between simulated and observed

        Args:
            times ([double]): array of times for each sample
            inputs ([dict]): array of input dictionaries where input[x] corresponds to time[x]
            outputs ([dict]): array of output dictionaries where output[x] corresponds to time[x]
            kwargs: Configuration parameters, such as:\n
             | dt [double] : time step

        Returns:
            double: Total error
        """
        params = {
            'dt': 1e99
        }
        params.update(kwargs)
        x = self.initialize(inputs[0], outputs[0])
        t_last = times[0]
        err_total = 0

        for t, u, z in zip(times, inputs, outputs):
            while t_last < t:
                t_new = min(t_last + params['dt'], t)
                x = self.next_state(x, u, t_new-t_last)
                t_last = t_new
            z_obs = self.output(x)
            err_total += sum([(z[key] - z_obs[key])**2 for key in z.keys()])

        return err_total
    
    def estimate_params(self, runs, keys, **kwargs):
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
