# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from . import ProgModelInputException, ProgModelTypeError, ProgModelException
from abc import abstractmethod, ABC
from numbers import Number
from numpy import random
from copy import deepcopy
from collections import UserDict
import types

class PrognosticsModelParameters(UserDict):
    """
    Prognostics Model Parameters - this class replaces a standard dictionary. 
    It includes the extra logic to process the different supported manners of defining noise.
    """
    def __init__(self, model, dict_in = {}):
        super().__init__()
        self.__m = model
        for (key, value) in dict_in.items():
            self[key] = value

    def __setitem__(self, key, value):
        """Set model configuration, overrides dict.__setitem__()

        Args:
            key (string): configuration key to set
            value: value to set that configuration value to

        Raises:
            ProgModelTypeError: Improper configuration for a model
        """
        super().__setitem__(key, value)
        
        if key == 'process_noise':
            if callable(self['process_noise']): # Provided a function
                self.__m.apply_process_noise = types.MethodType(self['process_noise'], self.__m)
            else: # Not a function
                # Process noise is single number - convert to dict
                if isinstance(self['process_noise'], Number):
                    self['process_noise'] = {key: self['process_noise'] for key in self.__m.states}
                
                # Process distribution type
                if 'process_noise_dist' in self and self['process_noise_dist'].lower() not in ["gaussian", "normal"]:
                    # Uxpdate process noise distribution to custom 
                    if self['process_noise_dist'].lower() == "uniform":
                        def uniform_process_noise(self, x, dt=1):
                            return {key: x[key] + dt*random.uniform(-self.parameters['process_noise'][key], self.parameters['process_noise'][key]) for key in self.states}
                        self.__m.apply_process_noise = types.MethodType(uniform_process_noise, self.__m)
                    elif self['process_noise_dist'].lower() == "triangular":
                        def triangular_process_noise(self, x, dt=1):
                            return {key: x[key] + dt*random.triangular(-self.parameters['process_noise'][key], 0, self.parameters['process_noise'][key]) for key in self.states}
                        self.__m.apply_process_noise = types.MethodType(triangular_process_noise, self.__m)
                    else:
                        raise ProgModelTypeError("Unsupported process noise distribution")
                
                # Make sure every key is present (single value already handled above)
                if not all([key in self['process_noise'] for key in self.__m.states]):
                    raise ProgModelTypeError("Process noise must have ever key in model.states")
        elif key == 'measurement_noise':
            if callable(self['measurement_noise']):
                self.__m.apply_measurement_noise = types.MethodType(self['measurement_noise'], self.__m)
            else:
                # Process noise is single number - convert to dict
                if isinstance(self['measurement_noise'], Number):
                    self['measurement_noise'] = {key: self['measurement_noise'] for key in self.__m.outputs}
                
                # Process distribution type
                if 'measurement_noise_dist' in self and self['measurement_noise_dist'].lower() not in ["gaussian", "normal"]:
                    # Update measurment noise distribution to custom 
                    if self['measurement_noise_dist'].lower() == "uniform":
                        def uniform_noise(self, x):
                            return {key: x[key] + random.uniform(-self.parameters['measurement_noise'][key], self.parameters['measurement_noise'][key]) for key in self.outputs}
                        self.__m.apply_measurement_noise = types.MethodType(uniform_noise, self.__m)
                    elif self['measurement_noise_dist'].lower() == "triangular":
                        def triangular_noise(self, x):
                            return {key: x[key] + random.triangular(-self.parameters['measurement_noise'][key], 0, self.parameters['measurement_noise'][key]) for key in self.outputs}
                        self.__m.apply_measurement_noise = types.MethodType(triangular_noise, self.__m)
                    else:
                        raise ProgModelTypeError("Unsupported measurement noise distribution")
                
                # Make sure every key is present (single value already handled above)
                if not all([key in self['measurement_noise'] for key in self.__m.outputs]):
                    raise ProgModelTypeError("Measurement noise must have ever key in model.states")

class PrognosticsModel(ABC):
    """
    A general time-variant state space model of system degradation behavior.

    The PrognosticsModel class is a wrapper around a mathematical model of a
    system as represented by a state, output, input, event_state and threshold equations.

    A Model also has a parameters structure, which contains fields for
    various model parameters.

    Parameters
    ----------
    kwargs : keyword arguments, optional
        Configuration parameters for model. Parameters supported by every model include:\n
            * process_noise : Process noise (applied at dx/next_state). 
                Can be number (e.g., .2) applied to every state, a dictionary of values for each 
                state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x\n
            * process_noise_dist : Optional, distribution for process noise (e.g., normal, uniform, triangular)\n
            * measurement_noise : Measurement noise (applied in output eqn)
                Can be number (e.g., .2) applied to every output, a dictionary of values for each 
                output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z\n
            * measurement_noise_dist : Optional, distribution for measurement noise (e.g., normal, uniform, triangular)\n
        E.g., PrognosticsModel(process_noise= 0.3, measurement_noise= {'z1': 0.1, 'z2': 0.3})
    
    Raises
    ------
    ProgModelTypeError

    Example
    -------
    m = PrognosticsModel({'process_noise': 3.2})
    """

    default_parameters = {
        'process_noise': 0.1,
        'measurement_noise': 0.0
    } # Configuration Parameters for model

    # inputs = []     # Identifiers for each input
    # states = []     # Identifiers for each state
    # outputs = []    # Identifiers for each output
    observables = []  # Identifies for each observable
    events = []       # Identifiers for each event

    def __init__(self, **kwargs):
        try:
            if not hasattr(self, 'inputs'):
                raise ProgModelTypeError('Must have `inputs` attribute')
            
            if not hasattr(self, 'states'):
                raise ProgModelTypeError('Must have `states` attribute')
            if len(self.states) <= 0:
                raise ProgModelTypeError('`states` attribute must have at least one state key')
            try:
                a = iter(self.states)
            except TypeError:
                raise ProgModelTypeError('model.states must be iterable')

            if not hasattr(self, 'outputs'):
                raise ProgModelTypeError('Must have `outputs` attribute')
            try:
                a = iter(self.outputs)
            except TypeError:
                raise ProgModelTypeError('model.outputs must be iterable')
        except Exception:
            raise ProgModelTypeError('Could not check model configuration')

        self.parameters = PrognosticsModelParameters(self, self.__class__.default_parameters)
        try:
            self.parameters.update(kwargs)
        except TypeError:
            raise ProgModelTypeError("couldn't update parameters. `options` must be type dict (was {})".format(type(kwargs)))

        try:
            if 'process_noise' not in self.parameters:
                self.parameters['process_noise'] = 0.1
            else:
                self.parameters['process_noise'] = self.parameters['process_noise'] # To force  __setitem__

            if 'measurement_noise' not in self.parameters:
                self.parameters['measurement_noise'] = 0.0
            else:
                self.parameters['measurement_noise'] = self.parameters['measurement_noise']
        except Exception:
            raise ProgModelTypeError('Model noise poorly configured')

        # TODO(CT): SOMEHOW CHECK IF DX OR STATE_EQN HAS BEEN OVERRIDDEN - ONE MUST

    def __str__(self):
        return "{} Prognostics Model (Events: {})".format(type(self).__name__, self.events, self.inputs, self.outputs)
    
    @abstractmethod
    def initialize(self, u, z) -> dict:
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
        pass

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
        return {key: z[key] + random.normal(0, self.parameters['measurement_noise'][key]) for key in self.outputs}

        
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
        return {key: x[key] + dt*random.normal(0, self.parameters['process_noise'][key]) for key in self.states}

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
        A model should overwrite either `next_state` or `dx`. Override `dx` for continuous models, and `next_state` for discrete, where the behavior cannot be described by the first derivative
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
        return {key: x[key] + dx[key]*dt for key in x.keys()}

    def observables(self, x) -> dict:
        """
        Calculate observables where

        Parameters
        ----------
        x : dict
            state, with keys defined by model.states \n
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        obs : dict
            Observables, with keys defined by model.observables. \n
            e.g., obs = {'tMax':33, 'iMax':19} given observables = ['tMax', 'iMax']

        Example
        -------
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = {'u1': 3.2}
        | z = {'z1': 2.2}
        | x = m.initialize(u, z) # Initialize first state
        | obs = m.observables(3.0, x) # Returns {'tMax':33, 'iMax':19}
        """
        return {}

    @abstractmethod
    def output(self, x) -> dict:
        """
        Calculate next state, forward one timestep

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
        | z = m.output(3.0, x) # Returns {'o1': 1.2}
        """

        return {}

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
            If each threshold has been met (bool), with deys defined by prognostics_model.events\n
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

        return {key: event_state <= 0 for (key, event_state) in self.event_state(x).items()} 

    def simulate_to(self, time, future_loading_eqn, first_output, **kwargs) -> tuple:
        """
        Simulate prognostics model for a given number of seconds

        Parameters
        ----------
        time : number
            Time to which the model will be simulated in seconds (≥ 0.0) \n
            e.g., time = 200
        future_loading_eqn : callable
            Function of (t) -> z used to predict future loading (output) at a given time (t)
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
 
    def simulate_to_threshold(self, future_loading_eqn, first_output, threshold_keys = None, **kwargs) -> tuple:
        """
        Simulate prognostics model until any or specified threshold(s) have been met

        Parameters
        ----------
        future_loading_eqn : callable
            Function of (t) -> z used to predict future loading (output) at a given time (t)
        first_output : dict
            First measured output, needed to initialize state
        threshold_keys: [str], optional
            Keys for events that will trigger the end of simulation. 
            If blank, simulation will occur if any event will be met ()
        options: keyword arguments, optional
            Configuration options for the simulation \n
            Note: configuration of the model is set through model.parameters.\n
            Supported parameters:\n
             * dt (Number0: time step (s), e.g. {'dt': 0.1} \n
             * save_freq (Number): Frequency at which output is saved (s), e.g., save_freq = 10 \n
             * save_pts ([Number]): Additional ordered list of custom times where output is saved (s), e.g., save_pts= [50, 75] \n
             * horizon (Number): maximum time that the model will be simulated forward (s), e.g., horizon = 1000 \n
             * x (dict): optional, initial state dict, e.g., x= {'x1': 10, 'x2': -5.3}\n
             * thresholds_met_eqn (function/lambda): optional, custom equation to indicate logic for when to stop sim f(thresholds_met) -> bool\n
            e.g., m.simulate_to_threshold(eqn, z, dt=0.1, save_pts=[1, 2])
        
        Returns
        -------
        times: [number]
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
        simulate_to

        Example
        -------
        | def future_load_eqn(t):
        |     if t< 5.0: # Load is 3.0 for first 5 seconds
        |         return 3.0
        |     else:
        |         return 5.0
        | first_output = {'o1': 3.2, 'o2': 1.2}
        | m = PrognosticsModel() # Replace with specific model being simulated
        | (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load_eqn, first_output)
        """
        # Input Validation
        if not all(key in first_output for key in self.outputs):
            raise ProgModelInputException("Missing key in 'first_output', must have every key in model.outputs")

        if not (callable(future_loading_eqn)):
            raise ProgModelInputException("'future_loading_eqn' must be callable f(t)")

        if threshold_keys and not all([key in self.events for key in threshold_keys]):
            raise ProgModelInputException("threshold_keys must be event names")

        # Configure
        config = { # Defaults
            'dt': 1.0,
            'save_pts': [],
            'save_freq': 10.0,
            'horizon': 1e100 # Default horizon (in s), essentially inf
        }
        config.update(kwargs)
        
        # Configuration validation
        if not isinstance(config['dt'], Number):
            raise ProgModelInputException("'dt' must be a number, was a {}".format(type(config['dt'])))
        if config['dt'] <= 0:
            raise ProgModelInputException("'dt' must be positive, was {}".format(config['dt']))
        if not isinstance(config['save_freq'], Number):
            raise ProgModelInputException("'save_freq' must be a number, was a {}".format(type(config['save_freq'])))
        if config['save_freq'] <= 0:
            raise ProgModelInputException("'save_freq' must be positive, was {}".format(config['save_freq']))
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

        # Setup
        t = 0
        u = future_loading_eqn(t)
        if 'x' in config:
            x = config['x']
        else:
            x = self.initialize(u, first_output)
        
        times = [t]
        inputs = [u]
        states = [deepcopy(x)] # Avoid optimization where x is not copied
        outputs = [self.output(x)]
        event_states = [self.event_state(x)]
        dt = config['dt'] # saving to optimize access in while loop
        save_freq = config['save_freq']
        horizon = config['horizon']
        next_save = save_freq
        save_pt_index = 0
        save_pts = config['save_pts']
        save_pts.append(1e99) # Add last endpoint
        threshold_met = False

        # Optimization
        next_state = self.next_state
        output = self.output
        thresthold_met_eqn = self.threshold_met
        event_state = self.event_state
        if 'thresholds_met_eqn' in config:
            check_thresholds = config['thresholds_met_eqn']
        elif threshold_keys is None: 
            # Note: Dont use implicit boolean in this check- it would then activate for an empty array
            def check_thresholds(thresholds_met):
                return any(thresholds_met.values())
        else:
            def check_thresholds(thresholds_met):
                return any([thresholds_met[key] for key in threshold_keys])

        def update_all():
            times.append(t)
            inputs.append(u)
            states.append(deepcopy(x))
            outputs.append(output(x))
            event_states.append(event_state(x))
        
        # Simulate
        while not threshold_met and t < horizon:
            t += dt
            u = future_loading_eqn(t, x)
            x = next_state(x, u, dt)
            threshold_met = check_thresholds(thresthold_met_eqn(x))
            if (t >= next_save):
                next_save += save_freq
                update_all()
            if (t >= save_pts[save_pt_index]):
                save_pt_index += 1
                update_all()

        # Save final state
        if times[-1] != t:
            # This check prevents double recording when the last state was a savepoint
            update_all()
        
        return (times, inputs, states, outputs, event_states)
    
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
