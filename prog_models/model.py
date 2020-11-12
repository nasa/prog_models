# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

# Todo(CT): Should we name this SytemModel?
from abc import ABC, abstractmethod
from . import ProgModelInputException, ProgModelException, ProgModelTypeError
from numbers import Number
import numpy as np
import copy

class Model(ABC):
    """
    A general time-variant state space model of system behavior.

    The Model class is a wrapper around a mathematical model of a system as
    represented by a state and output equation. Optionally, it may also
    include an input equation, which defines what the system input should be
    at any given time, and an initialize equation, which computes the initial
    system state given the inputs and outputs.

    A Model also has a parameters structure, which contains fields for
    various model parameters. The parameters structure is always given as a
    first argument to all provided equation handles. However, when calling
    the methods for these equations, it need not be specified as it is passed
    by default since it is a property of the class.
    
    The process and sensor noise variances are represented by vectors. When
    using the generate noise methods, samples are generated from zero-mean
    uncorrelated Gaussian noise as specified by these variances.
    """

    parameters = {} # Configuration Parameters for model
    # inputs = []     # Identifiers for each input
    # states = []     # Identifiers for each state
    # outputs = []    # Identifiers for each output

    # TODO(CT): Check if properties above are defined (in constructor?)

    def __init__(self):
        if 'process_noise' not in self.parameters:
            raise ProgModelTypeError('Missing `process_noise` parameter')

        if not hasattr(self, 'inputs'):
            raise ProgModelTypeError('Must have `inputs` attribute')
        if len(self.inputs) <= 0:
            raise ProgModelTypeError('`inputs` attribute must have at least one input key')

        if not hasattr(self, 'states'):
            raise ProgModelTypeError('Must have `states` attribute')
        if len(self.states) <= 0:
            raise ProgModelTypeError('`states` attribute must have at least one state key')

        if not hasattr(self, 'outputs'):
            raise ProgModelTypeError('Must have `outputs` attribute')
        if len(self.outputs) <= 0:
            raise ProgModelTypeError('`outputs` attribute must have at least one output key')

        if isinstance(self.parameters['process_noise'], Number):
            self.parameters['process_noise'] = {key: self.parameters['process_noise'] for key in self.states}

    @abstractmethod
    def initialize(self, u, z) -> dict:
        """
        Calculate initial state given inputs and outputs

        Parameters
        ----------
        u : dict
            Inputs, with keys defined by model.inputs.
            e.g., u = {'i':3.2} given inputs = ['i']
        z : dict
            Outputs, with keys defined by model.outputs.
            e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']

        Returns
        -------
        x : dict
            First state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        """
        pass
        
    def apply_process_noise(self, x):
        return {key: x[key] + np.random.normal(0, self.parameters['process_noise'][key]) for key in self.states}

    @abstractmethod
    def next_state(self, t, x, u, dt) -> dict: 
        """
        State transition equation: Calculate next state

        Parameters
        ----------
        t : number
            Current timestamp in seconds (≥ 0)
            e.g., t = 3.4
        x : dict
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        u : dict
            Inputs, with keys defined by model.inputs.
            e.g., u = {'i':3.2} given inputs = ['i']
        dt : number
            Timestep size in seconds (≥ 0)
            e.g., dt = 0.1
        

        Returns
        -------
        x : dict
            Next state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        """

        pass

    @abstractmethod
    def output(self, t, x) -> dict:
        """
        Calculate next statem, forward one timestep

        Parameters
        ----------
        t : number
            Current timestamp in seconds (≥ 0.0)
            e.g., t = 3.4
        x : dict
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        z : dict
            Outputs, with keys defined by model.outputs.
            e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']
        """

        pass

    def simulate_to(self, time, future_loading_eqn, first_output, options = {}):
        """
        Simulate model for a given time interval

        Parameters
        ----------
        time : number
            Time to which the model will be simulated in seconds (≥ 0.0)
            e.g., time = 200
        future_loading_eqn : function
            Function of (t) -> z used to predict future loading (output) at a given time (t)
        options: dict, optional:
            Configuration options for the simulation
            Note: configuration of the model is set through model.parameters
        
        Returns (tuple)
        -------
        times: number
            Times for each simulated point
        inputs: [dict]
            Future input (from future_loading_eqn) for each time in times
        states: [dict]
            Estimated states for each time in times
        outputs: [dict]
            Estimated outputs for each time in times
        
        Example
        -------
        (times, inputs, states, outputs) = m.simulate_to(200, future_load_eqn, first_output)
        """
        # Input Validation
        if not isinstance(time, Number) or time <= 0:
            raise ProgModelInputException("'time' must be number greater than 0, was {} ({})".format(time, type(time)))

        if not all(key in first_output for key in self.outputs):
            raise ProgModelInputException("Missing key in 'first_output', must have every key in model.outputs")

        if not (callable(future_loading_eqn)):
            raise ProgModelInputException("'future_loading_eqn' must be callable f(t)")

        # Configure
        config = { # Defaults
            'dt': 1,
            'save_freq': 10
        }
        config.update(options)

        # Configuration validation
        if type(config['dt']) is not int and type(config['dt']) is not float:
            raise ProgModelInputException("'dt' must be a number, was a {}".format(type(config['dt'])))
        if config['dt'] <= 0:
            raise ProgModelInputException("'dt' must be positive, was {}".format(config['dt']))
        if type(config['save_freq']) is not int and type(config['save_freq']) is not float:
            raise ProgModelInputException("'save_freq' must be a number, was a {}".format(type(config['save_freq'])))
        if config['save_freq'] <= 0:
            raise ProgModelInputException("'save_freq' must be positive, was {}".format(config['save_freq']))

        # Setup
        t = 0
        u = future_loading_eqn(t)
        if 'x' in config:
            x = config['x']
        else:
            x = self.initialize(u, first_output)
        times = np.array([t])
        inputs = np.array([u])
        states = np.array([copy.deepcopy(x)]) # Avoid optimization where x is not copied
        outputs = np.array([self.output(t, x)])
        dt = config['dt'] # saving to optimize access in while loop
        save_freq = config['save_freq']
        next_save = save_freq

        # Simulate Forward
        while t < time:
            t += dt
            u = future_loading_eqn(t)
            x = self.next_state(t, x, u, dt)
            if (t >= next_save):
                next_save += save_freq
                times = np.append(times,t)
                inputs = np.append(inputs,u)
                states = np.append(states,copy.deepcopy(x))
                outputs = np.append(outputs,self.output(t, x))
        
        # Record final state
        if times[-1] != t:
            # This check prevents double recording when the last state was a savepoint 
            times = np.append(times,t)
            inputs = np.append(inputs,u)
            states = np.append(states,x)
            outputs = np.append(outputs,self.output(t, x))

        return (times, inputs, states, outputs)