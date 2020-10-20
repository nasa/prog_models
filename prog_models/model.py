# Todo(CT): Should we name this SytemModel?
from abc import ABC, abstractmethod

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
        inputs = []     # Identifiers for each input
        states = []     # Identifiers for each state
        outputs = []    # Identifiers for each output

        # TODO(CT): Check if properties above are defined (in constructor?)

        @abstractmethod
        def initialize(self, u : dict, z : dict) -> dict:
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

        @abstractmethod
        def state(self, t, x : dict, u : dict, dt) -> dict: 
            """
            Calculate next state, forward one timestep

            Parameters
            ----------
            t : double
                Current timestamp in seconds (≥ 0.0)
                e.g., t = 3.4
            x : dict
                state, with keys defined by model.states
                e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
            u : dict
                Inputs, with keys defined by model.inputs.
                e.g., u = {'i':3.2} given inputs = ['i']
            dt : double
                Timestep size in seconds (≥ 0.0)
                e.g., dt = 0.1
            

            Returns
            -------
            x : dict
                Next state, with keys defined by model.states
                e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
            """

            pass

        @abstractmethod
        def output(self, t, x : dict) -> dict:
            """
            Calculate next statem, forward one timestep

            Parameters
            ----------
            t : double
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

        def simulate_to(self, time, future_loading_eqn, first_output : dict, options : dict = {}):
            """
            Simulate model for a given time interval

            Parameters
            ----------
            time : double
                Time to which the model will be simulated in seconds (≥ 0.0)
                e.g., time = 200
            future_loading_eqn : function
                Function of (t) -> z used to predict future loading (output) at a given time (t)
            options: dict, optional:
                Configuration options for the simulation
                Note: configuration of the model is set through model.parameters
            
            Returns
            -------
            results : dict
                Results recorded during simulation
                e.g., results = [{
                    't': 0,                             # Time (s)
                    'u': {'i': 12},                     # Inputs
                    'x': {'abc': 1233, 'def': 1933'},   # State
                    'z': {'v': 3.3, 't': 22.54}         # Outputs
                }, {
                    't': 0.1, ...
                }, ...]
            """
            
            config = { # Defaults
                'step_size': 1,
                'save_freq': 10
            }
            config.update(options)
            # TODO(CT): Add checks (e.g., stepsize, save_freq > 0)

            t = 0
            u = future_loading_eqn(t)
            x = self.initialize(u, first_output)
            times = [t]
            inputs = [u]
            states = [x]
            outputs = [first_output]
            next_save = config['save_freq']
            while t < time:
                t += config['step_size']
                u = future_loading_eqn(t)
                x = self.state(t, x, u, config['step_size'])
                if (t >= next_save):
                    next_save += config['save_freq']
                    times.append(t)
                    inputs.append(u)
                    states.append(x)
                    outputs.append(self.output(t, x))
            if times[-1] != t:
                times.append(t)
                inputs.append(u)
                states.append(x)
                outputs.append(self.output(t, x))
            return {
                't': times,
                'u': inputs,
                'x': states,
                'z': outputs, 
            }