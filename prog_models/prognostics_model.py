# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from . import model, ProgModelInputException, ProgModelTypeError
from abc import abstractmethod
from numbers import Number
from numpy import array, append
from copy import deepcopy

class PrognosticsModel(model.Model):
    """
    A general time-variant state space model of system degradation behavior.

    The PrognosticsModel class is a wrapper around a mathematical model of a
    system as represented by a state, output, input, and threshold equations.
    It is a subclass of the Model class, with the addition of a threshold
    equation, which defines when some condition, such as end-of-life, has
    been reached.
    """

    events = [] # Identifiers for each event

    def __init__(self, options = {}):
        self.parameters.update(options)

        super().__init__()

    def event_state(self, t, x) -> dict:
        """
        Calculate event states (i.e., measures of progress towards event (0-1, where 0 means event has occured))

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
        event_state : dict
            Event States, with keys defined by prognostics_model.events.
            e.g., event_state = {'EOL':0.32} given events = ['EOL']
        """

        return {}
    
    def threshold_met(self, t, x) -> dict:
        """
        For each event threshold, calculate if it has been met

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
        thresholds_met : dict
            If each threshold has been met (bool), with deys defined by prognostics_model.events
            e.g., thresholds_met = {'EOL': False} given events = ['EOL']
        """

        return {key: event_state < 0 for (key, event_state) in self.event_state(t, x).items()} 

    def simulate_to(self, time, future_loading_eqn, first_output, options = {}):
        """
            Simulate prognostics model for a given time interval

            Similar to model.simulate_to, only includes event_state

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
            event_states: [dict]
                Estimated event state (e.g., SOH), between 1-0 where 0 is event occurance, for each time in times
            
            Example
            -------
            (times, inputs, states, outputs, event_states) = m.simulate_to(200, future_load_eqn, first_output)
            """
        
        # Input Validation
        if not isinstance(time, Number) or time <= 0:
            raise ProgModelInputException("'time' must be number greater than 0, was {} ({})".format(time, type(time)))

        # Configure 
        config = { # Defaults
            'dt': 1,
            'save_freq': 10,
            'threshold_eqn': (lambda t,x : {'a': False}), # Override threshold
            'horizon': time
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

        return self.simulate_to_threshold(future_loading_eqn, first_output, config)
 
    def simulate_to_threshold(self, future_loading_eqn, first_output, options = {}):
        """
            Simulate prognostics model until at least any threshold has been met

            Parameters
            ----------
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
            event_states: [dict]
                Estimated event state (e.g., SOH), between 1-0 where 0 is event occurance, for each time in times
            
            Example
            -------
            (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load_eqn, first_output)
            """
        # Input Validation
        if not all(key in first_output for key in self.outputs):
            raise ProgModelInputException("Missing key in 'first_output', must have every key in model.outputs")

        if not (callable(future_loading_eqn)):
            raise ProgModelInputException("'future_loading_eqn' must be callable f(t)")

        # Configure
        config = { # Defaults
            'dt': 1.0,
            'save_freq': 10,
            'horizon': 1e100 # Default horizon (in s), essentially inf
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

        # TODO(CT): Add checks (e.g., stepsize, save_freq > 0)
        if 'threshold_eqn' in config:
            # Override threshold_met eqn
            threshold_met_eqn = config['threshold_eqn']
        else:
            threshold_met_eqn = self.threshold_met
        
        # Setup
        t = 0
        u = future_loading_eqn(t)
        if 'x' in config:
            x = config['x']
        else:
            x = self.initialize(u, first_output)
        times = array([t])
        inputs = array([u])
        states = array([deepcopy(x)]) # Avoid optimization where x is not copied
        outputs = array([self.output(t, x)])
        event_states = array([self.event_state(t, x)])
        dt = config['dt'] # saving to optimize access in while loop
        save_freq = config['save_freq']
        horizon = config['horizon']
        next_save = save_freq
        threshold_met = False

        # Optimization
        next_state = self.next_state
        output = self.output
        event_state = self.event_state

        # Simulate
        while not threshold_met and t < horizon:
            t += dt
            u = future_loading_eqn(t)
            x = next_state(t, x, u, dt)
            thresholds_met = threshold_met_eqn(t, x)
            threshold_met = any(thresholds_met.values())
            if (t >= next_save):
                next_save += save_freq
                times = append(times,t)
                inputs = append(inputs,u)
                states = append(states,deepcopy(x))
                outputs = append(outputs,output(t, x))
                event_states = append(event_states,event_state(t, x))

        # Save final state
        if times[-1] != t:
            # This check prevents double recording when the last state was a savepoint
            times = append(times,t)
            inputs = append(inputs,u)
            states = append(states,x)
            outputs = append(outputs,self.output(t, x))
            event_states = append(event_states,self.event_state(t, x))
        
        return (times, inputs, states, outputs, event_states)
    
    @staticmethod
    def generate_model(keys, initialize_eqn, next_state_eqn, output_eqn, event_state_eqn = None, threshold_eqn = None, config = {'process_noise': 0.1}):
        """
        Generate a new prognostics model from functions


        """
        # Input validation
        if not callable(initialize_eqn):
            raise ProgModelTypeError("Initialize Function must be callable")

        if not callable(next_state_eqn):
            raise ProgModelTypeError("Next_State Function must be callable")

        if not callable(output_eqn):
            raise ProgModelTypeError("Output Function must be callable")

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
            def next_state():
                pass
            def output():
                pass

        m = NewProgModel(config)

        m.initialize = initialize_eqn
        m.next_state = next_state_eqn
        m.output = output_eqn

        if 'events' in keys:
            m.events = keys['events']
        if event_state_eqn:
            m.event_state = event_state_eqn
        if threshold_eqn:
            m.threshold_met = threshold_eqn

        return m