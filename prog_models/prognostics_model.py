from . import model 
from abc import ABC, abstractmethod

class PrognosticsModel(model.Model, ABC):
    """
    A general time-variant state space model of system degradation behavior.

    The PrognosticsModel class is a wrapper around a mathematical model of a
    system as represented by a state, output, input, and threshold equations.
    It is a subclass of the Model class, with the addition of a threshold
    equation, which defines when some condition, such as end-of-life, has
    been reached.
    """

    events = []

    @abstractmethod
    def event_state(self, t, x : dict) -> dict:
        """
        Calculate event states (i.e., measures of progress towards event (0-1, where 0 means event has occured))

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
        event_state : dict
            Event States, with keys defined by prognostics_model.events.
            e.g., event_state = {'EOL':0.32} given events = ['EOL']
        """

        pass
    
    @abstractmethod
    def threshold_met(self, t, x : dict) -> dict:
        """
        For each event threshold, calculate if it has been met

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
        thresholds_met : dict
            If each threshold has been met (bool), with deys defined by prognostics_model.events
            e.g., thresholds_met = {'EOL': False} given events = ['EOL']
        """

        pass

    def simulate_to(self, time, future_loading_eqn, first_output : dict, options : dict = {}):
        """
            Simulate prognostics model for a given time interval

            Similar to model.simulate_to, only includes event_state

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
                    'event_state': {'EOL': 0.23}        # Event States
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
        event_states = [self.event_state(t, x)]
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
                event_states.append(self.event_state(t, x))
        if times[-1] != t:
            times.append(t)
            inputs.append(u)
            states.append(x)
            outputs.append(self.output(t, x))
            event_states.append(self.event_state(t, x))
        return {
            't': times,
            'u': inputs,
            'x': states,
            'z': outputs, 
            'event_state': event_states
        }
 
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
            
            Returns
            -------
            results : dict
                Results recorded during simulation
                e.g., results = [{
                    't': 0,                             # Time (s)
                    'u': {'i': 12},                     # Inputs
                    'x': {'abc': 1233, 'def': 1933'},   # State
                    'z': {'v': 3.3, 't': 22.54}         # Outputs
                    'event_state': {'EOL': 0.23}        # Event States
                }, {
                    't': 0.1, ...
                }, ...]
            """

        config = { # Defaults
            'step_size': 1,
            'save_freq': 10,
            'horizon': 1e100 # Default horizon (in s), essentially inf
        }
        config.update(options)
        # TODO(CT): Add checks (e.g., stepsize, save_freq > 0)

        t = 0
        u = future_loading_eqn(t)
        if 'x' in config:
            x = config['x']
        else:
            x = self.initialize(u, first_output)
        times = [t]
        inputs = [u]
        states = [x]
        outputs = [first_output]
        event_states = [self.event_state(t, x)]
        next_save = config['save_freq']
        threshold_met = False
        while not threshold_met and t < config['horizon']:
            t += config['step_size']
            u = future_loading_eqn(t)
            x = self.state(t, x, u, config['step_size'])
            thresholds_met = self.threshold_met(t, x)
            threshold_met = any(thresholds_met.values())
            if (t >= next_save):
                next_save += config['save_freq']
                times.append(t)
                inputs.append(u)
                states.append(x)
                outputs.append(self.output(t, x))
                event_states.append(self.event_state(t, x))
        if times[-1] != t:
            times.append(t)
            inputs.append(u)
            states.append(x)
            outputs.append(self.output(t, x))
            event_states.append(self.event_state(t, x))
        return {
            't': times,
            'u': inputs,
            'x': states,
            'z': outputs, 
            'event_state': event_states
        }