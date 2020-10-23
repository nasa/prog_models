from . import model, ProgModelInputException
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

        pass
    
    @abstractmethod
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

        pass

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
        if time < 0:
            raise ProgModelInputException("'time' must be ≥ 0, was {}".format(time))

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
        times = [t]
        inputs = [u]
        states = [x]
        outputs = [first_output]
        event_states = [self.event_state(t, x)]
        dt = config['dt'] # saving to optimize access in while loop
        save_freq = config['save_freq']
        next_save = save_freq
        threshold_met = False

        # Simulate
        while not threshold_met and t < config['horizon']:
            t += dt
            u = future_loading_eqn(t)
            x = self.next_state(t, x, u, dt)
            thresholds_met = threshold_met_eqn(t, x)
            threshold_met = any(thresholds_met.values())
            if (t >= next_save):
                next_save += save_freq
                times.append(t)
                inputs.append(u)
                states.append(x)
                outputs.append(self.output(t, x))
                event_states.append(self.event_state(t, x))

        # Save final state
        if times[-1] != t:
            # This check prevents double recording when the last state was a savepoint
            times.append(t)
            inputs.append(u)
            states.append(x)
            outputs.append(self.output(t, x))
            event_states.append(self.event_state(t, x))
        
        return (times, inputs, states, outputs, event_states)