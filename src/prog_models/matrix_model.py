# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from .exceptions import ProgModelInputException
from . import PrognosticsModel
from .sim_result import SimResult, LazySimResult
from .utils import ProgressBar

from abc import ABC, abstractmethod
from array import array
from collections import abc
from numbers import Number
import numpy as np


class MatrixModel(PrognosticsModel, ABC):
    def initialize_matrix(self, u = None, z = None):
        pass

    def initialize(self, u = None, z = None):
        z_mat = np.vstack((z[key] for key in self.outputs))
        u_mat = np.vstack((u[key] for key in self.inputs))
        x = self.initialize_matrix(u_mat, z_mat)
        return {key: x_i for key, x_i in zip(self.states, x)}

    def dx_matrix(self, x, u):
        pass

    def next_state_matrix(self, x, u, dt):
        return self.dx_matrix(x, u) * dt

    def next_state(self, x, u, dt):
        x_mat = np.vstack((x[key] for key in self.states))
        u_mat = np.vstack((u[key] for key in self.inputs))
        x_next = self.next_state_matrix(x_mat, u_mat, dt)
        return {state: x_next_i for (state, x_next_i) in zip(self.states, x_next)}

    @abstractmethod
    def output_matrix(self, x):
        pass

    def output(self, x):
        x_mat = np.vstack((x[key] for key in self.states))
        z = self.output(x_mat)
        return {output: z_i for (output, z_i) in zip(self.outputs, z)}

    def event_state_matrix(self, x):
        pass

    def event_state(self, x):
        x_mat = np.vstack((x[key] for key in self.states))
        es = self.event_state_matrix(x_mat)
        return {event: es_i for (event, es_i) in zip(self.events, es)}

    def threshold_matrix(self, x):
        es = self.event_state_matrix(x)
        return  es[es <= 0]

    def threshold(self, x):
        x_mat = np.vstack((x[key] for key in self.states))
        t_met = self.threshold_matrix(x_mat)
        return {event: t_met_i for (event, t_met_i) in zip(self.events, t_met)}

    def simulate_to_threshold(self, future_loading_eqn, first_output = None, threshold_keys = None, **kwargs):
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
        next_state = self.next_state_matrix # TODO: noise, limits
        output = self.output_matrix # TODO: noise, limits
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
        saved_times = array('d')
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
                saved_inputs.append({key: u_i for key, u_i in zip(self.inputs, u)})
                saved_states.append({key: x_i for key, x_i in zip(self.states, x)})
                saved_outputs.append({key: z_i for key, z_i in zip(self.outputs, output(x))})
                saved_event_states.append({key: es_i for key, es_i in zip(self.events, event_state(x))})
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
                saved_inputs.append({key: u_i for key, u_i in zip(self.inputs, u)})
                saved_states.append({key: x_i for key, x_i in zip(self.states, x)})

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
