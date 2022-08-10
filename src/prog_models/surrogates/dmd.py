# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from numbers import Number
import numpy as np
from scipy.interpolate import interp1d
import types
from warnings import warn

from ..exceptions import ProgModelInputException
from ..sim_result import SimResult, LazySimResult
from .. import LinearModel, PrognosticsModel
from ..data_models import DataModel

class SurrogateDMDModel(LinearModel, DataModel):
    """
    A subclass of LinearModel that uses Dynamic Mode Decomposition to simulate a system throughout time.
    
    Given an initial state of the system (including internal states, outputs, and event_states), and the expected inputs throuhgout time, this class defines a surrogate model that can approximate the internal states, outputs, and event_states throughout time until threshold is met.

    Args
    ---------
        m : PrognosticsModel 
        load_functions : list of callable functions
            Each index is a callable loading function of (t, x = None) -> z used to predict future loading (output) at a given time (t) and state (x)

    Keyword Args
    ------------
        process_noise : Optional, float or Dict[Srt, float]
        trim_data_to: int, optional
            Value between 0 and 1 that determines fraction of data resulting from simulate_to_threshold that is used to train DMD surrogate model
            e.g. if trim_data_to = 0.7 and the simulated data spans from t=0 to t=100, the surrogate model is trained on the data from t=0 to t=70 \n   
            Note: To trim data to a set time, use the 'horizon' parameter\n   
        stability_tol: int, optional
            Value that determines the tolerance for DMD matrix stability\n
        training_noise: int, optional
            Noise added to the training data sampled from a standard normal distribution with standard deviation of training_noise \n

    See Also
    ---------
        LinearModel, DataModel

    Methods
    ----------
        initialize : 
            Calculate initial state, augmented with outputs and event_states

        next_state : 
            State transition equation: Calculate next state with matrix multiplication (overrides 'dx' defined in LinearModel)

        simulate_to_threshold:
            Simulate prognostics model until defined threshold is met, using simulate_to_threshold defined in PrognosticsModel, then interpolate results to be at user-defined times

    Note
    -------
    This is a first draft of a surrogate model generation using Dynamic Mode Decomposition. 
    DMD does not generate accurate approximations for all models, especially highly non-linear sections, and can be sensitive to the training data time step. 
    In general, the approximation is less accurate if the DMD matrix is unstable. 
    Additionally, this implementation does not yet include all functionalities of DMD (e.g. reducing the system's dimensions through SVD). Further functionalities will be included in future releases. \n
    """

    A = None
    B = None
    C = None
    F = None

    def __new__(cls, dmd_matrix, *args, **kwargs):
        if isinstance(dmd_matrix, PrognosticsModel):
            # Keep for backwards compatability (in first version, model was passed into constructor)
            warn('Passing a PrognosticsModel into SurrogateDMDModel is deprecated and will be removed in v1.5.', DeprecationWarning)
            return cls.from_model(dmd_matrix, args[0], **kwargs) 
        return DataModel.__new__(cls)

    def __init__(self, dmd_matrix, *args, **kwargs):
        if isinstance(dmd_matrix, PrognosticsModel):
            # Initialized in __new__
            # Remove in future version
            return

        params = {
            'dt': None,
            'input_keys': None,
            'state_keys': None,
            'output_keys': None,
            'event_keys': []
        }
        params.update(**kwargs)
        
        if params['input_keys'] is None:
            raise ValueError('input_keys must be specified')
        if params['dt'] is None:
            raise ValueError('dt must be specified')
        if params['state_keys'] is None:
            raise ValueError('state_keys must be specified')
        if params['output_keys'] is None:
            raise ValueError('output_keys must be specified')

        self.inputs = params['input_keys']
        n_inputs = len(params['input_keys'])
        
        self.states = params['state_keys'] + params['output_keys'] + params['event_keys']
        n_states = len(params['state_keys'])
        
        self.outputs = params['output_keys']
        n_outputs = len(params['output_keys'])
        
        self.events = params['event_keys']
        n_events = len(params['event_keys'])
        n_total = n_states + n_outputs + n_events

        self.A = dmd_matrix[:,0:n_total]
        self.B = np.vstack(dmd_matrix[:,n_total:n_total+n_inputs]) 
        self.C = np.zeros((n_outputs,n_total))
        for iter1 in range(n_outputs):
            self.C[iter1,n_states+iter1] = 1 
        self.F = np.zeros((n_events,n_total))
        for iter2 in range(n_events):
            self.F[iter2,n_states+n_outputs+iter2] = 1 
        
        self.dt = params['dt']

        super().__init__(**params)

    @classmethod
    def from_data(cls, times, inputs, states, outputs, event_states = None, **kwargs):

        # Configure
        config = { # Defaults
            'trim_data_to': 1,
            'stability_tol': 1e-05,
            'training_noise': 1e-05,
            'input_keys': None,
            'state_keys': None,
            'output_keys': None,
            'event_keys': None, 
            'dt': None
        }
        config.update(kwargs)

        # Input validation
        if not isinstance(config['trim_data_to'], Number) or config['trim_data_to']>1 or config['trim_data_to']<=0:
            raise ProgModelInputException("Invalid 'trim_data_to' input value, must be between 0 and 1.")
        if not isinstance(config['stability_tol'], Number) or  config['stability_tol'] < 0:
            raise ProgModelInputException(f"Invalid 'stability_tol' input value {config['stability_tol']}, must be a positive number.")
        if not isinstance(config['training_noise'], Number) or config['training_noise'] < 0:
            raise ProgModelInputException(f"Invalid 'training_noise' input value {config['training_noise']}, must be a positive number.")
        if len(inputs) != len(outputs) or len(inputs) != len(states) or (event_states is not None and len(event_states) != len(inputs)):
            raise ProgModelInputException("Must have same number of runs for inputs, states, and outputs")
        if isinstance(config['dt'], list):
            # This means one dt for each run
            # Use mean
            config['dt'] = sum(config['dt'])/len(config['dt'])
        elif config['dt'] == None:
            # Use times from data - calculate mean dt
            dts = [t[j+1] - t[j] for t in times for j in len(t)-1]
            config['dt'] = sum(dts)/len(dts)
        if 'save_freq' not in config:
            config['save_freq'] = config['dt']
        elif isinstance(config['save_freq'], list):
            # This means one save_freq for each run
            # Use mean
            config['save_freq'] = sum(config['save_freq'])/len(config['save_freq'])

        # Initialize lists to hold individual matrices
        x_list = []
        xprime_list = []
        time_list = []

        # Train DMD model
        n_runs = len(inputs)
        for run in range(n_runs):
            t = times[run]
            u = inputs[run]
            x = states[run]
            z = outputs[run]

            if len(u) != len(x) or len(u) != len(z):
                raise ProgModelInputException(f"Must have same number of steps for inputs, states, and outputs in a single run. Not true for run {run}")
            if len(u) == 0:
                raise ProgModelInputException(f"Each run must have at least one timestep, not true for Run {run}")

            # Process times
            if isinstance(t, list):
                t = np.array(t)

            # Process inputs
            if isinstance(u, SimResult):                
                if config['input_keys'] == None:
                    config['input_keys'] = u.keys()
                u = u.to_numpy(config['input_keys'])
            elif config['input_keys'] == None:  # Is numpy array already, but no keys
                config['input_keys'] = ['u{i}' for i in range(u.shape[1])]
            n_inputs = u.shape[1]

            # Process states
            if isinstance(x, SimResult):                
                if config['state_keys'] == None:
                    config['state_keys'] = x.keys()
                x = x.to_numpy(config['state_keys'])
            elif config['state_keys'] == None:  # Is numpy array already, but no keys
                config['state_keys'] = ['x{i}' for i in range(x.shape[1])]
            n_states = x.shape[1]

            # Process outputs
            if isinstance(z, SimResult):                
                if config['output_keys'] == None:
                    config['output_keys'] = z.keys()
                z = z.to_numpy(config['output_keys'])
            elif config['output_keys'] == None:  # Is numpy array already, but no keys
                config['output_keys'] = ['z{i}' for i in range(z.shape[1])]
            n_outputs = z.shape[1]

            # Process events
            if event_states != None:
                es = event_states[run]
                if isinstance(es, SimResult):                
                    if config['event_keys'] == None:
                        config['event_keys'] = es.keys()
                    es = es.to_numpy(config['event_keys'])
                elif config['event_keys'] == None:  # Is numpy array already, but no keys
                    config['event_keys'] = ['event{i}' for i in range(es.shape[1])]    
            else:
                es = np.array([[] for _ in u.shape[0]])   
            n_events = es.shape[1]  

            # Trim
            if config['trim_data_to'] != 1:
                index = int(len(u)*config['trim_data_to'])
                t = t[:index]
                u = u[:index]
                x = x[:index]
                z = z[:index]
                es = es[:index]       
            
            # Interpolate
            # This is done when dt and save_freq are not equal. 
            # dt is the frequency at which simulation is performed to generate data, 
            # while save_freq is the frequency for the dmd_matrix
            if config['save_freq'] != config['dt'] or config['dt'] == None:
                if isinstance(config['save_freq'], (tuple, list, np.ndarray)):
                    # Tuple used to specify start and frequency
                    t_step = config['save_freq'][1]
                    # Use starting time or the next multiple
                    t_start = config['save_freq'][0]
                    start = max(t_start, t[0])
                else:
                    # Otherwise - start is t0
                    t_step = config['save_freq']
                    start = t[0]
                    
                t_new = np.arange(start, t[-1], t_step)
                u = interp1d(t, u, axis=0)(t_new)
                x = interp1d(t, x, axis=0)(t_new)
                z = interp1d(t, z, axis=0)(t_new)
                if n_events != 0:  
                    # Optimization - avoid interpolation of empty array
                    es = interp1d(t, es, axis=0)(t_new)
                t = t_new  # Reset time to new timestep

            # Apply training noise
            if config['training_noise'] != 0:
                n_steps = len(u)
                u += np.random.randn(n_steps, n_inputs)*config['training_noise']
                x += np.random.randn(n_steps, n_states)*config['training_noise']
                z += np.random.randn(n_steps, n_outputs)*config['training_noise']
                if n_events != 0:  
                    # Optimization - avoid generation for empty array
                    es += np.random.randn(n_steps, n_events)*config['training_noise']

            # Initialize DMD matrices
            time_list.append(t)
            x_mat = np.hstack((x, z, es, u)).T
            x_list.append(x_mat[:, :-1])
            xprime_list.append(x_mat[:-n_inputs, 1:] if n_inputs != 0 else x_mat[:, 1:])

        for state_key in config['state_keys']:
            if state_key in config['input_keys'] or state_key in config['output_keys'] or state_key in config['event_keys']:
                config['state_keys'].remove(state_key)
                warn(f"State value '{state_key}' is duplicated in inputs, outputs, or events; duplicate has been removed.")

        for input_key in config['input_keys']:
            if input_key in config['output_keys'] or input_key in config['event_keys']:
                warn(f"Input value '{input_key}' is duplicated in outputs or events")     

        for output_key in config['output_keys']:
            if output_key in config['event_keys']:
                warn(f"Output value '{output_key}' is duplicated in events")    

        # Format training data for DMD and solve for matrix A, in the form X' = AX 
        print('Generate DMD Surrogate Model')
     
        # Convert lists of datasets into arrays, sequentially stacking data in the horizontal direction
        x_mat = np.hstack((x_list[:]))
        xprime_mat = np.hstack((xprime_list[:]))

        # Calculate DMD matrix using the Moore-Penrose pseudo-inverse:
        dmd_matrix = np.dot(xprime_mat,np.linalg.pinv(x_mat))        

        # Check for stability of dmd_matrix
        eig_val, _ = np.linalg.eig(dmd_matrix[:,0:-n_inputs if n_inputs > 0 else None])            
        
        if sum(eig_val>1) != 0:
            for eig_val_i in eig_val:
                if eig_val_i>1 and eig_val_i-1>config['stability_tol']:
                    warn("The DMD matrix is unstable, may result in poor approximation.")

        del config['dt']

        # Build Model
        return cls(
            dmd_matrix, 
            dt = config['save_freq'], 
            **config)
        
    @classmethod
    def from_model(cls, m, load_functions, **kwargs):
        process_noise_temp = {key: 0 for key in m.events}
        config = {
            'add_dt': False,
            'process_noise': {**m.parameters['process_noise'],**m.parameters['measurement_noise'],**process_noise_temp},
            'measurement_noise': m.parameters['measurement_noise'],
            'process_noise_dist': m.parameters.get('process_noise_dist', 'normal'),
            'measurement_noise_dist': m.parameters.get('measurement_noise_dist', 'normal')
        }  # Defaults specific to this class
        config.update(kwargs)
        # Build Model
        m_dmd = super().from_model(m, load_functions, **config)

        # Override initialize for dmd to use base model (since we have it in the original model)
        def init_dmd(self, u=None, z=None):
            x = m.initialize(u,z)
            x.update(m.output(x))
            x.update(m.event_state(x))

            return self.StateContainer(x)
        m_dmd.initialize = types.MethodType(init_dmd, m_dmd)

        return m_dmd

    def initialize(self, u=None, z=None):
        # TODO(CT): WHAT DO WE PUT HERE AS DEFAULT

        return self.StateContainer(x)

    def next_state(self, x, u, _):   
        x.matrix = np.matmul(self.A, x.matrix) + self.E
        if self.B.shape[1] != 0:
           x.matrix += np.matmul(self.B, u.matrix)
        
        return x   

    def simulate_to_threshold(self, future_loading_eqn, first_output = None, threshold_keys = None, **kwargs):
        # Save keyword arguments same as DMD training for approximation 
        kwargs_sim = kwargs.copy()
        kwargs_sim['save_freq'] = self.dt
        kwargs_sim['dt'] = self.dt

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

        if (config['save_freq'] == self.dt or
            (isinstance(config['save_freq'], tuple) and
                config['save_freq'][0]%self.dt < 1e-9 and
                config['save_freq'][1] == self.dt)
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
