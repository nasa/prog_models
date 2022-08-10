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
            'dt': 1
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
        if 'save_freq' not in config:
            config['save_freq'] = config['dt']

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
                if n_events != 0:  # Optimization - avoid interpolation of empty array
                    es = interp1d(t, es, axis=0)(t_new)
                t = t_new  # Reset time to new timestep

            # Apply training noise
            if config['training_noise'] != 0:
                u += np.random.randn(n_inputs)*config['training_noise']
                x += np.random.randn(n_states)*config['training_noise']
                z += np.random.randn(n_outputs)*config['training_noise']
                if n_events != 0:
                    es += np.random.randn(n_events)*config['training_noise']

            # Initialize DMD matrices
            time_list.append(t)
            x_mat = np.hstack((x, z, es, u)).T
            x_list.append(x_mat[:, :-1])
            xprime_list.append(x_mat[:-n_inputs, 1:] if n_inputs != 0 else x_mat[:, 1:])

        # Format training data for DMD and solve for matrix A, in the form X' = AX 
        print('Generate DMD Surrogate Model')

        # TODO(CT):Trim 
     
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

        del kwargs['input_keys']
        del kwargs['state_keys']
        del kwargs['output_keys']
        del kwargs['event_keys']
        del kwargs['dt']

        # Build Model
        return cls(
            dmd_matrix, 
            dt = kwargs['save_freq'], 
            **kwargs)
        
    @classmethod
    def from_model2(cls, m, load_functions, **kwargs):
        # Configure
        config = { # Defaults
            'trim_data_to': 1,
            'stability_tol': 1e-05,
            'training_noise': 1e-05
        }
        config.update(kwargs)

        if not isinstance(config['trim_data_to'], Number) or config['trim_data_to']>1 or config['trim_data_to']<=0:
            raise ProgModelInputException("Invalid 'trim_data_to' input value, must be between 0 and 1.")
        if not isinstance(config['stability_tol'], Number) or  config['stability_tol'] < 0:
            raise ProgModelInputException(f"Invalid 'stability_tol' input value {config['stability_tol']}, must be a positive number.")
        if not isinstance(config['training_noise'], Number) or config['training_noise'] < 0:
            raise ProgModelInputException(f"Invalid 'training_noise' input value {config['training_noise']}, must be a positive number.")

        states_dmd = m.states.copy()
        inputs_dmd = m.inputs.copy()
        outputs_dmd = m.outputs.copy()
        events_dmd = m.events.copy()

        # Initialize lists to hold individual matrices
        x_list = []
        xprime_list = []
        time_list = []

        # Generate Data to train surrogate model: 
        for iter_load, load_fcn_now in enumerate(load_functions):
            print('Generating training data: loading profile {} of {}'.format(iter_load+1, len(load_functions)))

            # Simulate to threshold 
            (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load_fcn_now, **config)
        
            # Interpolate results to time step of save_freq
            time_data_interp = np.arange(times[0], times[-1], config['save_freq'])

            states_data_interp = {}
            outputs_data_interp = {}
            es_data_interp = {}

            for state_name in m.states:
                states_data_temp = [states[iter_data1][state_name] for iter_data1 in range(len(states))]
                states_data_interp[state_name] = interp1d(times,states_data_temp)(time_data_interp)
                if config['training_noise'] != 0:
                    states_data_interp[state_name] += np.random.randn(len(states_data_interp[state_name]))*config['training_noise']
            for output_name in m.outputs:
                outputs_data_temp = [outputs[iter_data3][output_name] for iter_data3 in range(len(outputs))]
                outputs_data_interp[output_name] = interp1d(times,outputs_data_temp)(time_data_interp)
                if config['training_noise'] != 0:
                    outputs_data_interp[output_name] += np.random.randn(len(outputs_data_interp[output_name]))*config['training_noise']
            for es_name in m.events:
                es_data_temp = [event_states[iter_data2][es_name] for iter_data2 in range(len(event_states))]
                es_data_interp[es_name] = interp1d(times,es_data_temp)(time_data_interp)
                if config['training_noise'] != 0:
                    es_data_interp[es_name] += np.random.randn(len(es_data_interp[es_name]))*config['training_noise']

            states_data = [
                m.StateContainer({
                    key: value[iter_dataT] for key, value in states_data_interp.items()
                }) for iter_dataT in range(len(time_data_interp))
                ]
            outputs_data = [
                m.OutputContainer({
                    key: value[iter_dataT] for key, value in outputs_data_interp.items()
                }) for iter_dataT in range(len(time_data_interp))
                ]
            event_states_data = [
                {key: value[iter_dataT] for key, value in es_data_interp.items()} for iter_dataT in range(len(time_data_interp))
                ]

            times = time_data_interp.tolist()
            states = SimResult(time_data_interp,states_data)
            outputs = SimResult(time_data_interp,outputs_data) 
            event_states = SimResult(time_data_interp,event_states_data)
            
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

            for state_key in config['state_keys']:
                if state_key in config['input_keys'] or state_key in config['output_keys'] or state_key in config['event_keys']:
                    config['state_keys'].remove(state_key)
                    warn(f"State value '{state_key}' is duplicated in inputs, outputs, or events; duplicate has been removed.")

            for input_key in config['input_keys']:
                if input_key in config['output_keys'] or input_key in config['event_keys']:
                    warn(f"Input value '{input_key}' is duplicated in outputs or events; duplicate has not been removed.")     

            for output_key in config['output_keys']:
                if output_key in config['event_keys']:
                    warn(f"Output value '{output_key}' is duplicated in events; duplicate has not been removed.")               
                                  
            if len(config['state_keys']) != len(m.states):
                user_val_set(m.states, 'state_keys', states_dmd, states)
            if len(config['output_keys']) != len(m.outputs):
                user_val_set(m.outputs, 'output_keys', outputs_dmd, outputs) 
            if len(config['event_keys']) != len(m.events):
                user_val_set(m.events, 'event_keys', events_dmd, event_states)  

            # Initialize DMD matrices
            x_mat_temp = np.zeros((len(states[0])+len(outputs[0])+len(event_states[0])+len(config['input_keys']), len(times))) 

            # Save DMD matrices
            for i, time in enumerate(times): 
                time_now = time + np.divide(config['save_freq'],2) 
                load_now = load_fcn_now(time_now) # Evaluate load_function at (t_now + t_next)/2 to be consistent with next_state implementation
                if len(config['input_keys']) != len(m.inputs): # Delete any input values not specified by user to be included in surrogate model 
                    for key in m.inputs:
                        if key not in config['input_keys']:
                            del load_now[key]

                states_now = states[i].matrix 
                stack = (
                        states_now,
                        outputs[i].matrix,
                        np.array([list(event_states[i].values())]).T,
                        np.array([[load_now[key]+ np.random.randn()*config['training_noise']] for key in load_now.keys()])
                    )
                x_mat_temp[:,i] = np.vstack(tuple(v for v in stack if v.shape != (0, )))[:,0]  # Filter out empty values (e.g., if there is no input)
            # Get X and Xprime matrices by shifting time index by 1
            x_mat_ = x_mat_temp[:, :-1] # X
            if len(inputs[0])==0:   xprime_mat_ = x_mat_temp[:, 1:]                 # Xprime if no input (len(inputs)==0 messes things up)
            else:                   xprime_mat_ = x_mat_temp[:-len(inputs[0]), 1:]  # Xprime if there's an input 

            # Save matrices in list, where each index in list corresponds to one of the user-defined loading equations 
            x_list.append(x_mat_)
            xprime_list.append(xprime_mat_)
            time_list.append(times)

        # 2 in each list. (13, 2345), (13, 1885)

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

        # Check for stability of dmd_matrix
        num_inputs = len(inputs[0].matrix)
        eig_val, _ = np.linalg.eig(dmd_matrix[:,0:-num_inputs if num_inputs > 0 else None])            
        
        if sum(eig_val>1) != 0:
            for eig_val_i in eig_val:
                if eig_val_i>1 and eig_val_i-1>config['stability_tol']:
                    warn("The DMD matrix is unstable, may result in poor approximation.")

        process_noise_temp = {key: 0 for key in m.events}  # Process noise for event states is zero

        params = {
            'process_noise': {**m.parameters['process_noise'],**m.parameters['measurement_noise'],**process_noise_temp},
                'measurement_noise': m.parameters['measurement_noise'],
                'process_noise_dist': m.parameters.get('process_noise_dist', 'normal'),
                'measurement_noise_dist': m.parameters.get('measurement_noise_dist', 'normal')
        }
        params.update(kwargs)
        del params['input_keys']
        del params['state_keys']
        del params['output_keys']
        del params['event_keys']
        del params['dt']

        # Build Model
        m_dmd = cls(
            dmd_matrix, 
            dt = params['save_freq'], 
            input_keys = inputs_dmd,
            state_keys = states_dmd,
            output_keys = outputs_dmd,
            event_keys = events_dmd,
            **params)

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
