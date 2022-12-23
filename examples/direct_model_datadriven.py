# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example demonstrates the Direct Models functionality with a data-driven model. 

"""

import numpy as np
import tensorflow as tf
from abc import ABC, abstractclassmethod

from prog_models.models import BatteryElectroChemEOD
from prog_models.prognostics_model import PrognosticsModel
from prog_models.data_models import DataModel

import matplotlib.pyplot as plt

class DirectDataModel(PrognosticsModel, ABC):

    @abstractclassmethod
    def from_data(cls, **kwargs) -> "DirectDataModel":
        """
        Create a Direct Data Model from data. 
        This class is overwritten by specific data-driven classes (e.g., :py:class:`LSTMStateTransitionModel`)

        Keyword Arguments:
            times (list[list]): list of input data for use in data. Each element is the times for a single run of size (n_times)
            inputs (list[np.array]): list of :term:`input` data for use in data. Each element is the inputs for a single run of size (n_times, n_inputs)
            states (list[np.array]): list of :term:`state` data for use in data. Each element is the states for a single run of size (n_times, n_states)
            outputs (list[np.array]): list of :term:`output` data for use in data. Each element is the outputs for a single run of size (n_times, n_outputs)
            event_states (list[np.array]): list of :term:`event state` data for use in data. Each element is the event states for a single run of size (n_times, n_event_states)
            input_keys (list[str]): 
                List of :term:`input` keys
            state_keys (list[str]): 
                List of :term:`state` keys
            output_keys (list[str]): 
                List of :term:`output` keys
            event_keys (list[str]): 
                List of :term:`event` keys
        
        See specific data class for more additional keyword arguments

        Returns:
            DataModel: Trained PrognosticsModel

        Example:
            |

                >>> # Replace DataModel with specific classname below
                >>> m = DataModel.from_data(data)
        """
        pass

    @staticmethod
    def check_data_format(inputs, event_times):
        if len(inputs) == 0:
            raise ValueError("No data provided. inputs must be in format [run1_inputs, ...] and have at least one element")
        if len(inputs) != len(event_times):
            raise ValueError("Inputs must be same length as event_times")
        pass

    @classmethod
    def from_model(cls, m: PrognosticsModel, load_functions: list, **kwargs) -> "DirectDataModel":
        """
        """
         # Configure
        config = { # Defaults
            'add_dt': True,
            'input_keys': m.inputs.copy(),
            'output_keys': m.outputs.copy(),
            'state_keys': m.states.copy(),
            'event_keys': m.events.copy()
        }
        config.update(kwargs)

        if callable(load_functions):
            # Only one function
            load_functions = [load_functions]

        sim_cfg_params = ['dt', 't0', 'integration_method', 'save_freq', 'save_pts', 'horizon', 'first_output', 'threshold_keys', 'x', 'thresholds_met_eqn']

        # Check format of cfg item and split into cfg for each sim if scalar
        for cfg in sim_cfg_params:
            if cfg in config:
                if np.isscalar(config[cfg]) or isinstance(config[cfg], tuple) or callable(config[cfg]):
                    # Single element to be applied to all
                    config[cfg] = [config[cfg] for _ in load_functions]
                elif len(config[cfg]) != len(load_functions):
                    raise ValueError(f"If providing multiple values for sim config item, must provide the same number of values as number of load functions. For {cfg} provided {len(config[cfg])}, expected {len(load_functions)}.")
                # Otherwise, already in correct form

        if 'dt' not in config:
            config['dt'] = [1.0 for _ in load_functions]  # default
        if 'save_freq' not in config:
            config['save_freq'] = config['dt']

        # Create sim config for each element
        sim_cfg = [{
            cfg : config[cfg][i]
               for cfg in sim_cfg_params if cfg in config
        } for i in range(len(load_functions))]

        # Simulate            
        data = [m.simulate_to_threshold(load, **sim_cfg[i]) for (i, load) in enumerate(load_functions)]
        time_of_events = m.time_of_event(input_values, future_loading_eqn = load_functions)
        
        # Prepare data
        times = [d.times for d in data]
        if config['add_dt']:
            config['input_keys'].append('dt')
            if len(data[0].inputs) > 0 and len(data[0].inputs[0]) == 0:
                # No inputs
                inputs = [np.array([[config['dt'][i]] for _ in data[i].inputs], dtype=float) for i in range(len(data))]
            else:
                inputs = [np.array([np.hstack((u_i.matrix[:][0].T, [config['dt'][i]])) for u_i in d.inputs], dtype=float) for i, d in enumerate(data)]
        else:
            inputs = [d.inputs for d in data]
        states = [d.states for d in data]
        t_met = [[list(m.threshold_met(x).values()) for x in state] for state in states]
        # this is to be checked: t_met == event_time (from time_of_event)?
        return cls.from_data(input_data = inputs, event_time_data=t_met, **config)


# class DirectFCN(DataModel):
class DirectFCN(DirectDataModel):
    default_params = {
            'process_noise': 0,  # Default 0 noise
            'measurement_noise': 0,  # Default 0 noise
        }
    
    def __init__(self, inputs : list, trained_model=None, **kwargs):
        self.model = trained_model
        self.states = ['',]
        self.inputs = inputs
        super().__init__(**kwargs)
    
    def time_of_event(self, x, *args, **kwargs):
        # calculate time when object hits ground given x['x'] and x['v']
        (x_norm, _) = self.__normalize_data(inputs=x, inputs_stats=self.inputs_stats, normalization_type=self.normalization_type)
        xnorm_tf = tf.convert_to_tensor(x_norm)
        
        # THis line doesn't work because model has not been assigned to self.model.
        # I don't know how the classmethod works, and that's why I haven't been able to correctly assign model to self.model.
        # if I assign it in the __init__, then I cannot access is in classmethod 'from_data'.
        # if I assign it within the classmethod 'from_data', it's not accessible through self later on.
        ynorm_tf = self.model(xnorm_tf)

        ynorm = ynorm_tf.numpy()
        (_, y) = self.__unnormalize_data(inputs_norm=None, event_times_norm=ynorm, normalization_type=self.normalization_type, event_times_stats=self.event_times_stats)
        return {'EOD': y}
    
    @staticmethod
    def __normalize_data(inputs=None, event_times=None, normalization_type='normal', inputs_stats=None, event_times_stats=None):
        
        assert inputs_stats is not None, "input_stats not provided. Must be a dictionary with statical values of inputs data for normalization."

        inputs_norm = inputs.copy()
        _, m = inputs_norm.shape

        if event_times is not None:
            assert event_times_stats is not None, "event_times_stats not provided. Must be a dictionary with statical values of event_times data for normalization."
            event_times_norm = event_times.copy()
        else:
            event_times_norm = None

        if any([normalization_type == name for name in ['normal', 'norm', 'gauss', 'gaussian']]):
            for i in range(m):
                inputs_norm[:, i]      = (inputs[:, i] - inputs_stats['m'][i]) / inputs_stats['s'][i]
            if event_times is not None:
                for i in range(m):
                    event_times_norm[:, i] = (event_times[:, i] - event_times_stats['m'][i]) / event_times_stats['s'][i]
        elif any([normalization_type == name for name in ['uniform', 'unif', 'minmax', 'maxmin']]):
            for i in range(m):
                inputs_norm[:, i]      = (inputs[:, i] - inputs_stats['min'][i]) / (inputs_stats['max'][i] - inputs_stats['min'][i])
            if event_times is not None:
                for i in range(m):
                    event_times_norm[:, i] = (event_times[:, i] - event_times_stats['min'][i]) / (event_times_stats['max'][i] - event_times_stats['min'][i])
        else:
            raise Exception(f'Normalization type {normalization_type} not recognized. Available options are Normal (normal, norm, gauss, Gaussian) or Uniform (uniform, unif, minmax, maxmin).')
        return (inputs_norm, event_times_norm)

    @staticmethod
    def __unnormalize_data(inputs_norm=None, event_times_norm=None, normalization_type='normal', inputs_stats=None, event_times_stats=None):
        if inputs_norm is not None:
            _, m = inputs_norm.shape
            inputs = inputs_norm.copy()
        if event_times_norm is not None:
            event_times = event_times_norm.copy()
            _, m = event_times_norm.shape
        else:
            event_times = None
        if any([normalization_type == name for name in ['normal', 'norm', 'gauss', 'gaussian']]):
            if inputs_norm is not None:
                for i in range(m):
                    inputs[:, i] = inputs_norm[:, i] * inputs_stats['s'][i] + inputs_stats['m'][i]
            if event_times_norm is not None:
                for i in range(m):
                    event_times[:, i] = event_times_norm[:, i] * event_times_stats['s'][i] + event_times_stats['m'][i]
        elif any([normalization_type == name for name in ['uniform', 'unif', 'minmax', 'maxmin']]):
            if inputs_norm is not None:
                for i in range(m):
                    inputs[:, i] = inputs_norm[:, i] * (inputs_stats['max'][i] - inputs_stats['min'][i]) + inputs_stats['min'][i]
            if event_times_norm is not None:
                for i in range(m):
                    event_times[:, i] = event_times_norm[:, i] * (event_times_stats['max'][i] - event_times_stats['min'][i]) + event_times_stats['min'][i]
        else:
            raise Exception(f'Normalization type {normalization_type} not recognized. Available options are Normal (normal, norm, gauss, Gaussian) or Uniform (uniform, unif, minmax, maxmin).')  
        return (inputs, event_times)

    @staticmethod
    def __compute_data_stats(inputs, event_times, normalization_type, **kwargs):
        
        DataModel.check_data_format(inputs, event_times)
        
        if any([normalization_type.lower()==name for name in ['normal', 'norm', 'gauss', 'gaussian']]):
            inputs_stats     = {'m': np.mean(inputs, axis=0),      's': np.std(inputs, axis=0)}
            event_time_stats = {'m': np.mean(event_times, axis=0), 's': np.std(event_times, axis=0)}
        elif any([normalization_type.lower() == name for name in ['uniform', 'unif', 'minmax', 'maxmin']]):
            inputs_stats     = {'min': np.min(inputs, axis=0), 'max': np.max(inputs, axis=0)}
            event_time_stats = {'min': np.min(inputs, axis=0), 'max': np.max(inputs, axis=0)}
        else:
            raise Exception(f'Normalization type {normalization_type} not recognized. Available options are Normal (normal, norm, gauss, Gaussian) or Uniform (uniform, unif, minmax, maxmin).')

        return (inputs_stats, event_time_stats)

    @classmethod
    def from_data(cls, model, input_data, event_time_data, **kwargs):

        params = {
            'normalize': True,
            'normalization_dist': 'normal', 
            'validation_split': 0.3,
            'epochs': 100,
            'prediction_steps': 1,
            'batch_size': 1,
            'early_stop': True,
            'early_stop.cfg': {'patience': 3, 'monitor': 'loss'},
            'workers': 1
        }
        params.update(DirectFCN.default_params)
        params.update(kwargs)

        input_data = input_data.copy()
        event_time_data = event_time_data.copy()

        # Build model
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
        ]
        
        if params['early_stop']:
            callbacks.append(tf.keras.callbacks.EarlyStopping(**params['early_stop.cfg']))

        if params['normalize']:
            DirectFCN.normalization_type = params['normalization_dist'].lower()
            (DirectFCN.inputs_stats, DirectFCN.event_times_stats) = DirectFCN.__compute_data_stats(inputs=input_data, event_times=event_time_data, normalization_type=DirectFCN.normalization_type)
            (input_data, event_time_data) = DirectFCN.__normalize_data(inputs=input_data, event_times=event_time_data, 
                                                                       inputs_stats=DirectFCN.inputs_stats, 
                                                                       event_times_stats=DirectFCN.event_times_stats, 
                                                                       normalization_type=DirectFCN.normalization_type)

        # Divide in batches
        # input_data = input_data.reshape((params['batch_size'], input_data.shape[1], 1))
        # event_time_data = event_time_data.reshape((params['batch_size'], event_time_data.shape[1], 1))

        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
        event_time_data = tf.convert_to_tensor(event_time_data, dtype=tf.float32)

        # Assign model:
        DirectFCN.model = model
        
        # Compile model
        DirectFCN.model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        
        # Train model
        history = DirectFCN.model.fit(input_data, 
                                      event_time_data, 
                                      epochs=params['epochs'],
                                      validation_split=params['validation_split'],
                                      callbacks=callbacks)
        DirectFCN.model = tf.keras.models.load_model("best_model.keras")
        return history

        
if __name__ == '__main__':

    print(' Building a Data-Driven Direct Model with ProgPy')
    
    # DATA GENERATION
    # ===============
    DT = 2.0            # Time step used in simulation
    N_SAMPLES = 20     # Samples to be used for training/testing

    # Generate Battery EOD model
    # --------------------------
    m  = BatteryElectroChemEOD(process_noise = 0.004) 
    x0 = m.initialize()  # Initial State
    
    # Simulate battery model to collect time_of_event data
    # ----------------------------------------------------
    time_of_event = np.empty((N_SAMPLES, len(m.events)), dtype=np.float64)
    input_values  = np.empty((N_SAMPLES, len(m.inputs)), dtype=np.float64)
    print('Generating data from battery discharge sims ... ', end=' ')
    for i in range(N_SAMPLES):
        current = np.random.uniform(low=1.0, high=4.0)
        # Generate load function
        def future_loading(t, x=None):
            return m.InputContainer({'i': current})
        
        # Simulate to get data
        time_of_event_i = m.time_of_event(x0, future_loading, dt=DT)
        # Store data
        time_of_event[i] = [time_of_event_i[key] for key in m.events]
        input_values[i]  = current
    print('complete')


    # Split into training and testing
    # --------------------------------
    train_size = 0.7
    dataset_index_list = np.linspace(0, int(len(input_values))-1, int(len(input_values)), dtype=int)
    n_train   = int(np.floor(len(input_values) * train_size))
    n_test    = int(len(input_values) - n_train)
    msk_train = np.random.choice(dataset_index_list, size=n_train, replace=False)
    msk_test  = np.delete(dataset_index_list, msk_train)
    input_values_train  = input_values[msk_train]
    time_of_event_train = time_of_event[msk_train]
    input_values_test   = input_values[msk_test]
    time_of_event_test  = time_of_event[msk_test]
    
    # DATA-DRIVEN MODEL GENERATION
    # =============================

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=3, input_shape=(1,), activation='tanh', use_bias=True, bias_initializer='random_normal'))
    model.add(tf.keras.layers.Dense(units=7, activation='tanh', use_bias=True))
    model.add(tf.keras.layers.Dense(units=1, use_bias=True))
    
    # DATA DRIVE MODEL TRAINING AND TESTING
    # ======================================    
    # Training using from_data method from DirectFCN
    eol_model = DirectFCN(inputs=['i',])
    eol_model.from_data(model, input_values_train, time_of_event_train, 
                        normalize=True, 
                        normalization_dist='Gauss', 
                        epochs=250)

    # Testing
    # ------
    eol_data_tf = eol_model.time_of_event(input_values_test)
    eol_data = eol_data_tf['EOD']

    # Visualize the data
    fig = plt.figure(figsize=(12, 8))
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    ax1.plot(input_values_train.reshape((-1,)), time_of_event_train.reshape((-1,)), 'b.', label='train')
    ax2.plot(input_values_test.reshape((-1,)), time_of_event_test.reshape((-1,)), 'r.', label='test')
    ax2.plot(input_values_test.reshape((-1,)), eol_data.reshape((-1,)), 'kx', label='pred')
    ax1.set_xlabel('input current, A', fontsize=14)
    ax1.set_ylabel('time to end of discharge, s', fontsize=14)
    ax2.set_xlabel('input current, A', fontsize=14)
    ax1.legend(fontsize=14, fancybox=True, shadow=True)
    ax2.legend(fontsize=14, fancybox=True, shadow=True)
    plt.show()
