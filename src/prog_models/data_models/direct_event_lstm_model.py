# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections.abc import Iterable
from numbers import Number
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from . import DataModel
from ..sim_result import SimResult


class LSTMDirectModel(DataModel):
    """
    """
    def __init__(self, model, **kwargs):
        # Setup inputs, outputs, states 
        self.outputs = kwargs.get('output_keys', [f'z{i}' for i in range(model.output.shape[1])])

        input_shape = model.input.shape
        input_keys = kwargs.get('input_keys', [f'u{i}' for i in range(input_shape[2]-len(self.outputs))])
        self.inputs = input_keys.copy()
        # Outputs from the last step are part of input
        self.inputs.extend([f'{z_key}_t-1' for z_key in self.outputs])

        # States are in format [u_t-n+1, z_t-n, ..., u_t, z_t-1]
        self.states = []
        for j in range(input_shape[1]-1, -1, -1):
            self.states.extend([f'{input_i}_t-{j}' for input_i in input_keys])
            self.states.extend([f'{output_i}_t-{j+1}' for output_i in self.outputs])

        kwargs['window'] = input_shape[1]
        kwargs['model'] = model  # Putting it in the parameters dictionary simplifies pickling

        super().__init__(**kwargs)

        # Save Model
        self.model = model

    def initialize(self, u=None, z=None):
        """
        Initialize the model with the given inputs and outputs. For LSTMStateTransitionModel, the initial state is set to None. As data is added (i.e., next_state is called), the initial state is updated.

        This is one of the rare models that cannot be initialized right away.

        Args:
            u (InputContainer, optional): first input
            z (OutputContainer, optional): first output

        Returns:
            StateContainer: First State
        """
        return self.StateContainer(np.array([[None] for _ in self.states]))
    
    # TODO(CT): Output

    @staticmethod
    def pre_process_data(data, window, **kwargs):
        """
        Pre-process data for the LSTMStateTransitionModel. This is run inside from_data to convert the data into the desired format 

        Args:
            data (List[Tuple] or Tuple (where Tuple is equivilant to [Tuple]))): Data to be processed. each element is of format (input, output), where input and output can be ndarray or SimulationResult
            window (int): Length of a single sequence

        Returns:
            Tuple[ndarray, ndarray]: pre-processed data (input, output). Where input is of size (num_sequences, window, num_inputs) and output is of size (num_sequences, num_outputs)
        """
        # Data is a List[Tuple] or Tuple (where Tuple is equivilant to [Tuple]))
        # Tuple is (input, output)

        u_all = []
        z_all = []
        for (u, z) in data:
            # Each item (u, z) is a 1-d array, a 2-d array, or a SimResult

            # Process Input
            if isinstance(u, SimResult):
                if len(u[0].keys()) == 0:
                    # No inputs
                    u = []
                else:
                    u = np.array([u_i.matrix[:,0] for u_i in u])

            if isinstance(u, (list, np.ndarray)):
                if len(u) == 0:
                    # No inputs
                    u_i = []
                elif np.isscalar(u[0]):
                    # Input is 1-d array (i.e., 1 input)
                    # Note: 1 is added to account for current time (current input used to predict output at time i)
                    u_i = [[[u[i+j]] for j in range(1, window+1)] for i in range(len(u)-window-1)]
                elif isinstance(u[0], (list, np.ndarray)):
                    # Input is d-d array
                    # Note: 1 is added to account for current time (current input used to predict output at time i)
                    n_inputs = len(u[0])
                    u_i = [[[u[i+j][k] for k in range(n_inputs)] for j in range(1,window+1)] for i in range(len(u)-window-1)]
                else:
                    raise TypeError(f"Unsupported input type: {type(u)} for internal element (data[0][i]")  
            else:
                raise TypeError(f"Unsupported data type: {type(u)}. input u must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")


            # Process Output
            if isinstance(z, SimResult):
                if len(z[0].keys()) == 0:
                    # No outputs
                    z = []
                else:
                    z = np.array([z_i.matrix[:,0] for z_i in z])

            if isinstance(z, (list, np.ndarray)):
                if len(z) != len(u) and len(u) != 0 and len(z) != 0:
                    # Checked here to avoid SimResults from accidentially triggering this check
                    raise IndexError(f"Number of outputs ({len(z)}) does not match number of inputs ({len(u)})")

                if len(z) == 0:
                    # No outputs
                    z_i = []
                elif np.isscalar(z[0]):
                    # Output is 1-d array (i.e., 1 output)
                    z_i = [[z[i]] for i in range(window+1, len(z))]
                elif isinstance(z[0], (list, np.ndarray)):
                    # Input is d-d array
                    n_outputs = len(z[0])
                    z_i = [[z[i][k] for k in range(n_outputs)] for i in range(window+1, len(z))]
                else:
                    raise TypeError(f"Unsupported input type: {type(z)} for internal element (data[0][i]")  

                # Also add to input (past outputs are part of input)
                if len(u_i) == 0:
                    u_i = [[z_ii for _ in range(window)] for z_ii in z_i]
                else:
                    for i in range(len(z_i)):
                        for j in range(window):
                            u_i[i][j].extend(z_i[i])
            else:
                raise TypeError(f"Unsupported data type: {type(u)}. input u must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")
            
            u_all.extend(u_i)
            z_all.extend(z_i)
        
        u_all = np.array(u_all)
        z_all = np.array(z_all)
        return (u_all, z_all)

    @classmethod
    def from_data(cls, inputs, outputs, event_times, **kwargs):
        params = { # default_params
            'window': 128,
            'validation_split': 0.25,
            'epochs': 2,
            'prediction_steps': 1,
            'layers': 1,
            'units': 16,
            'activation': 'tanh',
            'dropout': 0.1,
            'horizon': 5000,
            'n_future_load_steps': 50,
            'normalize': True
        }.copy()  # Copy is needed to avoid updating default
        
        params.update(LSTMDirectModel.default_params)
        params.update(kwargs)

        # Input Validation
        if not np.isscalar(params['window']):
            raise TypeError(f"window must be an integer greater than 0, not {type(params['window'])}")
        if params['window'] <= 0:
            raise ValueError(f"window must be greater than 0, got {params['window']}")
        if not np.isscalar(params['layers']):
            raise TypeError(f"layers must be an integer greater than 0, not {type(params['layers'])}")
        if params['layers'] <= 0:
            raise ValueError(f"layers must be greater than 0, got {params['layers']}")
        if np.isscalar(params['units']):
            params['units'] = [params['units'] for _ in range(params['layers'])]
        if not isinstance(params['units'], (list, np.ndarray)):
            raise TypeError(f"units must be a list of integers, not {type(params['units'])}")
        if len(params['units']) != params['layers']:
            raise ValueError(f"units must be a list of integers of length {params['layers']}, got {params['units']}")
        for i in range(params['layers']):
            if params['units'][i] <= 0:
                raise ValueError(f"units[{i}] must be greater than 0, got {params['units'][i]}")
        if not isinstance(params['dropout'], Number):
            raise TypeError(f"dropout must be an float greater than or equal to 0, not {type(params['dropout'])}")
        if params['dropout'] < 0:
            raise ValueError(f"dropout must be greater than or equal to 0, got {params['dropout']}")
        if not isinstance(params['activation'], (list, np.ndarray)):
            params['activation'] = [params['activation'] for _ in range(params['layers'])]
        if not np.isscalar(params['validation_split']):
            raise TypeError(f"validation_split must be an float between 0 and 1, not {type(params['validation_split'])}")
        if params['validation_split'] < 0 or params['validation_split'] >= 1:
            raise ValueError(f"validation_split must be between 0 and 1, got {params['validation_split']}")
        if not np.isscalar(params['epochs']):
            raise TypeError(f"epochs must be an integer greater than 0, not {type(params['epochs'])}")
        if params['epochs'] < 1:
            raise ValueError(f"epochs must be greater than 0, got {params['epochs']}")
        if np.isscalar(inputs):  # Is scalar (e.g., SimResult)
            inputs = [inputs]
        if np.isscalar(outputs):
            outputs = [outputs]
        if len(inputs) != len(outputs):
            raise ValueError("Inputs must be same length as outputs")
        if not isinstance(inputs, Iterable):
            raise ValueError(f"inputs must be in format [run1_inputs, ...], got {type(inputs)}")
        if len(inputs) == 0:
            raise ValueError("No inputs provided. inputs must be in format [run1_inputs, ...] and have at least one element")
        if not isinstance(outputs, Iterable):
            raise ValueError(f"outputs must be in format [run1_outputs, ...], got {type(outputs)}")
        if len(inputs) != len(event_times):
            raise ValueError("Event Times must be same length as inputs")
        if not isinstance(event_times, Iterable):
            raise ValueError(f"event_times must be in format [run1_eventtimes, ...], got {type(event_times)}")
        if not isinstance(params['normalize'], bool):
            raise TypeError(f"normalize must be a boolean, not {type(params['normalize'])}")

        # Convert to previous format - used below
        data = [(u, z) for u, z in zip(inputs, outputs)]

        # Prepare datasets
        (u_all, z_all) = LSTMDirectModel.pre_process_data(data, **params)

        # Normalize
        if params['normalize']:
            n_inputs = len(data[0][0][0])
            u_mean = np.mean(u_all[:,0,:n_inputs], axis=0)
            u_std = np.std(u_all[:,0,:n_inputs], axis=0)
            # If there's no variation- dont normalize 
            u_std[u_std == 0] = 1
            z_mean = np.mean(z_all, axis=0)
            z_std = np.std(z_all, axis=0)
            # If there's no variation- dont normalize 
            z_std[z_std == 0] = 1

            # Add output (since z_t-1 is last input)
            u_mean = np.hstack((u_mean, z_mean))
            u_std = np.hstack((u_std, z_std))

        # Build model
        callbacks = [
            keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
        ]

        inputs = keras.Input(shape=u_all.shape[1:])
        x = inputs
        if params['normalize']:
            x = layers.Normalization(mean = u_mean, variance = u_std**2)(inputs)
        for i in range(params['layers']):
            if i == params['layers'] - 1:
                # Last layer
                x = layers.LSTM(params['units'][i], activation=params['activation'][i])(x)
            else:
                # Intermediate layer
                x = layers.LSTM(params['units'][i], activation=params['activation'][i], return_sequences=True)(x)
        
        if params['dropout'] > 0:
            # Dropout prevents overfitting
            x = layers.Dropout(params['dropout'])(x)

        x = layers.Dense(z_all.shape[1] if z_all.ndim == 2 else 1)(x)
        model = keras.Model(inputs, x)
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

        # Train model
        model.fit(u_all, z_all, epochs=params['epochs'], callbacks = callbacks, validation_split = params['validation_split'])

        return cls(keras.models.load_model("best_model.keras"), **params)
