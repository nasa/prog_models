# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections.abc import Iterable
from numbers import Number
import numpy as np
from tensorflow import keras
from prog_models.data_models.lstm_model import LSTMStateTransitionModel
from tensorflow.keras import layers

from . import DataModel


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
        if len(inputs) == 0:
            raise ValueError("No inputs provided. inputs must be in format [run1_inputs, ...] and have at least one element")
        if len(inputs) != len(event_times):
            raise ValueError("Event Times must be same length as inputs")
        if not isinstance(event_times, Iterable):
            raise ValueError(f"event_times must be in format [run1_eventtimes, ...], got {type(event_times)}")
        if not isinstance(params['normalize'], bool):
            raise TypeError(f"normalize must be a boolean, not {type(params['normalize'])}")

        # Prepare datasets
        (u_all, z_all) = LSTMStateTransitionModel.pre_process_data(inputs, outputs, **params)

        # Normalize
        if params['normalize']:
            n_inputs = len(inputs[0][0])
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
