# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np

from . import PrognosticsModel


class LSTMStateTransitionModel(PrognosticsModel):
    """
    A State Transition Model with no events using an Keras LSTM Model.

    Args:
        model (keras.Model): Keras model to use for state transition

    Keyword Args:
        inputs (list[str]): List of input keys

    TODO: Examples
    """
    default_params = {
        'delay': 1,
    }

    def __init__(self, model, **kwargs):
        model_shape = model.input.shape
        inputs = kwargs.get('inputs', [f'u{i}' for i in range(model_shape[2])])
        kwargs['sequence_length'] = model.input.shape[1]
        states = [f'{input_i}_{j}' for j in range(model_shape[1]+kwargs.get('delay', self.default_params['delay'])) for input_i in inputs]
        outputs = kwargs.get('outputs', [f'z{i}' for i in range(model.output.shape[1])])

        super().__init__(**kwargs)

        self.model = model

    def initialize(self, u=None, z=None):
        return self.StateContainer([None for _ in self.states])

    def next_state(self, x, u, dt):
        # Rotate new input into state
        input_data = u.matrix
        states = x.matrix[len(input_data):]
        return self.StateContainer(np.vstack(states, input_data))

    def output(self, x):
        if x[0] is None:
            raise Exception(f"Output estimation is not available until at least {self.parameters['delay']+self.parameters['sequence_length']} timesteps have passed.")
        m_input = x.matrix[:self.parameters['sequence_length']*len(self.inputs)].reshape(self.parameters['sequence_length'], len(self.inputs))
        m_output = self.model(m_input)
        return self.OutputContainer(m_output)

    @staticmethod
    def from_data(data, **kwargs):
        """
        Generate a LSTMStateTransitionModel from data

        Args:
            data (Tuple[Array, Array]): 

        Keyword Args:
            delay (int): Number of timesteps to delay the input
            sequence_length (int): Length of the input sequence
            inputs (List[str]): List of input keys
            outputs (List[str]): List of outputs keys
            shuffle (bool): If the data is shuffled in data preparation
            validation_percentage (float): Percentage of data to use for validation, between 0-1
            lstm_layers (int): Number of LSTM layers in the model
            epochs (int): Number of epochs to use in training

        Returns:
            LSTMStateTransitionModel: Generated Model
        """
        params = { # default_params
            'sequence_length': 128,
            'validation_percentage': 0.25,
            'shuffle': True,
            'lstm_layers': 1,
            'epochs': 2
        }
        params.update(LSTMStateTransitionModel.default_params)
        params.update(kwargs)

        # TODO(CT): Check parameters

        from tensorflow import keras
        from tensorflow.keras import layers

        # Prepare datasets
        u_all = []
        z_all = []
        SEQUENCE_LENGTH = params['sequence_length']
        DELAY = params['delay']

        for (u, z) in data:
            datem = keras.utils.timeseries_dataset_from_array(
                    np.array([u[:-DELAY], z[:-DELAY]]).T,
                    targets = z[DELAY:],
                    sequence_length = SEQUENCE_LENGTH,
                    shuffle = params['shuffle']
                )
            for u_i, z_i in datem:
                if len(u_i) == SEQUENCE_LENGTH:
                    u_all.extend(u_i)
                    z_all.extend(z_i)

        u_all = np.array(u_all)
        z_all = np.array(z_all)

        # Build model
        VALIDATION_PERCENT = params['validation_percentage']
        TRAINING_SIZE = int(len(u_all)*(1-VALIDATION_PERCENT))

        callbacks = [
            keras.callbacks.ModelCheckpoint("jena_sense.keras", save_best_only=True)
        ]

        training_data = (u_all[:TRAINING_SIZE], z_all[:TRAINING_SIZE])
        validation_data = (u_all[TRAINING_SIZE:], z_all[TRAINING_SIZE:])

        inputs = keras.Input(shape=training_data[0].shape[1:])
        x = inputs
        for i in range(params['lstm_layers']):
            x = layers.LSTM(16)(x)
        x = layers.Dense(1)(x)
        model = keras.Model(inputs, x)
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        model.fit(training_data[0], training_data[1], epochs=params['epochs'], callbacks = callbacks, validation_data = validation_data)
        return LSTMStateTransitionModel(keras.models.load_model("jena_sense.keras"), **params)
        
    @staticmethod
    def from_model(model, **kwargs):
        config = { # default_params
            'include_time': True, # Include time as a config parameter
        }
        pass
