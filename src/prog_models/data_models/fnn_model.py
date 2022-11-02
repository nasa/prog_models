# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
from tensorflow.keras import layers

from . import NNStateTransitionModel


class FNNStateTransitionModel(NNStateTransitionModel):
    """
    .. versionadded:: 1.4.0

    A State Transition Model with no :term:`event` using an Keras CNN Lyer.
    State transition models map from the :term:`input` at time t and :term:`output` at time t-1 plus historical data from a set window to the :term:`output` at time t.

    Most users will use the :py:func:`CNNStateTransitionModel.from_data` method to create a model, but the model can be created by passing in a model directly into the constructor. The CNN model in this method maps from [u_t-n+1, z_t-n, ..., u_t, z_t-1] to z_t. Past :term:`input` are stored in the :term:`model` internal :term:`state`. Actual calculation of :term:`output` is performed when :py:func:`CNNStateTransitionModel.output` is called. When using in simulation that may not be until the simulation results are accessed.

    Args:
        output_model (keras.Model): If a state model is present, maps from the state_model outputs to model :term:`output`. Otherwise, maps from model inputs to model :term:`output`
        state_model (keras.Model, optional): Keras model to use for state transition
        event_state_model (keras.Model, optional): If a state model is present, maps from the state_model outputs to :term:`event state`. Otherwise, maps from model inputs to :term:`event state`
        t_met_model (keras.Model, optional): If a state model is present, maps from the state_model outputs to if the threshold has been met. Otherwise, maps from model inputs to if the threshold has not been met

    Keyword Args:
        input_keys (list[str]): List of input keys
        output_keys (list[str]): List of output keys
        event_keys (list[str]): List of event keys

    Attributes:
        model (keras.Model): Keras model to use for state transition

    See Also:
        CNNStateTransitionModel.from_data
        examples.lstm_model
    """

    default_params = {
        'process_noise': 0,  # Default 0 noise
        'measurement_noise': 0,  # Default 0 noise
    }

    @classmethod
    def from_data(cls, inputs, outputs, event_states = None, t_met = None, **kwargs):
        """
        Generate a CNNStateTransitionModel from data

        Args:
            inputs (list[np.array]): 
                list of :term:`input` data for use in data. Each element is the inputs for a single run of size (n_times, n_inputs)
            outputs (list[np.array]): 
                list of :term:`output` data for use in data. Each element is the outputs for a single run of size (n_times, n_outputs)
            event_states (list[np.array], optional): 
                list of :term:`event state` data for use in data. Each element is the event state for a single run of size (n_times, n_events)
            t_met (list[np.array], optional): 
                list of :term:`threshold` met data for use in data. Each element is if the threshold has been met for a single run of size (n_times, n_events) 

        Keyword Args:
            window (int): 
                Number of historical points used in the model. I.e, if window is 3, the model will map from [t-3, t-2, t-1] to t
            input_keys (list[str]): 
                List of keys to use to identify :term:`input`. If not supplied u[#] will be used to idenfiy inputs
            output_keys (list[str]): 
                List of keys to use to identify :term:`output`. If not supplied z[#] will be used to idenfiy outputs
            event_keys (list[str]):
                List of keys to use to identify events for :term:`event state` and :term:`threshold` met. If not supplied event[#] will be used to identify events
            validation_percentage (float): 
                Percentage of data to use for validation, between 0-1
            epochs (int): 
                Number of epochs (i.e., iterations) to train the model. More epochs means better results (to a point), but more time to train. Note: large numbers of epochs may result in overfitting.
            layers (int): 
                Number of CNN layers to use. More layers can represent more complex systems, but are less efficient. Default: 3
            units (int or list[int]): 
                number of units (i.e., dimensionality of output state) used in each cnn layer. Using a scalar value will use the same number of units for each layer.
            activation (str or list[str]): 
                Activation function to use for each layer
            dropout (float): 
                Dropout rate to be applied. Dropout helps avoid overfitting
            normalize (bool): 
                If the data should be normalized. This is recommended for most cases.
            early_stopping (bool):
                If early stopping is desired. Default is True
            early_stop.cfg (dict):
                Configuration to pass into early stopping callback (if enabled). See keras documentation (https://keras.io/api/callbacks/early_stopping) for options. E.g., {'patience': 5}
            workers (int):
                Number of workers to use when training. One worker indicates no multiprocessing

        Returns:
            CNNStateTransitionModel: Generated Model

        See Also:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/CNN
        """
        params = { # default_params
            'layers': 3,
            'units': 128,
            'activation': 'tanh',
        }.copy()  # Copy is needed to avoid updating default
        params.update(FNNStateTransitionModel.default_params)
        params.update(kwargs)

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
        if not isinstance(params['activation'], (list, np.ndarray)):
            params['activation'] = [params['activation'] for _ in range(params['layers'])]

        internal_layers = [layers.Flatten()]
        internal_layers.extend([layers.Dense(params['units'][i], activation=params['activation'][i]) for _ in range(params['layers'])])
        return NNStateTransitionModel.from_data(internal_layers, inputs, outputs, event_states, t_met, **params)
