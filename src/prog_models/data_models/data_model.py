# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from abc import ABC, abstractclassmethod
import numpy as np

from .. import PrognosticsModel
import sys

class DataModel(PrognosticsModel, ABC):
    """
    .. versionadded:: 1.4.0

    Abstract Base Class for all Data Models (e.g., :py:class:`LSTMStateTransitionModel`). Defines the interface and all common tools. To create a new Data-Driven model, first subclass this, then define the abstract methods from this class and :py:class:`prog_models.PrognosticsModel`. 

    See Also:
        PrognosticsModel
    """

    @abstractclassmethod
    def from_data(cls, **kwargs) -> "DataModel":
        """
        Create a Data Model from data. This class is overwritten by specific data-driven classes (e.g., :py:class:`LSTMStateTransitionModel`)

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
    
    def summary(self, file = sys.stdout):
        """
        Print a summary of the model
        """
        print(self.__class__.__name__, file=file)
        
    @classmethod
    def from_model(cls, m: PrognosticsModel, load_functions: list, **kwargs) -> "DataModel":
        """
        Create a Data Model from an existing PrognosticsModel (i.e., a surrogate model). Generates data through simulation with supplied load functions. Then calls :py:func:`from_data` to generate the model.

        Args:
            m (PrognosticsModel): 
                Model to generate data from
            load_functions (list[function]): 
                Each index is a callable loading function of (t, x = None) -> z used to predict :term:`future load` at a given time (t) and :term:`state` (x)

        Keyword Args:
            add_dt (bool): If the timestep should be added as an input

        Addditional configuration parameters from :py:func:`prog_models.PrognosticsModel.simulate_to_threshold`. These can be an array (of same length as load_functions) of config for each individual sim, or one value to apply to all
        Additional configuration parameters from `from_data`

        Returns:
            DataModel: Trained PrognosticsModel
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
        outputs = [d.outputs for d in data]
        states = [d.states for d in data]
        event_states = [d.event_states for d in data]

        return cls.from_data(times = times, inputs = inputs, states = states, outputs = outputs, event_states = event_states, **config)
