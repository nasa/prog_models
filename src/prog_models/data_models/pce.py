# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

import chaospy as cp
import numpy as np
import scipy as sp

from prog_models.data_models import DataModel
from prog_models.utils.containers import DictLikeMatrixWrapper


class PolynomialChaosExpansion(DataModel):
    """
    .. versionadded:: 1.5.0

    Polynomial Chaos Expansion direct data model. Uses chaospy to generate a polynomial chaos expansion from data or a model to learn the relationship between future inputs and time of event.

    Generally this is used as a :term:`surrogate model` for a :term:`model` that is too expensive to simulate. This is done using the :meth:`PolynomialChaosExpansion.from_model` method. The model is used to generate data, which is then used to train the polynomial chaos expansion. The polynomial chaos expansion can then be used to predict the time of event for future inputs.

    Args:
        model (chaospy.Poly):
            Polynomial chaos expansion model
        times (list[float]):
            List of times to use for the polynomial chaos expansion
        input_keys (list[str]):
            List of input keys for the inputs

    Keyword Args:
        event_keys (list[str], optional):
            List of event keys for the events. If not provided, will be generated as e0, e1, e2, etc.

    See Also:
        :class:`prog_models.data_models.DataModel`
        PolynomialChaosExpansion.from_data
        PolynomialChaosExpansion.from_model

    Note:
        The generated model is only valid for the intial state at which the data was generated. If the initial state is different, the model will not be valid.
    """
    def __init__(self, model, times, input_keys, **kwargs):
        self.inputs = input_keys
        self.states = []
        self.outputs = []
        self.events = kwargs.get('event_keys', [f'e{i}' for i in range(model.size)])

        super().__init__(**kwargs)
        self.parameters['model'] = model
        self.parameters['times'] = times

    def time_of_event(self, x, future_loading_eqn, **kwargs) -> dict:
        loading = np.array([future_loading_eqn(t, x).matrix for t in self.parameters['times']])
        return {key: value for key, value in zip(self.events, self.parameters['model'](*loading.T[0].T))}

    @classmethod
    def from_data(cls, times, inputs, time_of_event, input_keys, **kwargs):
        """
        Create a PolynomialChaosExpansion from data.

        Args:
            times (list[float]):
                list of times data for use in data. Each element is the time such that inputs[i] is the inputs at time[i]
            inputs (list[np.array]): 
                list of :term:`input` data for use in data. Each element is the inputs for a single run of size (n_times, n_inputs)
            time_of_event (np.array):
                Array of time of event data for use in data. Each element is the time of event for a single run of size (n_times, n_events)
            input_keys (list[str]):
                List of input keys for the inputs

        Keyword Args:
            J (chaospy.Distribution, optional):
                Joint distribution to sample from. If not included, input_dists must be provided
            input_dists (list[chaospy.Distribution], optional):
                List of chaospy distributions for each input
        """
        default_params = {
            'J': None,
            'input_dists': None,
        }
        params = default_params.copy()
        params.update(kwargs)
        
        if len(input_keys) == 0:
            raise ValueError('Must provide at least one input key, was empty')
        if params['J'] is None and params['input_dists'] is None:
            raise ValueError('Either J or input_dists must be provided')
        if params['J'] is None:
            if isinstance(params['input_dists'][0], dict) or isinstance(params['input_dists'][0], DictLikeMatrixWrapper):
                # Convert to list
                input_dists = [input_dists[key] for key in m.inputs]
            params['J'] = cp.J(*params['input_dists'])

        # Train
        expansion = cp.generate_expansion(order=2, dist=params['J']) # Order=2 is the only hyperparameter
        surrogate = cp.fit_regression(expansion, inputs.T, time_of_event.T[0])

        return cls(surrogate, times = times, input_keys = input_keys, **params)

    @classmethod
    def from_model(cls, m, input_dists, **kwargs):
        """
        Create a PolynomialChaosExpansion from a model.

        Args:
            m (Model):
                Model to create PolynomialChaosExpansion from
            input_dists (dict[key, chaospy.Distribution]):"
                List of chaospy distributions for each input            

        Keyword Args:
            discretization (int, optional):
                Number of points to discretize each input
            N (int, optional):
                Number of samples to use for training
            dt (float, optional):
                Time step to use for simulation
            order (int, optional):
                Order of the polynomial chaos expansion
            max_time (float, optional):
                Maximum time to simulate to. Either max_time or times must be provided
            times (list[float], optional):
                List of times to simulate to. If provided, max_time is ignored
        """
        default_params = {
            'discretization': 5,
            'N': 1000, 
            'dt': 0.1, 
            'order': 2,
            'max_time': None,
            'times': None
        }
        params = default_params.copy()
        params.update(kwargs)

        if params['N'] < 1:
            raise ValueError(f'N must be greater than 0, was {params["N"]}. At least one sample required')
        if params['discretization'] < 1:
            raise ValueError(f'discretization must be greater than 0, was {params["discretization"]}')
        if params['order'] < 1:
            raise ValueError(f'order must be greater than 0, was {params["order"]}')
        if params['max_time'] is None and params['times'] is None:
            raise ValueError('Either max_time or times must be provided')

        # Setup data
        if params['times'] is None:
            params['times'] = np.linspace(0, params['max_time'], params['discretization'])
        input_dists = [input_dists[key] for key in m.inputs]
        J = cp.J(*input_dists)  # Joint distribution to sample from
        
        # Simulate to collect time_of_event data
        time_of_event = []
        inputs = []
        def future_loading(t, x=None):
            nonlocal interpolator
            return m.InputContainer(interpolator(t)[np.newaxis].T)
        
        for i in range(params['N']):
            # Sample
            samples = J.sample(size=params['discretization'], rule='latin_hypercube')
            interpolator = sp.interpolate.interp1d(params['times'], samples)

            # Simulate to get data
            time_of_event_i = m.time_of_event(m.initialize(), future_loading, dt=params['dt'])

            # Add to list
            time_of_event.append(np.array([time_of_event_i[key] for key in m.events]))
            inputs.append(samples)
        
        params['input_keys'] = m.inputs
        return cls.from_data(inputs = np.array(inputs), time_of_event = np.array(time_of_event), event_keys = m.events, J=J, **params)

PCE = PolynomialChaosExpansion
