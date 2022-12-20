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
        models (List[chaospy.Poly]):
            Polynomial chaos expansion models (one for each event)
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
    def __init__(self, models, times, input_keys, **kwargs):
        self.inputs = input_keys
        self.states = []
        self.outputs = []
        self.events = kwargs.get('event_keys', [f'e{i}' for i in range(len(models))])

        super().__init__(**kwargs)
        self.parameters['models'] = models
        self.parameters['times'] = times

    def time_of_event(self, x, future_loading_eqn, **kwargs) -> dict:
        loading = np.array([future_loading_eqn(t, x).matrix for t in self.parameters['times']]).T[0].T
        return {key: model(*loading)[0] for key, model in zip(self.events, self.parameters['models'])}

    @classmethod
    def from_data(cls, times, inputs, time_of_event, input_keys, **kwargs):
        """
        Create a PolynomialChaosExpansion from data.

        Args:
            times (list[float]):
                list of times data for use in data. Each element is the time such that inputs[i] is the inputs at time[i]
            inputs (list[np.array]): 
                list of :term:`input` data for use in data. Each  eelement is the inputs for a single run of size (n_times, n_inputs)
            time_of_event (np.array):
                Array of time of event data for use in data. Each element is the time of event for a single run of size (n_times, n_events)
            input_keys (list[str]):
                List of input keys for the inputs
                # TODO(CT): MAKE OPTIONAL ----------------

        Keyword Args:
            J (chaospy.Distribution, optional):
                Joint distribution to sample from. If not included, input_dists must be provided
            input_dists (list[chaospy.Distribution], optional):
                List of chaospy distributions for each input
            order (int, optional):
                Order of the polynomial chaos expansion
        """
        default_params = {
            'J': None,
            'input_dists': None,
            'order': 2,
        }
        params = default_params.copy()
        params.update(kwargs)
        
        if len(input_keys) == 0:
            raise ValueError('Must provide at least one input key, was empty')
        if params['J'] is None and params['input_dists'] is None:
            raise ValueError('Either J or input_dists must be provided')
        if params['J'] is None:
            params['J'] = cp.J(*params['input_dists'])
        if params['order'] < 1:
            raise ValueError(f'order must be greater than 0, was {params["order"]}')
        if len(time_of_event) == 0:
            raise ValueError('Training data must include at least one event')
        if len(times) == 0:
            raise ValueError('Times must include at least one time')
        if len(inputs) == 0:
            raise ValueError('Inputs must include at least one input')

        n_events = len(time_of_event[0])
        if n_events == 0:
            raise ValueError('There must be at least one event to train an PCE model')

        if len(time_of_event) != inputs.shape[1]:
            raise ValueError('There must be the same number of samples for inputs and time of event')

        # Train
        expansion = cp.generate_expansion(order=params['order'], dist=params['J']) # Order=2 is the only hyperparameter
        surrogates = [
            cp.fit_regression(expansion, inputs, toe_i) for toe_i in time_of_event.T
        ]

        return cls(surrogates, times = times, input_keys = input_keys, **params)

    @classmethod
    def from_model(cls, m, x, input_dists, **kwargs):
        """
        Create a PolynomialChaosExpansion from a model.

        Args:
            m (Model):
                Model to create PolynomialChaosExpansion from
            x (StateContainer):
                Initial state to use for simulation
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
            'max_time': None,
            'times': None
        }
        params = default_params.copy()
        params.update(kwargs)

        if params['N'] < 1:
            raise ValueError(f'N must be greater than 0, was {params["N"]}. At least one sample required')
        if params['discretization'] < 2:
            raise ValueError(f'discretization must be greater than 1, was {params["discretization"]}')
            # TODO(CT): HANDLE DISCRETIZATION OF 1 (i.e., const load)
        if params['max_time'] is None and params['times'] is None:
            raise ValueError('Either max_time or times must be provided')
        if len(m.events) < 1:
            raise ValueError('Model must have at least one event')
        if len(m.inputs) < 1:
            raise ValueError('Model must have at least one input')

        # Setup data
        if params['times'] is None:
            params['times'] = np.linspace(0, params['max_time'], params['discretization'])
        input_dists = [cp.UserDistribution(
                            cdf = input_dists[key].cdf,
                            pdf = input_dists[key].pdf,
                            ppf = input_dists[key].ppf
                        )
                        for key in m.inputs 
                        for _ in range(params['discretization'])
                        ]
        J = cp.J(*input_dists)  # Joint distribution to sample from
        
        # Simulate to collect time_of_event data
        time_of_event = np.empty((params['N'], len(m.events)), dtype=np.float64)
        def future_loading(t, x=None):
            nonlocal interpolator
            return m.InputContainer(interpolator(t)[np.newaxis].T)
        
        all_samples = J.sample(size=params['N'], rule='latin_hypercube')
        for i in range(params['N']):
            # Sample
            interpolator = sp.interpolate.interp1d(params['times'], all_samples[:, i], bounds_error = False, fill_value = all_samples[:, i][-1])
            # TODO(CT): RAISE DESCRIPTIVE ERROR IF TIMES IS NOT ENOUGH

            # Simulate to get data
            time_of_event_i = m.time_of_event(x, future_loading, dt=params['dt'])

            # Add to list
            time_of_event[i] = [time_of_event_i[key] for key in m.events]
        
        params['input_keys'] = m.inputs
        params['x'] = x
        return cls.from_data(inputs = all_samples, time_of_event = time_of_event, event_keys = m.events, J=J, **params)

PCE = PolynomialChaosExpansion
