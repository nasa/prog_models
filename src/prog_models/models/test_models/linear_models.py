# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

import numpy as np

from prog_models import LinearModel

class OneInputNoOutputOneEventLM(LinearModel):
    """
    Simple model that increases state by u1 every step. Event occurs when state == 10
    """
    inputs = ['u1']
    states = ['x1']
    events = ['x1 == 10']

    A = np.array([[0]])
    B = np.array([[1]])
    C = np.empty((0,1))
    F = np.array([[-0.1]])
    G = np.array([[1]])

    default_parameters = {
        'process_noise': 0,
        'x0': {
            'x1': 0
        }
    }

class OneInputNoOutputTwoEventLM(LinearModel):
    """
    Simple model that increases state by u1 every step. Event occurs when state == 10, 5
    """
    inputs = ['u1']
    states = ['x1']
    events = ['x1 == 10', 'x1 == 5']

    A = np.array([[0]])
    B = np.array([[1]])
    C = np.empty((0,1))
    F = np.array([[-0.1], [-0.2]])
    G = np.array([[1], [1]])

    default_parameters = {
        'process_noise': 0,
        'x0': {
            'x1': 0
        }
    }

m = OneInputNoOutputOneEventLM()
m2 = OneInputNoOutputTwoEventLM()

print('m')
m.simulate_to_threshold(lambda t, x=None: m.InputContainer({'u1': 1}), save_freq = 1, print=True)

print('m2')
m2.simulate_to_threshold(lambda t, x=None: m.InputContainer({'u1': 1}), save_freq = 1, print=True)
