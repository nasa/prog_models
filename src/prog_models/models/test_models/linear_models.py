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


class TwoInputNoOutputOneEventLM(LinearModel):
    """
    Simple model that increases state by u1+0.5*u2 every step. Event occurs when state == 10
    """
    inputs = ['u1', 'u2']
    states = ['x1']
    events = ['x1 == 10']

    A = np.array([[0]])
    B = np.array([[1, 0.5]])
    C = np.empty((0,1))
    F = np.array([[-0.1]])
    G = np.array([[1]])

    default_parameters = {
        'process_noise': 0,
        'x0': {
            'x1': 0
        }
    }


class TwoInputNoOutputTwoEventLM(LinearModel):
    """
    Simple model that increases state by u1+0.5*u2 every step. Event occurs when state == 10
    """
    inputs = ['u1', 'u2']
    states = ['x1']
    events = ['x1 == 10', 'x1 == 5']

    A = np.array([[0]])
    B = np.array([[1, 0.5]])
    C = np.empty((0,1))
    F = np.array([[-0.1], [-0.2]])
    G = np.array([[1], [1]])

    default_parameters = {
        'process_noise': 0,
        'x0': {
            'x1': 0
        }
    }
