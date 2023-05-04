# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example using the estimate_params edge case features

.. dropdown:: More details

    estimate_params is a useful way of re-evaluating current parameters values by estimating the model parameters given data. 

    The feature will override model parameters by minimizing the error between simulated data and the given data with the set of initial parameters.

    Furthermore, estimate_params takes in a list of 'keys', or parameters that the user intends to selectively change. Additionally, user has the freedom to utilize any of scipy's minimize options i.e. tolerance, error-method, and bounds for the parameters.

    There are a few limitations that are seen specifically when passing a subset of parameters to keys and when using the bounds feature, even on fairly simple models. 

"""

from matplotlib import pyplot as plt
from prog_models import *
from prog_models.models import *

def run_example():
    m = ThrownObject()
    gt = ThrownObject()

    times = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    inputs = [{}]*9
    outputs = [
        {'x': 1.83},
        {'x': 36.95},
        {'x': 62.36},
        {'x': 77.81},
        {'x': 83.45},
        {'x': 79.28},
        {'x': 65.3},
        {'x': 41.51},
        {'x': 7.91},
    ]

    # Now, set some incorrect parameters
    m.parameters['thrower_height'] = 1.5
    m.parameters['throwing_speed'] = 25
    keys = ['thrower_height', 'throwing_speed']
    
    for key in keys:
        print("-", key, m.parameters[key])
    print(' Error: ', m.calc_error(times, inputs, outputs, dt=1e-4))

    # Now, after calling estimate_params, we will notice the error decrease significantly

    m.estimate_params(times = times, inputs = inputs, outputs = outputs, keys = keys)

    for key in keys:
        print("-", key, m.parameters[key])
    print(' Error: ', m.calc_error(times, inputs, outputs, dt=1e-4))

    # Now for some edge cases!

    # Notice how the initial parameter is set to a number outside of the bounds.
    m.parameters['thrower_height'] = 5
    m.parameters['throwing_speed'] = 25 
    m.estimate_params(times = times, inputs = inputs, outputs = outputs, keys = keys, bounds=((1, 4), (20, 42)))
    # Turns out that because of how scipy runs its minimize function, our thrower_height will result in converging to 4.
    for key in keys:
        print("-", key, m.parameters[key])
    print(' Error: ', m.calc_error(times, inputs, outputs, dt=1e-4))



def params_est(m, times, inputs, outputs, keys, bounds):

    m.estimate_params(times = times, inputs = inputs, outputs = outputs, keys = keys, bounds = bounds)
    
if __name__=='__main__':
    run_example()
