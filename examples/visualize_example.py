# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
Visualization Module Example
"""

import matplotlib.pyplot as plt
from prog_models.visualize import plot_timeseries

from .new_model_example import *


if __name__ == '__main__':

    print('Visualize Module Example')

    # New Model Example
    # ===============
    m = ThrownObject()

    # Step 2: Setup for simulation 
    def future_load(t):
        return {}

    # Step 3: Simulate to impact
    event = 'impact'
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load,
                                                                             {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    

    # Display states
    # ==============
    fig = plot_timeseries(times, states, 
                          options = {'compact': False, 'suptitle': 'state evolution', 'title': True,
                                     'xlabel': 'time', 'ylabel': {'x': 'position', 'v': 'velocity'}, 'display_labels': 'minimal'},
                          legend  = {'display': True, 'display_at_subplot': 'all'} )
    fig = plot_timeseries(times, states, options = {'compact': True, 'suptitle': 'state evolution', 'title': 'example title',
                                                    'xlabel': 'time', 'ylabel':'position'})


    plt.show()