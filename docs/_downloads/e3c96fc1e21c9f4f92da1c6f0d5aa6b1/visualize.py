# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of Visualization Module. Run using the command `python -m examples.visualize`
"""

import matplotlib.pyplot as plt
from prog_models.visualize import plot_timeseries

from .new_model import ThrownObject

def run_example():
    print('Visualize Module Example')
    m = ThrownObject()

    # Step 2: Setup for simulation 
    def future_load(t, x=None):
        return {}

    # Step 3: Simulate to impact
    event = 'impact'
    first_output = {'x':m.parameters['thrower_height']}
    options={'dt':0.005, 'save_freq':1}
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load,
                                                                             first_output, 
                                                                             threshold_keys=[event], 
                                                                             **options)
    

    # Display states
    # ==============
    plot_timeseries(times, states, 
                          options = {'compact': False, 'suptitle': 'state evolution', 'title': True,
                                     'xlabel': 'time', 'ylabel': {'x': 'position', 'v': 'velocity'}, 'display_labels': 'minimal'},
                          legend  = {'display': True, 'display_at_subplot': 'all'} )
    plot_timeseries(times, states, options = {'compact': True, 'suptitle': 'state evolution', 'title': 'example title',
                                                    'xlabel': 'time', 'ylabel':'position'})
    plt.show()

if __name__ == '__main__':
    run_example()
