# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of a battery being simulated for a set period of time and then till threshold is met. Run using the command `python -m examples.sim_battery_eol`
"""

import numpy as np
from prog_models.models import BatteryElectroChem as Battery

def run_example(): 
    # Step 1: Create a model object
    batt = Battery()

    # Step 2: Define future loading function 
    # Here we're using a function designed to charge until 0.95, 
    # then discharge until 0.05
    load = 1

    def future_loading(t, x=None):
        nonlocal load 

        # Rule for loading after initialization
        if x is not None:
            # Current event state in the form {'EOD': <(0, 1)>, 'InsufficientCapacity': <(0, 1)>}
            event_state = batt.event_state(x)
            if event_state["EOD"] > 0.95:
                load = 1  # Discharge
            elif event_state["EOD"] < 0.05:
                load = -1  # Charge
        # Rule for loading at initialization
        return {'i': load}

    # Simulate to EOL Threshold
    print('\n\n------------------------------------------------')
    print('Simulating to threshold\n\n')
    options = {
        'save_freq': 1000,  # Frequency at which results are saved
        'dt': 2,  # Timestep
        'threshold_keys': ['InsufficientCapacity']  # Simulate to InsufficientCapacity
    }
    (times, inputs, states, outputs, event_states) = batt.simulate_to_threshold(future_loading, {'t': 18.95, 'v': 4.183}, **options)

    for i in range(len(times)): # Print Results
        print("Time: {}\n\tInput: {}\n\tState: {}\n\tOutput: {}\n\tEvent State: {}\n".format(times[i], inputs[i], states[i], outputs[i], event_states[i]))

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()