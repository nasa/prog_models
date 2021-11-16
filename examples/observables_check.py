# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of implementation of CLE algorithm to determine maximum current possible from battery on time interval 
"""

from timeit import timeit
from prog_models.models import BatteryElectroChemEOD

def run_example():
    # Step 1: Create a model object
    batt = BatteryElectroChemEOD()
    
    # Step 2: Define future loading function 
    def future_loading(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 600):
            i = 2
        elif (t < 900):
            i = 1
        elif (t < 1800):
            i = 4
        elif (t < 3000):
            i = 2
        else:
            i = 3
        return {'i': i}

    # Define parameters for CLE test calculations: 
    time_to_sim_to = 300 # time cutoff value/time interval of simulation 
    V_cutoff = 3.2 # voltage cutoff value 

    # Initialize to default parameters
    initial_state = batt.parameters['x0']

    # Run CLE algorithm from default inital state
    i_start = future_loading(0)
    CLE_estimate = batt.current_limit_est(initial_state, i_start, delta_t=time_to_sim_to, VCutoff=V_cutoff)

    # Simulate to 600 sec
    (times, inputs, states, outputs, event_states) = batt.simulate_to(600, future_loading)

    # Using state at 600 sec, run CLE algorithm 
    CLE_estimate_2 = batt.current_limit_est(states[-1], inputs[-1], delta_t=time_to_sim_to, VCutoff=V_cutoff)

    debug_check = 1

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
