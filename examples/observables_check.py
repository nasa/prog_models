# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of benchmarking models. 
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

    # Simulate to 600 sec
    (times, inputs, states, outputs, event_states) = batt.simulate_to(600, future_loading)

    # Using state at 600 sec, run CLE algorithm 
    CLE_estimate = batt.current_limit_est(states[-1], inputs[-1], delta_t=10, VCutoff=3.0)

    # Step 3: Benchmark simulation for 600 seconds
    # print('Benchmarking:')
    # def sim():  
    #     (times, inputs, states, outputs, event_states) = batt.simulate_to_threshold(future_loading)
    # time = timeit(sim, number=10)

    # Print results
    # print('Simulation Time: {} ms/sim'.format(time*2))

    debug_check = 1

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
