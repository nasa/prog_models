# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example demonstrating the limitations of model parameter estiamtion feature.
"""

from prog_models.models.thrown_object import ThrownObject
from prog_models.models.battery_electrochem import BatteryElectroChemEOD
import sys

import warnings
warnings.filterwarnings("ignore")

# NOTE - These examples depict what NOT to do!
def run_example():
    
    # There are some examples where we want to showcase some limitations of the parameter estimation feature not working as intended.
    # Note, these examples depict what NOT to do 

    # Example 1: Unrealistic Parameter Selection.
    # Typically, our parameter estimation feature would fix errenous parameter selections, howeverr
    # however, when a parameter(s) is too unrealistic, we receive unexpected behavior


    # Step 1: Build the model with one particularly terrible guess in parameters
    # Here we're guessing that our battery's 'Alpha' value (which is typically a value between 0 and 1) is 200. Obviously not true!
    # Let's see how parameter estimation breaks!
    m = BatteryElectroChemEOD()

    options = {
        'save_freq': 200, # Frequency at which results are saved
        'dt': 10 # Timestep
    }

    # Defining Future Loading function. Look at sim_battery_eol.py for more details.
    def future_loading(t, x=None):
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
        return m.InputContainer({'i': i})

    # Simulating Results
    simulated_results = m.simulate_to(100, future_loading, **options)
    
    # With default parameters, we will recieve a small calc_error.
    defValue = m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt=10)
    
    # Printing out Value
    print(f'\nError calculation with default parameters: {defValue}\n')

    if defValue > 0.1:
        sys.exit('Parameter Estimation of Default Values results in too great of an error.')


    # Defining Parameters to be r
    # m.parameters['alpha'] = 200
    m.parameters['VolS'] = 4000
    # Division by zero occurs here
    # keys = ['qMax', 'VolS']
    keys = ['alpha']

    # Estimate Params on these unrealistic values
    # m.estimate_params([(simulated_results.times, simulated_results.inputs, simulated_results.outputs)], keys, dt=0.5)

    broken1 = m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt=10)

    print(f'\nAfter passing in an unrealistic value into our parameters, we recieve our new calc_error value would be {broken1}!')

    print(f"\nTo make matters worse, our 'alpha' value would be also be {m.parameters['alpha']}")

    if broken1 != 0 or m.parameters['alpha'] != 0:
        sys.exit('Parameter Estimation is working when it should be failing? Has Parameter Estimation changed?')
    
    # Sure enough- parameter estimation determined that the thrower's height wasn't 20 m, instead was closer to 1.9m, a much more reasonable height!

if __name__=='__main__':
    run_example()
