# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of a battery being simulated for a set period of time and then till threshold is met.
"""

from prog_models.models import BatteryCircuit as Battery
# VVV Uncomment this to use Electro Chemistry Model VVV
# from prog_models.models import BatteryElectroChem as Battery

def run_example(): 
    # Step 1: Create a model object
    batt = Battery()

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
        return batt.InputContainer({'i': i})
    # simulate for 200 seconds
    print('\n\n------------------------------------------------')
    print('Simulating for 200 seconds\n\n')
    simulated_results = batt.simulate_to(200, future_loading, print = True, progress = True)

    # Simulate to threshold
    print('\n\n------------------------------------------------')
    print('Simulating to threshold\n\n')
    options = {
        'save_freq': 100, # Frequency at which results are saved
        'dt': 2, # Timestep
        'print': True,
        'progress': True
    }
    simulated_results = batt.simulate_to_threshold(future_loading, **options)

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
