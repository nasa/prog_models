# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of a battery being simulated for a set period of time and then till threshold is met. Run using the command `python -m examples.sim`
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
        return {'i': i}
    # simulate for 200 seconds
    print('\n\n------------------------------------------------')
    print('Simulating for 200 seconds\n\n')
    (times, inputs, states, outputs, event_states) = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183}, print = True)

    # Simulate to threshold
    print('\n\n------------------------------------------------')
    print('Simulating to threshold\n\n')
    options = {
        'save_freq': 100, # Frequency at which results are saved
        'dt': 2, # Timestep
        'print': True
    }
    (times, inputs, states, outputs, event_states) = batt.simulate_to_threshold(future_loading, {'t': 18.95, 'v': 4.183}, **options)

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
