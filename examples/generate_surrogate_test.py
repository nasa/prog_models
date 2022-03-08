# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of generating a surrogate model using the battery model 
"""
# from prog_models.models import BatteryCircuit as Battery
# VVV Uncomment this to use Electro Chemistry Model VVV
# from prog_models.models import BatteryElectroChem as Battery
from prog_models.models import BatteryElectroChemEOD as Battery

def run_example(): 
    # Step 1: Create a model object
    batt = Battery()

    # Step 2: Define future loading function 
    def future_loading(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 1000):
            i = 1
        elif (t < 2000):
            i = 1
        elif (t < 3000):
            i = 1
        else:
            i = 1
        return {'i': i}

    # simulate for 360 seconds
    print('\n\n------------------------------------------------')
    print('Simulating for 360 seconds\n\n')

    options = {
        'save_freq': 1, # Frequency at which results are saved
        'dt': 0.1 # Timestep
    #    'print': True
    }

    # Set noise to 0
    batt.parameters['process_noise'] = 0

    # Test surrogate model function 
    DMD_Mat = batt.generate_surrogate(future_loading,**options)

    return DMD_Mat

results = run_example() 