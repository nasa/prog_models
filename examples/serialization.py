# Copyright © 2022 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of serializing and de-serializing a surrogate model using JSON and pickling methods
"""

from prog_models.models import BatteryElectroChemEOD as Battery
from prog_models.data_models import DMDModel

import matplotlib.pyplot as plt
import pickle

def run_example():  
    ## Step 1: Create a model object
    batt = Battery()

    ## Step 2: Define future loading functions for training data 
    def future_loading_1(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 500):
            i = 3
        elif (t < 1000):
            i = 2
        elif (t < 1500):
            i = 0.5
        else:
            i = 4.5
        return batt.InputContainer({'i': i})
    
    def future_loading_2(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 300):
            i = 2
        elif (t < 800):
            i = 3.5
        elif (t < 1300):
            i = 4
        elif (t < 1600):
            i = 1.5
        else:
            i = 5
        return batt.InputContainer({'i': i})
    
    load_functions = [future_loading_1, future_loading_2]

    ## Step 3: Create surrogate model 
    options_surrogate = {
        'save_freq': 1, # For DMD, this value is the time step for which the surrogate model is generated
        'dt': 0.1, # For DMD, this value is the time step of the training data
        'trim_data_to': 0.7 # Value between 0 and 1 that determines the fraction of data resulting from simulate_to_threshold that is used to train DMD surrogate model
    }

    # Set noise in Prognostics Model, default for surrogate model is also this value. Set to 0 to illustrate match between original surrogate and serialized version
    batt.parameters['process_noise'] = 0 

    # Generate surrogate model  
    surrogate_orig = batt.generate_surrogate(load_functions,**options_surrogate)

    ### Step 4: serialize model for future use 
    # METHOD 1: Serialize with JSON 
    save_surrogate = surrogate_orig.to_json()

    # DMDModel can be called directly with serialized result
    surrogate_serial_1 = DMDModel.from_json(save_surrogate)

    # Serialized result can also be saved to a text file and uploaded later using the following code:
    txtFile = open("surrogate_model_save_json.txt", "w")
    txtFile.write(save_surrogate)
    txtFile.close() 

    with open('surrogate_model_save_json.txt') as infile: 
        load_surrogate_json = infile.read()

    surrogate_serial_2 = DMDModel.from_json(load_surrogate_json)

    # METHOD 2: Serialize by pickling
    pickle.dump(surrogate_orig, open('surrogate_model_save_pkl.pkl','wb'))
    surrogate_pkl = pickle.load(open('surrogate_model_save_pkl.pkl','rb'))

    ## Step 5: Simulate to threshold and compare results
    # Simulation options for implementation of surrogate model
    options_sim = {
        'save_freq': 1 # Frequency at which results are saved, or equivalently time step in results
    }

    # Define loading profile 
    def future_loading(t, x=None):
        if (t < 600):
            i = 3
        elif (t < 1000):
            i = 2
        elif (t < 1500):
            i = 1.5
        else:
            i = 4
        return batt.InputContainer({'i': i})

    # Simulate to threshold 
    results_orig = surrogate_orig.simulate_to_threshold(future_loading,**options_sim)
    results_serial_1 = surrogate_serial_1.simulate_to_threshold(future_loading, **options_sim)
    results_serial_2 = surrogate_serial_2.simulate_to_threshold(future_loading, **options_sim)
    results_serial_3 = surrogate_pkl.simulate_to_threshold(future_loading, **options_sim)

    # Plot results for comparison
    voltage_orig = [results_orig.outputs[iter]['v'] for iter in range(len(results_orig.times))]
    voltage_serial_1 = [results_serial_1.outputs[iter]['v'] for iter in range(len(results_serial_1.times))]
    voltage_serial_2 = [results_serial_2.outputs[iter]['v'] for iter in range(len(results_serial_2.times))]
    voltage_serial_3 = [results_serial_3.outputs[iter]['v'] for iter in range(len(results_serial_3.times))]

    plt.plot(results_orig.times,voltage_orig,'-b',label='Original surrogate') 
    plt.plot(results_serial_1.times,voltage_serial_1,'--r',label='First JSON serialized surrogate') 
    plt.plot(results_serial_2.times,voltage_serial_2,'-.g',label='Second JSON serialized surrogate') 
    plt.plot(results_serial_3.times, voltage_serial_3, '--y', label='Pickled serialized surrogate')
    plt.legend()
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage (volts)')
    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
