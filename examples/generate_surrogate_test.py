# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of generating a Dynamic Mode Decomposition surrogate model using the battery model 
"""
# from prog_models.models import BatteryCircuit as Battery
# VVV Uncomment this to use Electro Chemistry Model VVV
# from prog_models.models import BatteryElectroChem as Battery
from prog_models.models import BatteryElectroChemEOD as Battery
from scipy.interpolate import interp1d
import numpy as np

def run_example(): 
    # Step 1: Create a model object
    batt = Battery()

    # Step 2: Define future loading function 
    def future_loading_1(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 1000):
            i = 3
        else:
            i = 4
        return batt.InputContainer({'i': i})
    
    def future_loading_2(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 2000):
            i = 2
        else:
            i = 5
        return batt.InputContainer({'i': i})
    
    load_functions = [future_loading_1, future_loading_2]

    options_surrogate = {
        'save_freq': 1, # For DMD, this value is the time step for which the surrogate model is generated, and therefore the dt imposed in simulate_to_threshold (via next_state, where dt is arbitrary)
        'dt': 0.1, # For DMD, this value is the time step for the predictions that generate training data
        'data_len': 0.67, # Value between 0 and 1 that determines fraction of data resulting from simulate_to_threshold that is used to train DMD surrogate model
        # 'states_dmd': ['qnB','qnS','qpB','qpS','Vo','Vsn','Vsp'],
        'outputs_dmd': ['v']
    }

    # Set noise to 0
    batt.parameters['process_noise'] = 0

    # Generate surrogate model function 
    DMD_approx = batt.generate_surrogate_dmd(load_functions,**options_surrogate)

    # Simulation Options
    options_sim = {
        'save_freq': 1 # Frequency at which results are saved
        # 'save_pts': [2],
    }

    # Define Loading Profile 
    def future_loading(t, x=None):
        # Adjust time to previous time step for DMD consistency
            # simulate_to_threshold in PrognosticsModel calculates load at the next time point and uses this as input to next_state
            # DMD, however, takes the state and load at a particular (same) time, and uses this to calculate the state at the next time 
            # Thus, when calling future_loading with DMD + LinearModel, we need the load input for next_state to be at the previous time point to be consistent with the previous state, so we subtract dt from the input time 
        # Note: this should be made more rigorous in the future 
 
        if (t < 1000):
            i = 3
        elif (t < 2000):
            i = 2
        else:
            i = 4
        return batt.InputContainer({'i': i})

    # Simulate to threshold
    simulated_results = DMD_approx.simulate_to_threshold(future_loading,**options_sim)
    # simulated_results = DMD_approx.simulate_to_threshold(future_loading)

    # Debugging - plot
    time_vec = []
    voltage_vec = []
    voltage_vec2 = []
    current_vec = []
    SOC_vec = []
    SOC_vec2 = []
    ## Use if dmd_dt ~= user_dt 
    # for iter in range(len(simulated_results)):
    #     time_vec.append(simulated_results[iter]['time'])
    #     voltage_vec.append(simulated_results[iter]['v'])
    #     SOC_vec.append(simulated_results[iter]['EOD'])

    ## Use if dmd_dt == user_dt 
    for iter in range(len(simulated_results.times)):
        time_vec.append(simulated_results.times[iter])
        voltage_vec.append(simulated_results.states[iter]['v'])
        voltage_vec2.append(simulated_results.outputs[iter]['v'])
        SOC_vec.append(simulated_results.states[iter]['EOD'])
        SOC_vec2.append(simulated_results.event_states[iter]['EOD'])
        current_vec.append(simulated_results.inputs[iter]['i'])


    return simulated_results

results = run_example() 