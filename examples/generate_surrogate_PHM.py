# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of generating a Dynamic Mode Decomposition surrogate model using the battery model 
"""

from sys import get_coroutine_origin_tracking_depth
from prog_models.models import BatteryElectroChemEOD as Battery

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def run_example(): 
    ### Example 1: Standard DMD Application 
    ## Step 1: Create a model object
    batt = Battery()

    ## Step 2: Define future loading functions for training data 
    # Here, we define two specific loading profiles. These could also be generated programmatically, for as many loading profiles as desired 
    def future_loading_1(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 600):
            i = 2
        elif (t < 900):
            i = 1
        elif (t < 1800):
            i = 4
        else:
            i = 2
        return batt.InputContainer({'i': i})
    
    def future_loading_2(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 200):
            i = 4
        elif (t < 500):
            i = 2
        elif (t < 1300):
            i = 1
        else:
            i = 6
        return batt.InputContainer({'i': i})
    
    def future_loading_3(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 1000):
            i = 2
        else:
            i = 6.5
        return batt.InputContainer({'i': i})
    
    def future_loading_4(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 700):
            i = 5.5
        elif (t < 1400):
            i = 1.3
        elif (t < 1800):
            i = 3.8
        else:
            i = 4
        return batt.InputContainer({'i': i})

    def future_loading_5(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 300):
            i = 1.2
        elif (t < 600):
            i = 1.8
        elif (t < 800):
            i = 4
        elif (t < 1000):
            i = 2.5
        elif (t < 1200):
            i = 5.3
        elif (t < 1600):
            i = 3.8
        else:
            i = 6.2
        return batt.InputContainer({'i': i})

    def future_loading_6(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 500):
            i = 2.7
        elif (t < 800):
            i = 5
        elif (t < 1200):
            i = 3.9
        elif (t < 1500):
            i = 2.5
        else:
            i = 6
        return batt.InputContainer({'i': i})

    def future_loading_7(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 200):
            i = 0.5
        elif (t < 400):
            i = 1.2
        elif (t < 600):
            i = 3.9
        elif (t < 800):
            i = 2.3
        elif (t < 1000):
            i = 0.8
        elif (t < 1400):
            i = 4.7
        elif (t < 1800):
            i = 3.1
        elif (t < 2200):
            i = 1
        else:
            i = 3
        return batt.InputContainer({'i': i})

    def future_loading_8(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 200):
            i = 2.5
        elif (t < 400):
            i = 3
        elif (t < 600):
            i = 1.7
        elif (t < 800):
            i = 2.3
        elif (t < 1000):
            i = 0.5
        elif (t < 1400):
            i = 2.2
        elif (t < 1800):
            i = 3.6
        elif (t < 2200):
            i = 2
        else:
            i = 3
        return batt.InputContainer({'i': i})

    def future_loading_9(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 800):
            i = 2.5
        else:
            i = 7
        return batt.InputContainer({'i': i})
    
    def future_loading_10(t, x=None):
        # Variable (piece-wise) future loading scheme 
        i = 1.5
        return batt.InputContainer({'i': i})

    def future_loading_11(t, x=None):
        # Variable (piece-wise) future loading scheme 
        i = 3
        return batt.InputContainer({'i': i})

    def future_loading_12(t, x=None):
        # Variable (piece-wise) future loading scheme 
        i = 4.5
        return batt.InputContainer({'i': i})

    def future_loading_13(t, x=None):
        # Variable (piece-wise) future loading scheme 
        i = 6
        return batt.InputContainer({'i': i})

    def future_loading_14(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 500):
            i = 2.5
        elif (t < 900):
            i = 3
        elif (t < 1400):
            i = 4
        elif (t < 2000):
            i = 1.3
        else:
            i = 2.6
        return batt.InputContainer({'i': i})

    def future_loading_15(t, x=None):
        if (t < 650):
            i = 1.5
        elif (t < 900):
            i = 3.9
        elif (t < 1600):
            i = 2.7
        elif (t < 2000):
            i = 1
        else:
            i = 3
        return batt.InputContainer({'i': i})

    def future_loading_16(t, x=None):
        if (t < 1000):
            i = 4.3
        elif (t < 1500):
            i = 2.9
        elif (t < 2000):
            i = 1
        else:
            i = 5
        return batt.InputContainer({'i': i})

    def future_loading_17(t, x=None):
        if (t < 1000):
            i = 2
        elif (t < 1500):
            i = 3
        elif (t < 2000):
            i = 1
        else:
            i = 4.5
        return batt.InputContainer({'i': i})

    def future_loading_18(t, x=None):
        if (t < 500):
            i = 2.5
        elif (t < 1000):
            i = 1.5
        elif (t < 2000):
            i = 3
        else:
            i = 5
        return batt.InputContainer({'i': i})

    def future_loading_19(t, x=None):
        if (t < 800):
            i = 3.8
        elif (t < 1000):
            i = 6
        else:
            i = 2.3
        return batt.InputContainer({'i': i})

    def future_loading_20(t, x=None):
        if (t < 800):
            i = 2
        elif (t < 1000):
            i = 4.5
        else:
            i = 1.5
        return batt.InputContainer({'i': i})

    load_functions = [future_loading_1, future_loading_2, future_loading_3, future_loading_4, future_loading_5, future_loading_6, 
                    future_loading_7, future_loading_8, future_loading_9, future_loading_10, future_loading_11, future_loading_12]

    # Define loading profile 
    future_loading = future_loading_15

    # Define noise for HF model and training data
    batt.parameters['process_noise'] = 0 #2e-05
    batt.parameters['process_noise']['qpS'] = 0.25
    batt.parameters['process_noise']['qpB'] = 0.25

    ### Ground truth/"highest" high-fidelity
    options_gt = {
        'dt': 0.1, #options_surrogate['dt'], #0.1,
        'save_freq': 0.1 #options_surrogate['save_freq']
    }
    ground_truth_results = batt.simulate_to_threshold(future_loading,**options_gt)

    save_freq_vec = [1, 5]
    voltError_surrogate = []
    voltError_hf = []
    socError_surrogate = []
    socError_hf = []
    for iter, save_val in enumerate(save_freq_vec):

        options_surrogate = {
            'save_freq': save_val, # For DMD, this value is the time step for which the surrogate model is generated
            'dt': 0.1, # For DMD, this value is the time step of the training data
            'trim_data_to': 0.8, # Value between 0 and 1 that determines the fraction of data resulting from simulate_to_threshold that is used to train DMD surrogate model
            'training_noise': 0.01,
            'outputs': ['v']
        }

        # Generate surrogate model  
        surrogate = batt.generate_surrogate(load_functions,**options_surrogate)
        surrogate.parameters['process_noise'] = 0

        # Surrogate
        options_surr = {
            'save_freq': options_surrogate['save_freq'] # Frequency at which results are saved, or equivalently time step in results
        }
        surrogate_results = surrogate.simulate_to_threshold(future_loading,**options_surr)

        # High-fidelity
        options_hf = {
            'dt': options_surrogate['save_freq'], 
            'save_freq': options_surrogate['save_freq']
        }
        high_fidelity_results = batt.simulate_to_threshold(future_loading,**options_hf)

        # Extract results
        time_dmd = surrogate_results.times
        time_hf = high_fidelity_results.times
        time_gt = ground_truth_results.times
        voltage_dmd = [surrogate_results.outputs[iter1]['v'] for iter1 in range(len(surrogate_results.times))]
        voltage_hf = [high_fidelity_results.outputs[iter2]['v'] for iter2 in range(len(high_fidelity_results.times))]
        voltage_gt_temp = [ground_truth_results.outputs[iter3]['v'] for iter3 in range(len(ground_truth_results.times))]
        soc_dmd = [surrogate_results.event_states[iter1]['EOD'] for iter1 in range(len(surrogate_results.times))]
        soc_hf = [high_fidelity_results.event_states[iter2]['EOD'] for iter2 in range(len(high_fidelity_results.times))]
        soc_gt_temp = [ground_truth_results.event_states[iter3]['EOD'] for iter3 in range(len(ground_truth_results.times))]

        # Check time vector: times should be at the same indices, but not necessarily the same lengths
        if time_dmd[0:len(time_hf)] != time_hf:
            breakpoint()

        time_keep = time_hf[0:-1] # Get rid of last time point so time_hf is smaller than time_gt for interpolation

        # Interpolate ground truth
        voltage_gt = interp1d(time_gt,voltage_gt_temp)(time_keep) # interp1d(time_gt,voltage_gt_temp)(time_hf)
        soc_gt = interp1d(time_gt,soc_gt_temp)(time_keep)           

        voltError_surrogate.append(sum([(voltage_gt[iterB] - voltage_dmd[iterB])**2 for iterB in range(len(time_keep))])/len(time_keep))
        voltError_hf.append(sum([(voltage_gt[iterB] - voltage_hf[iterB])**2 for iterB in range(len(time_keep))])/len(time_keep))
        socError_surrogate.append(sum([(soc_gt[iterB] - soc_dmd[iterB])**2 for iterB in range(len(time_keep))])/len(time_keep))
        socError_hf.append(sum([(soc_gt[iterB] - soc_hf[iterB])**2 for iterB in range(len(time_keep))])/len(time_keep))
    
    ### For computational efficiency (run separately from above)
    """
    import time
    time_surrogate = []
    time_hf = []
    time_gt = [] 
    
    for ii in range(30):
        t1_gt = time.process_time()
        ground_truth_results = batt.simulate_to_threshold(future_loading,**options_gt)
        t2_gt = time.process_time()
        time_surrogate.append(t2_gt - t1_gt)

    for ii in range(30):
        t1_s = time.process_time()
        simulated_results = surrogate.simulate_to_threshold(future_loading,**options_surr)
        t2_s = time.process_time()
        time_surrogate.append(t2_s - t1_s)
    
        t1_h = time.process_time()
        high_fidelity_results = batt.simulate_to_threshold(future_loading,**options_hf)
        t2_h = time.process_time()
        time_hf.append(t2_h - t1_h)
    """
    
    ### Plotting
    """
    plt.subplots()
    plt.plot(high_fidelity_results.times, voltage_hf,'-b',label='High fidelity result')
    plt.plot(simulated_results.times,voltage_dmd,'--r',label='DMD approximation')
    plt.legend()
    plt.title('Comparing DMD approximation to high-fidelity model results')

    plt.subplots()
    plt.plot(high_fidelity_results.times, soc_hf,'-b',label='High fidelity result')
    plt.plot(simulated_results.times,soc_dmd,'--r',label='DMD approximation')
    plt.legend()
    plt.title('Comparing DMD approximation to high-fidelity model results')
    """

    debug = 1

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()