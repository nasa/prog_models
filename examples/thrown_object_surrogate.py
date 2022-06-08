# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of generating a Dynamic Mode Decomposition surrogate model using the battery model 
"""
from prog_models.models import ThrownObject
import matplotlib.pyplot as plt
import math
import numpy as np

def run_example(): 
    ### Example 1: Standard DMD Application 
    ## Step 1: Create a model object
    m = ThrownObject(process_noise = 0, measurement_noise = 0)
    m.parameters['cd'] = 0.1

    ## Step 2: Define future loading functions for training data 
    # Here, we define two specific loading profiles. These could also be generated programmatically, for as many loading profiles as desired 
    def future_loading(t, x=None):
        return m.InputContainer({})
    
    load_functions = [future_loading]

    ## Step 3: generate surrogate model 
    # Simulation options for training data and surrogate model generation
    # Note: here dt is less than save_freq. This means the model will iterate forward multiple steps per saved point.
    # This is commonly done to ensure accuracy. 
    options_surrogate = {
        'save_freq': 0.1, # For DMD, this value is the time step for which the surrogate model is generated
        'dt': 0.1, # For DMD, this value is the time step of the training data
        'threshold_keys': 'impact',
        'states': ['v'],
        'training_noise': 0
    }
    options_surrogate_dup = {
        'save_freq': 0.1, # For DMD, this value is the time step for which the surrogate model is generated
        'dt': 0.1, # For DMD, this value is the time step of the training data
        'threshold_keys': 'impact',
        'training_noise': 0
    }

    # Generate surrogate model 
    DMD_approx = m.generate_surrogate(load_functions,**options_surrogate) 
    DMD_approx_dup = m.generate_surrogate(load_functions,**options_surrogate_dup)

    ## Step 4: Use surrogate model 
    # Simulation options for implementation of surrogate model
    options_sim = {
        'save_freq': 0.1, # Frequency at which results are saved, or equivalently time step in results    
        'threshold_keys': 'impact'
    }
    options_hf = {
        'save_freq': 0.1,
        'dt': 0.1,
        'threshold_keys': 'impact'
    }
    

    # Define loading profile 
    def future_loading(t, x=None):
        return m.InputContainer({})

    # Simulate to threshold using DMD approximation
    surrogate_results = DMD_approx.simulate_to_threshold(future_loading,**options_sim)
    surrogate_results_duplicate = DMD_approx_dup.simulate_to_threshold(future_loading,**options_sim)
    high_fidelity_results = m.simulate_to_threshold(future_loading,**options_hf)

    # Extract DMD results 
    velocity = [surrogate_results.states[iter]['v'] for iter in range(len(surrogate_results.times))]
    position = [surrogate_results.states[iter]['x'] for iter in range(len(surrogate_results.times))]
    impact = [surrogate_results.states[iter]['impact'] for iter in range(len(surrogate_results.times))]
    falling = [surrogate_results.states[iter]['falling'] for iter in range(len(surrogate_results.times))]
    velocity_duplicate = [surrogate_results_duplicate.states[iter]['v'] for iter in range(len(surrogate_results_duplicate.times))]
    position_duplicate = [surrogate_results_duplicate.states[iter]['x'] for iter in range(len(surrogate_results_duplicate.times))]
    position_duplicate2 = [surrogate_results_duplicate.outputs[iter]['x'] for iter in range(len(surrogate_results_duplicate.times))]
    impact_duplicate = [surrogate_results_duplicate.states[iter]['impact'] for iter in range(len(surrogate_results_duplicate.times))]
    falling_duplicate = [surrogate_results_duplicate.states[iter]['falling'] for iter in range(len(surrogate_results_duplicate.times))]
    velocity_hf = [high_fidelity_results.states[iter]['v'] for iter in range(len(high_fidelity_results.times))]
    position_hf = [high_fidelity_results.states[iter]['x'] for iter in range(len(high_fidelity_results.times))]
    impact_hf = [high_fidelity_results.event_states[iter]['impact'] for iter in range(len(high_fidelity_results.times))]
    falling_hf = [high_fidelity_results.event_states[iter]['falling'] for iter in range(len(high_fidelity_results.times))]

    # plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.plot(high_fidelity_results.times,position_hf,'-b',label='HF')
    ax1.plot(surrogate_results.times,position,'--r',label='No duplicate')
    ax1.plot(surrogate_results_duplicate.times,position_duplicate,'-.g',label='Duplicate')
    ax1.set_title('Position')
    ax1.legend()
    ax2.plot(high_fidelity_results.times,velocity_hf,'-b',label='HF')
    ax2.plot(surrogate_results.times,velocity,'--r',label='No duplicate')
    ax2.plot(surrogate_results_duplicate.times,velocity_duplicate,'-.g',label='Duplicate')
    ax2.set_title('Velocity')
    ax3.plot(high_fidelity_results.times,impact_hf,'-b',label='HF')
    ax3.plot(surrogate_results.times,impact,'--r',label='No duplicate')
    ax3.plot(surrogate_results_duplicate.times,impact_duplicate,'-.g',label='Duplicate')
    ax3.set_title('Impact')
    ax4.plot(high_fidelity_results.times,falling_hf,'-b',label='HF')
    ax4.plot(surrogate_results.times,falling,'--r',label='No duplicate')
    ax4.plot(surrogate_results_duplicate.times,falling_duplicate,'-.g',label='Duplicate')
    ax4.set_title('Falling')
    plt.show()
    
    figB, axB = plt.subplots()
    axB.plot(surrogate_results_duplicate.times, position_duplicate,'-g',label='x in states')
    axB.plot(surrogate_results_duplicate.times, position_duplicate2,'--k',label='x in outputs')
    axB.set_title('Comparing duplicated x value')
    axB.legend()
    plt.show()


# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
