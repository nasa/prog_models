# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of generating a Dynamic Mode Decomposition surrogate model using the battery model 
"""
from prog_models.models import ThrownObject
import matplotlib.pyplot as plt

def run_example(): 
    ### Example 1: Standard DMD Application 
    ## Step 1: Create a model object
    m = ThrownObject(process_noise = 0, measurement_noise = 0)

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
        'states': ['v']
    }

    # Generate surrogate model  
    DMD_approx = m.generate_surrogate(load_functions,**options_surrogate)

    ## Step 4: Use surrogate model 
    # Simulation options for implementation of surrogate model
    options_sim = {
        'save_freq': 0.1, # Frequency at which results are saved, or equivalently time step in results    
        'threshold_keys': 'impact'
    }
    options_hf = {
        'dt': 0.1,
        'save_freq': 0.1,
        'threshold_keys': 'impact'
    }

    # Define loading profile 
    def future_loading(t, x=None):
        return m.InputContainer({})

    # Simulate to threshold using DMD approximation
    simulated_results = DMD_approx.simulate_to_threshold(future_loading,**options_sim)
    high_fidelity_results = m.simulate_to_threshold(future_loading, **options_hf)

    # Plot results
    simulated_results.event_states.plot(title='Surrogate Model Event States')
    high_fidelity_results.event_states.plot(title='Full Fidelity Model Event States')

    simulated_results.outputs.plot(title='Surrogate Model Position')
    high_fidelity_results.outputs.plot(title='Full Fidelity Model Position')
    plt.show()

    debug = 1

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()