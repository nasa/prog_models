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
        'events': [],
        'training_noise': 0 # 0.06
    }

    # Generate surrogate model  
    DMD_approx = m.generate_surrogate(load_functions,**options_surrogate)

    ## Step 4: Use surrogate model 
    # Simulation options for implementation of surrogate model
    options_sim = {
        'save_freq': 0.1, # Frequency at which results are saved, or equivalently time step in results    
        # 'threshold_keys': 'impact',
        # 'dt': 0.1
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
    # simulated_results = DMD_approx.simulate_to_threshold(future_loading,**options_sim)
    simulated_results = DMD_approx.simulate_to(8,future_loading,**options_sim)
    # high_fidelity_results = m.simulate_to_threshold(future_loading, **options_hf)
    high_fidelity_results = m.simulate_to(8, future_loading, **options_hf)

    # Plot results
    # simulated_results.event_states.plot(title='Surrogate Model Event States')
    # high_fidelity_results.event_states.plot(title='Full Fidelity Model Event States')

    # simulated_results.outputs.plot(title='Surrogate Model Position')
    # high_fidelity_results.outputs.plot(title='Full Fidelity Model Position')
    # plt.show()

    # Extract DMD results 
    time_dmd = [simulated_results.times[iter] for iter in range(len(simulated_results.times))]
    velocity_dmd = [simulated_results.states[iter]['v'] for iter in range(len(simulated_results.times))]
    position_dmd = [simulated_results.states[iter]['x'] for iter in range(len(simulated_results.times))]
    # falling_dmd = [simulated_results.states[iter]['falling'] for iter in range(len(simulated_results.times))]
    # impact_dmd = [simulated_results.states[iter]['impact'] for iter in range(len(simulated_results.times))]

    # Extract HF results
    time_hf = [high_fidelity_results.times[iter] for iter in range(len(high_fidelity_results.times))]
    velocity_hf = [high_fidelity_results.states[iter]['v'] for iter in range(len(high_fidelity_results.times))]
    position_hf = [high_fidelity_results.states[iter]['x'] for iter in range(len(high_fidelity_results.times))]
    # falling_hf = [high_fidelity_results.event_states[iter]['falling'] for iter in range(len(high_fidelity_results.times))]
    # impact_hf = [high_fidelity_results.event_states[iter]['impact'] for iter in range(len(high_fidelity_results.times))]

    # Test if DMD and HF at same times 
    time_bool = [time_dmd[iter] == time_hf[iter] for iter in range(min(len(time_dmd),len(time_hf)))]
    if sum(time_bool) != min(len(time_dmd),len(time_hf)):
        print('Error: times are not equal.')

    # Calculate error:
    # rmse_impact = math.sqrt(sum([((impact_hf[iter] - impact_dmd[iter])**2)/min(len(impact_dmd),len(impact_hf)) for iter in range(min(len(impact_dmd),len(impact_hf)))]))
    # rmse_falling = math.sqrt(sum([((falling_hf[iter] - falling_dmd[iter])**2)/min(len(falling_dmd),len(falling_hf)) for iter in range(min(len(falling_dmd),len(falling_hf)))]))
    rmse_position = math.sqrt(sum([((position_hf[iter] - position_dmd[iter])**2)/min(len(position_dmd),len(position_hf)) for iter in range(min(len(position_dmd),len(position_hf)))]))
    rmse_velocity = math.sqrt(sum([((velocity_hf[iter] - velocity_dmd[iter])**2)/min(len(velocity_dmd),len(velocity_hf)) for iter in range(min(len(velocity_dmd),len(velocity_hf)))]))    

    """
    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(time_hf,position_hf,'-b',label='High fidelity')
    ax1.plot(time_dmd,position_dmd,'--r',label='DMD')
    ax1.set_title('Position')
    ax2.plot(time_hf,falling_hf,'-b',label='High fidelity')
    ax2.plot(time_dmd,falling_dmd,'--r',label='DMD')
    ax2.set_title('Falling')
    ax3.plot(time_hf,impact_hf,'-b',label='High fidelity')
    ax3.plot(time_dmd,impact_dmd,'--r',label='DMD')
    ax3.set_title('Impact')
    """

    debug = 1

def run_comparison(): 
    
    drag_vec = np.arange(0.001, 1, 0.01) # [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.145, 0.25, 0.5, 0.75, 1, 1.3, 1.5] #, 1.8, 2, 2.3, 2.5, 2.8, 3]
    rmse_impact = []
    rmse_falling = []
    rmse_position = []
    rmse_velocity = []

    for drag_iter, drag_coeff in enumerate(drag_vec):
        m = ThrownObject(process_noise = 0, measurement_noise = 0)
        m.parameters['cd'] = drag_coeff

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
            # 'events': [],
            'training_noise': 0 # 0.06
        }

        # Generate surrogate model  
        DMD_approx = m.generate_surrogate(load_functions,**options_surrogate)

        ## Step 4: Use surrogate model 
        # Simulation options for implementation of surrogate model
        options_sim = {
            'save_freq': 0.1, # Frequency at which results are saved, or equivalently time step in results    
            # 'threshold_keys': 'impact',
            # 'dt': 0.1
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
        # simulated_results = DMD_approx.simulate_to_threshold(future_loading,**options_sim)
        simulated_results = DMD_approx.simulate_to(8,future_loading,**options_sim)
        high_fidelity_results = m.simulate_to_threshold(future_loading, **options_hf)
        # high_fidelity_results = m.simulate_to(8, future_loading, **options_hf)

        # Plot results
        # simulated_results.event_states.plot(title='Surrogate Model Event States')
        # high_fidelity_results.event_states.plot(title='Full Fidelity Model Event States')

        # simulated_results.outputs.plot(title='Surrogate Model Position')
        # high_fidelity_results.outputs.plot(title='Full Fidelity Model Position')
        # plt.show()

        # Extract DMD results 
        time_dmd = [simulated_results.times[iter] for iter in range(len(simulated_results.times))]
        velocity_dmd = [simulated_results.states[iter]['v'] for iter in range(len(simulated_results.times))]
        position_dmd = [simulated_results.states[iter]['x'] for iter in range(len(simulated_results.times))]
        # falling_dmd = [simulated_results.states[iter]['falling'] for iter in range(len(simulated_results.times))]
        # impact_dmd = [simulated_results.states[iter]['impact'] for iter in range(len(simulated_results.times))]

        # Extract HF results
        time_hf = [high_fidelity_results.times[iter] for iter in range(len(high_fidelity_results.times))]
        velocity_hf = [high_fidelity_results.states[iter]['v'] for iter in range(len(high_fidelity_results.times))]
        position_hf = [high_fidelity_results.states[iter]['x'] for iter in range(len(high_fidelity_results.times))]
        # falling_hf = [high_fidelity_results.event_states[iter]['falling'] for iter in range(len(high_fidelity_results.times))]
        # impact_hf = [high_fidelity_results.event_states[iter]['impact'] for iter in range(len(high_fidelity_results.times))]

        # Test if DMD and HF at same times 
        time_bool = [time_dmd[iter] == time_hf[iter] for iter in range(min(len(time_dmd),len(time_hf)))]
        if sum(time_bool) != min(len(time_dmd),len(time_hf)):
            print('Error: times are not equal.')

        # Calculate error:
        # rmse_impact.append(math.sqrt(sum([((impact_hf[iter] - impact_dmd[iter])**2)/min(len(impact_dmd),len(impact_hf)) for iter in range(min(len(impact_dmd),len(impact_hf)))])))
        # rmse_falling.append(math.sqrt(sum([((falling_hf[iter] - falling_dmd[iter])**2)/min(len(falling_dmd),len(falling_hf)) for iter in range(min(len(falling_dmd),len(falling_hf)))])))
        rmse_position.append(math.sqrt(sum([((position_hf[iter] - position_dmd[iter])**2)/min(len(position_dmd),len(position_hf)) for iter in range(min(len(position_dmd),len(position_hf)))])))
        rmse_velocity.append(math.sqrt(sum([((velocity_hf[iter] - velocity_dmd[iter])**2)/min(len(velocity_dmd),len(velocity_hf)) for iter in range(min(len(velocity_dmd),len(velocity_hf)))])))  

        a = 1
        """
        # plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.plot(time_hf,position_hf,'-b',label='High fidelity')
        ax1.plot(time_dmd,position_dmd,'--r',label='DMD')
        ax1.set_title('Position')
        ax2.plot(time_hf,falling_hf,'-b',label='High fidelity')
        ax2.plot(time_dmd,falling_dmd,'--r',label='DMD')
        ax2.set_title('Falling')
        ax3.plot(time_hf,impact_hf,'-b',label='High fidelity')
        ax3.plot(time_dmd,impact_dmd,'--r',label='DMD')
        ax3.set_title('Impact')
        """
    results = {'drag_coefficients': drag_vec, 'rmse_impact': rmse_impact, 'rmse_falling': rmse_falling, 'rmse_position': rmse_position, 'rmse_velocity': rmse_velocity}
    return results

# This allows the module to be executed directly 
if __name__ == '__main__':
    # run_example()
    run_comparison()
