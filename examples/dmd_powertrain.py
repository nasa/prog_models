
# Imports:
import matplotlib.pyplot as plt

from prog_models.models import Powertrain, ESC, DCMotor


# Create a model object
esc = ESC()
motor = DCMotor()
powertrain = Powertrain(esc, motor)

# Define future loading function - 100% duty all the time
def future_loading(t, x=None):
    return powertrain.InputContainer({
        'duty': 1,
        'v': 23
    })

# Simulate to threshold
print('\n\n------------------------------------------------')
print('Simulating to threshold\n\n')
options_hf = {'dt': 2e-5, 
              'save_freq': 0.1,
              'print': False
            }
sim_hf = powertrain.simulate_to(2, future_loading, **options_hf)
v_rot_hf = [sim_hf.outputs[iter]['v_rot'] for iter in range(len(sim_hf.times))]
theta_hf = [sim_hf.outputs[iter]['theta'] for iter in range(len(sim_hf.times))]


## Generate surrogate model 
# Simulation options for training data and surrogate model generation
options_surrogate = {
    'save_freq': options_hf['save_freq'], # For DMD, this value is the time step for which the surrogate model is generated
    'dt': options_hf['dt'], # For DMD, this value is the time step of the training data
    'threshold': 2, 
    'trim_data': 1, # Value between 0 and 1 that determines the fraction of data resulting from simulate_to_threshold that is used to train DMD surrogate model
    # 'state_keys': ['Vsn','Vsp','tb'], # Define internal states to be included in surrogate model
    # 'output_keys': ['v'] # Define outputs to be included in surrogate model 
}

# Generate surrogate model  
load_functions = [future_loading]

surrogate = powertrain.generate_surrogate(load_functions,**options_surrogate)

## Use surrogate model 
# The surrogate model can now be used anywhere the original model is used. It is interchangeable with the original model. 
# The surrogate model results will be faster but less accurate than the original model. 

# Simulation options for implementation of surrogate model
options_sim = {
    'save_freq': options_hf['save_freq'] # Frequency at which results are saved, or equivalently time step in results
}

# Simulate to threshold using DMD approximation
sim_surr = surrogate.simulate_to(2, future_loading,**options_sim)

# Calculate Error
MSE = powertrain.calc_error(sim_surr.times, sim_surr.inputs, sim_surr.outputs)
print('MSE:',MSE)

# Plot results
# simulated_results.inputs.plot(ylabel = 'Current (amps)',title='Example 3 Input')
# simulated_results.outputs.plot(ylabel = 'Outputs (voltage)',title='Example 3 Predicted Output')
# simulated_results.event_states.plot(ylabel = 'State of Charge',title='Example 3 Predicted SOC')
# plt.show()