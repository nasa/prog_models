import matplotlib.pyplot as plt
import warnings
from prog_models.models import ThrownObject, BatteryElectroChemEOD

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()



def future_load(t=None, x=None):  
    # The thrown object model has no inputs- you cannot load the system (i.e., affect it once it's in the air)
    # So we return an empty input container
    return m.InputContainer({})

# Define configuration for simulation
config = {
    'threshold_keys': 'impact', # Simulate until the thrown object has impacted the ground
    'dt': 0.005, # Time step (s)
    'save_freq': 0.5, # Frequency at which results are saved (s)
}

# Define a function to print the results - will be used later
def print_results(simulated_results):
    # Print results
    print('states:')
    for (t,x) in zip(simulated_results.times, simulated_results.states):
        print('\t{:.2f}s: {}'.format(t, x))

    print('outputs:')
    for (t,x) in zip(simulated_results.times, simulated_results.outputs):
        print('\t{:.2f}s: {}'.format(t, x))

    print('\nimpact time: {:.2f}s'.format(simulated_results.times[-1]))
    # The simulation stopped at impact, so the last element of times is the impact time

    # Plot results
    simulated_results.states.plot()


process_noise_dist = 'triangular'
process_noise = {'tb': 30, 'Vo': 15}
model_config = {'process_noise_dist': process_noise_dist, 'process_noise': process_noise}
m = BatteryElectroChemEOD(**model_config)

options = {
    'save_freq': 200, # Frequency at which results are saved
    'dt': 1, # Time step
}

def future_loading_BATT(t, x=None):
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
    return m.InputContainer({'i': i})

simulated_results = m.simulate_to(2000, future_loading_BATT, **options)
print('\nExample with triangular process noise')
print_results(simulated_results)
plt.title('Ex4: Ex3 with triangular dist')
