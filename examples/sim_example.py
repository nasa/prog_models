# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
"""
An example where a battery is simulated first for a set period of time and then till threshold is met. Run using the command `python -m examples.sim_example`
"""

from prog_models.models import battery_circuit, battery_electrochem

def run_example(): 
    # Step 1: Create a model object
    batt = battery_circuit.BatteryCircuit()
    # batt = battery_electrochem.BatteryElectroChem() # Uncomment this to use Electro Chemistry Model

    # Step 2: Define future loading function 
    def future_loading(t):
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
    (times, inputs, states, outputs, event_states) = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})

    for i in range(len(times)): # Print Results
        print("Time: {}\n\tInput: {}\n\tState: {}\n\tOutput: {}\n\tEvent State: {}\n".format(times[i], inputs[i], states[i], outputs[i], event_states[i]))

    # Simulate to threshold
    print('\n\n------------------------------------------------')
    print('Simulating to threshold\n\n')
    options = {
        'save_freq': 100, # Frequency at which results are saved
        'dt': 2 # Timestep
    }
    (times, inputs, states, outputs, event_states) = batt.simulate_to_threshold(future_loading, {'t': 18.95, 'v': 4.183}, options)

    for i in range(len(times)): # Print Results
        print("Time: {}\n\tInput: {}\n\tState: {}\n\tOutput: {}\n\tEvent State: {}\n".format(times[i], inputs[i], states[i], outputs[i], event_states[i]))

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()