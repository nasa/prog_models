# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

import matplotlib.pyplot as plt
import numpy as np

from prog_models.lstm_model import LSTMStateTransitionModel
from prog_models.models import BatteryElectroChemEOD

def run_example():
        # -----------------------------------------------------
    # Example 3- More complicated system
    # Here we will create a model for a more complicated system
    # For this example we will use the BatteryCircuit model
    # NOTE: THIS EXAMPLE CURRENTLY DOESN'T PERFORM WELL, but it is 
    #    kept here as a goal for the future
    # -----------------------------------------------------
    print('\n------------------------------------------\nExample 3...')
    print('Generating data...')
    batt = BatteryElectroChemEOD()
    future_loading_eqns = [lambda t, x=None: batt.InputContainer({'i': 1+1.5*load}) for load in range(6)]
    # Generate data with different loading and step sizes
    # Adding the step size as an element of the output
    training_data = []
    for i in range(9):
        dt = i/3+0.25
        for loading_eqn in future_loading_eqns:
            d = batt.simulate_to_threshold(loading_eqn, save_freq=dt, dt=dt) 
            u = np.array([np.hstack((u_i.matrix[:][0].T, [dt])) for u_i in d.inputs], dtype=float)
            z = d.outputs
            training_data.append((u, z))
  
    # Step 2: Generate Model
    print('Building model...')
    m_batt = LSTMStateTransitionModel.from_data(
        training_data,  
        sequence_length=24, 
        epochs=50, 
        layers=1,
        units=64,
        inputs = ['i', 'dt'],
        outputs = ['t', 'v'])    

    # Step 3: Simulate with model
    t_counter = 0
    x_counter = batt.initialize()

    def future_loading(t, x=None):
        return batt.InputContainer({'i': 3})

    def future_loading2(t, x = None):
        nonlocal t_counter, x_counter
        z = batt.output(x_counter)
        z = m_batt.InputContainer({'i': 3, 't_t-1': z['t'], 'v_t-1': z['v'], 'dt': t - t_counter})
        x_counter = batt.next_state(x_counter, future_loading(t), t - t_counter)
        t_counter = t
        return z

    # Use new dt, not used in training
    data = batt.simulate_to_threshold(future_loading, dt=1, save_freq=1)
    results = m_batt.simulate_to(data.times[-1], future_loading2, dt=1, save_freq=1)

    # Step 5: Compare Results
    print('Comparing results...')
    data.outputs.plot(title='original model', compact=False)
    results.outputs.plot(title='generated model', compact=False)
    plt.show()

if __name__ == '__main__':
    run_example()
