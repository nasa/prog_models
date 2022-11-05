# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

"""
Example building a Data Models from data. This is a simple example of how to use the LSTMStateTransitionModel and FNNStateTransitionModel classes.

.. dropdown:: More details

    In this example, we generate fake data using the ThrownObject model. This is a case where we're generating a surrogate model from the physics-based model. For cases where you're generating a model from data (e.g., collected from a testbed or a real-world environment), you'll replace that generated data with your own. We then use the generated model and compare to the original model.

    Finally, we repeat the exercise with data from the more complex BatteryElectroChemEOD model.
"""

import matplotlib.pyplot as plt
import numpy as np
from prog_models.data_models import FNNStateTransitionModel, LSTMStateTransitionModel
from prog_models.models import ThrownObject, BatteryElectroChemEOD

def run_example():
    # -----------------------------------------------------
    # Example 1- set timestep 
    # Here we will create a model for a specific timestep.
    # The model will only work with that timestep 
    # This is useful if you know the timestep you would like to use
    # -----------------------------------------------------
    TIMESTEP = 0.01

    # Step 1: Generate data
    # We'll use the ThrownObject model to generate data.
    # For cases where you're generating a model from data (e.g., collected from a testbed or a real-world environment), 
    # you'll replace that generated data with your own.
    print('Generating data')
    m = ThrownObject()

    def future_loading(t, x=None):
        return m.InputContainer({})  # No input for thrown object 

    data = m.simulate_to_threshold(future_loading, threshold_keys='impact', save_freq=TIMESTEP, dt=TIMESTEP)

    # Step 2: Generate model
    # We'll use the LSTMStateTransitionModel class to generate a model from the data.
    print('Building model...')
    cfg = {  # Configuration for training
        'inputs': [data.inputs],
        'outputs': [data.outputs],
        'window': 4,
        'epochs': 50,  # Maximum number of epochs, may stop earlier if early stopping enabled
        'output_keys': ['x']
    }
    lstm_model = LSTMStateTransitionModel.from_data(**cfg)
    fnn_model = FNNStateTransitionModel.from_data(**cfg, layers=4, units=[4086, 10000, 1024, 512], activation='sigmoid')

    # We can see the training history
    # Should show the model progressively getting better (i.e., the loss going down).
    # If val_loss starts going up again, then we may be overtraining
    lstm_model.plot_history()
    fnn_model.plot_history()
    plt.show()
    
    # Step 3: Use model to simulate_to time of threshold
    print('Simulating with generated model...')

    t_counter = 0
    x_counter = m.initialize()
    def future_loading2(t, x = None):
        # Future Loading is a bit complicated here 
        # Loading for the resulting model includes the data inputs, 
        # and the output from the last timestep
        nonlocal t_counter, x_counter
        z = m.output(x_counter)
        z = lstm_model.InputContainer(z.matrix)
        x_counter = m.next_state(x_counter, future_loading(t), t - t_counter)
        t_counter = t
        return z
    
    lstm_results = lstm_model.simulate_to(data.times[-1], future_loading2, dt=TIMESTEP, save_freq=TIMESTEP)
    fnn_results = fnn_model.simulate_to(data.times[-1], future_loading2, dt=TIMESTEP, save_freq=TIMESTEP)

    # Step 4: Compare model to original model
    print('Comparing results...')
    data.outputs.plot(title='original model')
    lstm_results.outputs.plot(title='LSTM generated model')
    fnn_results.outputs.plot(title='FNN generated model')
    plt.show()

    # -----------------------------------------------------
    # Example 2- variable timestep 
    # Here we will create a model to work with any timestep
    # We do this by adding timestep as a variable in the model
    # -----------------------------------------------------

    # Step 1: Generate additional data
    # We will use data generated above, but we also want data at additional timesteps 
    print('\n------------------------------------------\nExample 2...')
    print('Generating additional data...')
    data_half = m.simulate_to_threshold(future_loading, threshold_keys='impact', save_freq=TIMESTEP/2, dt=TIMESTEP/2)
    data_quarter = m.simulate_to_threshold(future_loading, threshold_keys='impact', save_freq=TIMESTEP/4, dt=TIMESTEP/4)
    data_twice = m.simulate_to_threshold(future_loading, threshold_keys='impact', save_freq=TIMESTEP*2, dt=TIMESTEP*2)
    data_four = m.simulate_to_threshold(future_loading, threshold_keys='impact', save_freq=TIMESTEP*4, dt=TIMESTEP*4)

    # Step 2: Data Prep
    # We need to add the timestep as a input
    u = np.array([[TIMESTEP] for _ in data.inputs])
    u_half = np.array([[TIMESTEP/2] for _ in data_half.inputs])
    u_quarter = np.array([[TIMESTEP/4] for _ in data_quarter.inputs])
    u_twice = np.array([[TIMESTEP*2] for _ in data_twice.inputs])
    u_four = np.array([[TIMESTEP*4] for _ in data_four.inputs])

    input_data = [u, u_half, u_quarter, u_twice, u_four]
    output_data = [data.outputs, data_half.outputs, data_quarter.outputs, data_twice.outputs, data_four.outputs]

    # Step 3: Generate Model
    print('Building model...')
    cfg = {
        'window': 4,
        'epochs': 50,
        'input_keys': ['dt'],
        'output_keys': ['x']
    }
    lstm_model = LSTMStateTransitionModel.from_data(
        inputs = input_data,  
        outputs = output_data,
        **cfg) 
    # Note, since we're generating from a model, we could also have done this:
    # lstm_model = LSTMStateTransitionModel.from_model(
    #     m,
    #     [future_loading for _ in range(5)],
    #     dt = [TIMESTEP, TIMESTEP/2, TIMESTEP/4, TIMESTEP*2, TIMESTEP*4],
    #     **cfg)  

    fnn_model = FNNStateTransitionModel.from_data(
        inputs = input_data,  
        outputs = output_data,
        units = 128,
        **cfg) 

    # Take a look at the training history
    lstm_model.plot_history()
    plt.show() 

    # Step 4: Simulate with model
    t_counter = 0
    x_counter = m.initialize()
    def future_loading3(t, x = None):
        nonlocal t_counter, x_counter
        z = lstm_model.InputContainer({'x_t-1': x_counter['x'], 'dt': t - t_counter})
        x_counter = m.next_state(x_counter, future_loading(t), t - t_counter)
        t_counter = t
        return z

    # Use new dt, not used in training
    # Using a dt not used in training will demonstrate the model's 
    # ability to handle different timesteps not part of training set
    data = m.simulate_to(data.times[-1], future_loading, dt=TIMESTEP*3, save_freq=TIMESTEP*3)
    lstm_results = lstm_model.simulate_to(data.times[-1], future_loading3, dt=TIMESTEP*3, save_freq=TIMESTEP*3)
    fnn_results = fnn_model.simulate_to(data.times[-1], future_loading3, dt=TIMESTEP*3, save_freq=TIMESTEP*3)

    # Step 5: Compare Results
    print('Comparing results...')
    data.outputs.plot(title='original model')
    lstm_results.outputs.plot(title='lstm model')
    fnn_results.outputs.plot(title='fnn model')
    plt.show()

    # -----------------------------------------------------
    # Example 3- More complicated system
    # Here we will create a model for a more complicated system
    # For this example we will use the BatteryElectroChemEOD model
    # We also include the event state (SOC)
    # -----------------------------------------------------
    print('\n------------------------------------------\nExample 3...')
    print('Generating data...')
    batt = BatteryElectroChemEOD(process_noise = 0, measurement_noise=0)
    future_loading_eqns = [lambda t, x=None, load=load: batt.InputContainer({'i': 1+1.5*load}) for load in range(6)]
    # Generate data with different loading and step sizes
    # Adding the step size as an element of the output
    input_data = []
    output_data = []
    es_data = []
    t_met_data = []
    for i in range(9):
        dt = i/3+0.25
        for loading_eqn in future_loading_eqns:
            d = batt.simulate_to_threshold(loading_eqn, save_freq=dt, dt=dt) 
            input_data.append(np.array([np.hstack((u_i.matrix[:][0].T, [dt])) for u_i in d.inputs], dtype=float))
            output_data.append(d.outputs)
            es_data.append(d.event_states)
            t_met = [[False]for _ in d.times]
            t_met[-1][0] = True  # Threshold has been met at the last timestep
            t_met_data.append(t_met)
  
    # Step 2: Generate Model
    print('Building model...')
    cfg = {
        'inputs': input_data,
        'outputs': output_data,
        'event_states': es_data,
        't_met': t_met_data,
        'window': 12,
        'epochs': 50,
        'input_keys': ['i', 'dt'],
        'output_keys': ['t', 'v'],
        'event_keys':['EOD']
    }
    m_batt_lstm = LSTMStateTransitionModel.from_data(
        units=64,  # Additional units given the increased complexity of the system
        **cfg)
        
    m_batt_fnn = FNNStateTransitionModel.from_data(
        units=128,
        **cfg) 

    # Take a look at the training history.
    m_batt_lstm.plot_history()
    plt.show()

    # Step 3: Simulate with model
    t_counter = 0
    x_counter = batt.initialize()

    def future_loading(t, x=None):
        return batt.InputContainer({'i': 3})

    def future_loading2(t, x = None):
        nonlocal t_counter, x_counter
        z = batt.output(x_counter)
        z = m_batt_lstm.InputContainer({'i': 3, 't_t-1': z['t'], 'v_t-1': z['v'], 'dt': t - t_counter})
        x_counter = batt.next_state(x_counter, future_loading(t), t - t_counter)
        t_counter = t
        return z

    # Use a new dt, not used in training. 
    # Using a dt not used in training will demonstrate the model's 
    # ability to handle different timesteps not part of training set
    data = batt.simulate_to_threshold(future_loading, dt=1, save_freq=1)
    results = m_batt_lstm.simulate_to_threshold(future_loading2, dt=1, save_freq=1)
    fnn_results = m_batt_fnn.simulate_to_threshold(future_loading2, dt=1, save_freq=1)

    # Step 5: Compare Results
    print('Comparing results...')
    data.outputs.plot(title='original model', compact=False)
    results.outputs.plot(title='lstm model', compact=False)
    fnn_results.outputs.plot(title='fnn model', compact=False)
    data.event_states.plot(title='original model', compact=False)
    results.event_states.plot(title='lstm model', compact=False)
    fnn_results.event_states.plot(title='fnn model', compact=False)
    plt.show()

    # This last example isn't a perfect fit, but it matches the behavior pretty well
    # Especially the voltage curve

if __name__ == '__main__':
    run_example()
