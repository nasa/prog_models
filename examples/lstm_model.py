# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

"""
Example building a LSTMStateTransitionModel from data. This is a simple example of how to use the LSTMStateTransitionModel class.

In this example, we generate fake data using the ThrownObject model. This is a case where we're generating a surrogate model from the physics-based model. For cases where you're generating a model from data (e.g., collected from a testbed or a real-world environment), you'll replace that generated data with your own. We then use the generated model and compare to the original model.
"""

from prog_models.lstm_model import LSTMStateTransitionModel
from prog_models.models import ThrownObject

def run_example():
    # Step 1: Generate data
    # We'll use the ThrownObject model to generate data.
    # For cases where you're generating a model from data (e.g., collected from a testbed or a real-world environment), 
    # you'll replace that generated data with your own.
    print('Generating data')
    m = ThrownObject()

    def future_loading(t, x=None):
        return m.InputContainer({})  # No input for thrown object 

    data = m.simulate_to_threshold(future_loading, threshold_keys='impact', save_freq=0.01, dt=0.01)

    # Step 2: Generate model
    # We'll use the LSTMStateTransitionModel class to generate a model from the data.
    print('Building model...')
    m2 = LSTMStateTransitionModel.from_data((data.inputs, data.outputs),  
    sequence_length=4, epochs=5)    
    
    # Step 3: Use model to simulate_to time of threshold
    print('Simulating with generated model...')
    t_counter = 0
    x_counter = m.initialize()
    def future_loading2(t, x = None):
        nonlocal t_counter, x_counter
        z = m.output(x_counter)
        z = {'z0_t-1': z['x']}
        z = m2.InputContainer(z)
        x_counter = m.next_state(x_counter, future_loading(t), t - t_counter)
        t_counter = t
        return z
    m2.simulate_to(2, future_loading2, dt=0.01, save_freq = 0.01, print=True)

    # Step 4: Compare model to original model
    print('Comparing results...')

if __name__ == '__main__':
    run_example()
