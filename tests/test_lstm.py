# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

import unittest

from prog_models.data_models import LSTMStateTransitionModel, DataModel
from prog_models.models import ThrownObject


class TestLSTM(unittest.TestCase):
    def test_simple_case(self):
        TIMESTEP = 0.01

        # Step 1: Generate data
        m = ThrownObject()

        def future_loading(t, x=None):
            return m.InputContainer({})  # No input for thrown object 

        data = m.simulate_to_threshold(future_loading, threshold_keys='impact', save_freq=TIMESTEP, dt=TIMESTEP)

        # Step 2: Generate model
        m2 = LSTMStateTransitionModel.from_data(
            (data.inputs, data.outputs),  
            window=5, 
            epochs=50,
            outputs = ['x'])  
        self.assertIsInstance(m2, LSTMStateTransitionModel)
        self.assertIsInstance(m2, DataModel)
        self.assertListEqual(m2.outputs, ['x'])
        self.assertListEqual(m2.inputs, ['x_t-1'])
        # Use set below so there's no issue with ordering
        self.assertSetEqual(set(m2.states), set(['x_t-1', 'x_t-2', 'x_t-3', 'x_t-4', 'x_t-5']))
        
        # Step 3: Use model to simulate_to time of threshold
        t_counter = 0
        x_counter = m.initialize()
        def future_loading2(t, x = None):
            # Future Loading is a bit complicated here 
            # Loading for the resulting model includes the data inputs, 
            # and the output from the last timestep
            nonlocal t_counter, x_counter
            z = m.output(x_counter)
            z = m2.InputContainer(z.matrix)
            x_counter = m.next_state(x_counter, future_loading(t), t - t_counter)
            t_counter = t
            return z
        
        results2 = m2.simulate_to(data.times[-1], future_loading2, dt=TIMESTEP, save_freq=TIMESTEP)

        # Have to do it this way because the other way (i.e., using the LSTM model- the states are not a subset)
        # Compare RMSE of the results to the original data
        error = m.calc_error(results2.times, results2.inputs, results2.outputs)
        self.assertLess(error, 2)

        # Create from model
        m3 = LSTMStateTransitionModel(m2.model, outputs = ['x'])

        # More tests in examples.lstm_model

    def test_improper_input(self):
        with self.assertRaises(ValueError, msg='No inputs'):
            LSTMStateTransitionModel.from_data([])
        with self.assertRaises(ValueError, msg="Not iterable"):
            LSTMStateTransitionModel.from_data(12)
        with self.assertRaises(ValueError, msg="Empty Tuple"):
            LSTMStateTransitionModel.from_data(())
        with self.assertRaises(ValueError, msg="Small Tuple"):
            LSTMStateTransitionModel.from_data((1))
        with self.assertRaises(ValueError, msg="Too large Tuple"):
            LSTMStateTransitionModel.from_data((1, 2, 3))
        with self.assertRaises(ValueError, msg="Too small sequence length"):
            LSTMStateTransitionModel.from_data((1, 2), window=0)
        with self.assertRaises(TypeError, msg="Non-scalar sequence length"):
            LSTMStateTransitionModel.from_data((1, 2), window=[])
        with self.assertRaises(ValueError, msg="Too few layers"):
            LSTMStateTransitionModel.from_data((1, 2), layers=0)
        with self.assertRaises(TypeError, msg="Non-scalar layers"):
            LSTMStateTransitionModel.from_data((1, 2), layers=[])
        with self.assertRaises(ValueError, msg="Too few epochs"):
            LSTMStateTransitionModel.from_data((1, 2), epochs=0)
        with self.assertRaises(TypeError, msg="Non-scalar epochs"):
            LSTMStateTransitionModel.from_data((1, 2), epochs=[])
        with self.assertRaises(IndexError, msg="Uneven Inputs/outputs"):
            LSTMStateTransitionModel.from_data(([1], [1, 2]))
        with self.assertRaises(IndexError, msg="Uneven Inputs/outputs-2"):
            LSTMStateTransitionModel.from_data(([1, 2, 3], [1, 2]))
        with self.assertRaises(TypeError, msg="Element Format"):
            LSTMStateTransitionModel.from_data((1, [2]))
        with self.assertRaises(TypeError, msg="Element Format-2"):
            LSTMStateTransitionModel.from_data(([1], 2))
        with self.assertRaises(TypeError, msg="Normalize type"):
            LSTMStateTransitionModel.from_data(([1], [2]), normalize=[1, 2])
        with self.assertRaises(ValueError, msg="Negative dropout"):
            LSTMStateTransitionModel.from_data(([1], [2]), dropout = -1)   
        with self.assertRaises(ValueError, msg="Zero units"):
            LSTMStateTransitionModel.from_data(([1], [2]), units = 0)  
        with self.assertRaises(ValueError, msg="Units, layers mismatch"):
            LSTMStateTransitionModel.from_data(([1], [2]), units = [1], layers=2)  

# This allows the module to be executed directly
def run_tests():
    unittest.main()

def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting LSTM")
    result = runner.run(l.loadTestsFromTestCase(TestLSTM)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
