# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

import unittest

from prog_models.data_models import LSTMStateTransitionModel, DataModel, DMDModel
from prog_models.models import ThrownObject


class TestDataModel(unittest.TestCase):
    def _test_simple_case(self, 
        DataModelType, 
        m = ThrownObject(), 
        max_error=2, 
        TIMESTEP = 0.01, 
        WITH_STATES = True, 
        **kwargs):

        # Step 1: Generate data
        def future_loading(t, x=None):
            return m.InputContainer({})  # No input for thrown object 

        data = m.simulate_to_threshold(future_loading, threshold_keys='impact', save_freq=TIMESTEP, dt=TIMESTEP)

        if WITH_STATES:
            kwargs['states'] = [data.states]

        # Step 2: Generate model
        m2 = DataModelType.from_data(
            times = [data.times],
            inputs = [data.inputs],
            outputs = [data.outputs],
            event_states = [data.event_states],  
            output_keys = list(m.outputs),
            dt = TIMESTEP,
            save_freq = TIMESTEP,
            **kwargs)  
        self.assertIsInstance(m2, DataModelType)
        self.assertIsInstance(m2, DataModel)
        self.assertListEqual(m2.outputs, list(m.outputs))
        
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
        self.assertLess(error, max_error)

        return m2

    def test_lstm_simple(self):
        m = self._test_simple_case(LSTMStateTransitionModel, window=5, epochs=20)
        self.assertListEqual(m.inputs, ['x_t-1'])
        # Use set below so there's no issue with ordering
        self.assertSetEqual(set(m.states), set(['x_t-1', 'x_t-2', 'x_t-3', 'x_t-4', 'x_t-5']))

        # Create from model
        LSTMStateTransitionModel(m.model, output_keys = ['x'])

        # More tests in examples.lstm_model

    def test_dmd_simple(self):
        self._test_simple_case(DMDModel, max_error=4)

        # Without velocity, DMD doesn't perform well
        self._test_simple_case(DMDModel, WITH_STATES = False, max_error=100)

    def test_lstm_from_model_thrown_object(self):
        TIMESTEP = 0.01

        # Step 1: Generate dataed
        m = ThrownObject()

        def future_loading(t, x=None):
            return m.InputContainer({})  # No input for thrown object 
        m3 = LSTMStateTransitionModel.from_model(
            m,
            [future_loading for _ in range(5)],
            dt = [TIMESTEP, TIMESTEP/2, TIMESTEP/4, TIMESTEP*2, TIMESTEP*4],
            window=2, 
            epochs=20)  

        # Should get keys from original model
        self.assertSetEqual(set(m3.inputs), set(['dt', 'x_t-1']))
        self.assertSetEqual(set(m3.outputs), set(m.outputs))

         # Step 3: Use model to simulate_to time of threshold
        t_counter = 0
        x_counter = m.initialize()
        def future_loading2(t, x = None):
            # Future Loading is a bit complicated here 
            # Loading for the resulting model includes the data inputs, 
            # and the output from the last timestep
            nonlocal t_counter, x_counter
            z = m.output(x_counter)
            z = m3.InputContainer(
                {
                    'x_t-1': z['x'],
                    'dt': TIMESTEP
                })
            x_counter = m.next_state(x_counter, future_loading(t), t - t_counter)
            t_counter = t
            return z
        
        results2 = m3.simulate_to(7.5, future_loading2, dt=TIMESTEP, save_freq=TIMESTEP)

        # Have to do it this way because the other way (i.e., using the LSTM model- the states are not a subset)
        # Compare RMSE of the results to the original data
        error = m.calc_error(results2.times, results2.inputs, results2.outputs)
        self.assertLess(error, 2)

        # Now with no 'dt'
        #----------------------------------------------------
        m3 = LSTMStateTransitionModel.from_model(
            m,
            [future_loading for _ in range(5)],
            dt = [TIMESTEP, TIMESTEP/2, TIMESTEP/4, TIMESTEP*2, TIMESTEP*4],
            window=2, 
            epochs=30,
            add_dt = False)  

        # Should get keys from original model
        self.assertSetEqual(set(m3.inputs), set(['x_t-1']))
        self.assertSetEqual(set(m3.outputs), set(m.outputs))

         # Step 3: Use model to simulate_to time of threshold
        t_counter = 0
        x_counter = m.initialize()
        def future_loading2(t, x = None):
            # Future Loading is a bit complicated here 
            # Loading for the resulting model includes the data inputs, 
            # and the output from the last timestep
            nonlocal t_counter, x_counter
            z = m.output(x_counter)
            z = m3.InputContainer(
                {
                    'x_t-1': z['x']
                })
            x_counter = m.next_state(x_counter, future_loading(t), t - t_counter)
            t_counter = t
            return z
        
        results2 = m3.simulate_to(7.5, future_loading2, dt=TIMESTEP, save_freq=TIMESTEP)

        # Have to do it this way because the other way (i.e., using the LSTM model- the states are not a subset)
        # Compare RMSE of the results to the original data
        error = m.calc_error(results2.times, results2.inputs, results2.outputs)
        self.assertLess(error, 2)

    def test_improper_input(self):
        # Input, output format
        with self.assertRaises(Exception, msg='No outputs'):
            LSTMStateTransitionModel.from_data(inputs=[1])
        with self.assertRaises(Exception, msg='No inputs'):
            LSTMStateTransitionModel.from_data(outputs=[1])
        with self.assertRaises(ValueError, msg='Empty inputs/outputs'):
            LSTMStateTransitionModel.from_data(inputs=[], outputs=[])
        with self.assertRaises(ValueError, msg="Uneven Inputs/outputs"):
            LSTMStateTransitionModel.from_data([1], [1, 2])
        with self.assertRaises(ValueError, msg="Uneven Inputs/outputs-2"):
            LSTMStateTransitionModel.from_data([1, 2, 3], [1, 2])
        with self.assertRaises(TypeError, msg="Element Format"):
            LSTMStateTransitionModel.from_data(1, [2])
        with self.assertRaises(TypeError, msg="Element Format-2"):
            LSTMStateTransitionModel.from_data([1], 2)
        with self.assertRaises(TypeError, msg="Not iterable"):
            LSTMStateTransitionModel.from_data(12, 12)

        # Other Configurables
        with self.assertRaises(ValueError, msg="Too small sequence length"):
            LSTMStateTransitionModel.from_data([1], [2], window=0)
        with self.assertRaises(TypeError, msg="Non-scalar sequence length"):
            LSTMStateTransitionModel.from_data([1], [2], window=[])

        with self.assertRaises(ValueError, msg="Too few layers"):
            LSTMStateTransitionModel.from_data([1], [2], layers=0)
        with self.assertRaises(TypeError, msg="Non-scalar layers"):
            LSTMStateTransitionModel.from_data([1], [2], layers=[])

        with self.assertRaises(ValueError, msg="Too few epochs"):
            LSTMStateTransitionModel.from_data([1], [2], epochs=0)
        with self.assertRaises(TypeError, msg="Non-scalar epochs"):
            LSTMStateTransitionModel.from_data([1], [2], epochs=[])

        with self.assertRaises(TypeError, msg="Normalize type"):
            LSTMStateTransitionModel.from_data([1], [2], normalize=[1, 2])

        with self.assertRaises(TypeError, msg="Dropout type"):
            LSTMStateTransitionModel.from_data([1], [2], dropout = 'abc')   
        with self.assertRaises(ValueError, msg="Negative dropout"):
            LSTMStateTransitionModel.from_data([1], [2], dropout = -1)   

        with self.assertRaises(ValueError, msg="Zero units"):
            LSTMStateTransitionModel.from_data([1], [2], units = 0)  

        with self.assertRaises(ValueError, msg="Units, layers mismatch"):
            LSTMStateTransitionModel.from_data([1], [2], units = [1], layers=2)  
        with self.assertRaises(ValueError, msg="Units, layers mismatch-2"):
            LSTMStateTransitionModel.from_data([1], [2], units = [1, 2, 3], layers=2)     

        with self.assertRaises(TypeError, msg="Validation Split type"):
            LSTMStateTransitionModel.from_data([1], [2], validation_split = 'abc')   
        with self.assertRaises(ValueError, msg="Negative Validation Split"):
            LSTMStateTransitionModel.from_data([1], [2], validation_split = -1e-5)    
        with self.assertRaises(ValueError, msg="Validation Split of 1"):
            LSTMStateTransitionModel.from_data([1], [2], validation_split = 1.0)    

# This allows the module to be executed directly
def run_tests():
    unittest.main()

def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Data Models")
    result = runner.run(l.loadTestsFromTestCase(TestDataModel)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
