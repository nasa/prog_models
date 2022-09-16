# Copyright © 2022 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
import numpy as np

from prog_models.models import BatteryElectroChemEOD as Battery 
from prog_models.data_models import DMDModel

class TestSerialization(unittest.TestCase):

    def test_serialization_basic(self):
        batt = Battery()

        def future_loading_1(t, x=None):
            # Variable (piece-wise) future loading scheme 
            if (t < 500):
                i = 3
            elif (t < 1000):
                i = 2
            elif (t < 1500):
                i = 0.5
            else:
                i = 4.5
            return batt.InputContainer({'i': i})
        
        def future_loading_2(t, x=None):
            # Variable (piece-wise) future loading scheme 
            if (t < 300):
                i = 2
            elif (t < 800):
                i = 3.5
            elif (t < 1300):
                i = 4
            elif (t < 1600):
                i = 1.5
            else:
                i = 5
            return batt.InputContainer({'i': i})
        
        load_functions = [future_loading_1, future_loading_2]

        options_surrogate = {
            'save_freq': 1, # For DMD, this value is the time step for which the surrogate model is generated
            'dt': 0.1, # For DMD, this value is the time step of the training data
            'trim_data_to': 0.7 # Value between 0 and 1 that determines the fraction of data resulting from simulate_to_threshold that is used to train DMD surrogate model
        }

        batt.parameters['process_noise'] = 0

        # Generate surrogate model  
        surrogate_orig = batt.generate_surrogate(load_functions,**options_surrogate)

        # Serialize parameters
        save_json_dict = surrogate_orig.parameters.to_json()

        # Generate new surrogated with serialized version
        new_model = DMDModel.from_json(save_json_dict)

        # Check serialization 
        self.assertEqual(surrogate_orig.parameters['process_noise'], new_model.parameters['process_noise'])
        self.assertEqual(surrogate_orig.parameters['process_noise_dist'], new_model.parameters['process_noise_dist'])
        self.assertEqual(surrogate_orig.parameters['measurement_noise'], new_model.parameters['measurement_noise'])
        self.assertEqual(surrogate_orig.parameters['measurement_noise_dist'], new_model.parameters['measurement_noise_dist'])
        self.assertEqual(surrogate_orig.parameters['state_keys'], new_model.parameters['state_keys'])
        self.assertEqual(surrogate_orig.parameters['input_keys'], new_model.parameters['input_keys'])
        self.assertEqual(surrogate_orig.parameters['event_keys'], new_model.parameters['event_keys'])
        self.assertEqual(surrogate_orig.parameters['output_keys'], new_model.parameters['output_keys'])
        self.assertEqual(surrogate_orig.parameters['dt'], new_model.parameters['dt'])
        self.assertEqual(surrogate_orig.parameters['trim_data_to'], new_model.parameters['trim_data_to'])
        self.assertEqual(surrogate_orig.parameters['training_noise'], new_model.parameters['training_noise'])
        self.assertEqual(surrogate_orig.parameters['add_dt'], new_model.parameters['add_dt'])
        self.assertEqual(surrogate_orig.parameters['save_freq'], new_model.parameters['save_freq'])
        self.assertEqual(surrogate_orig.parameters['x0'], new_model.parameters['x0'])
        self.assertEqual((surrogate_orig.parameters['dmd_matrix']==new_model.parameters['dmd_matrix']).all(), True)

        # Check deserialization
        options_sim = {
            'save_freq': 1 # Frequency at which results are saved, or equivalently time step in results
        }

        # Define loading profile 
        def future_loading(t, x=None):
            if (t < 600):
                i = 3
            elif (t < 1000):
                i = 2
            elif (t < 1500):
                i = 1.5
            else:
                i = 4
            return batt.InputContainer({'i': i})

        # Simulate to threshold using DMD approximation
        surrogate_results = surrogate_orig.simulate_to_threshold(future_loading,**options_sim)
        new_results = new_model.simulate_to_threshold(future_loading, **options_sim) 

        for i in range(min(len(surrogate_results.times), len(new_results.times))):
            self.assertEqual(surrogate_results.times[i], new_results.times[i])
            self.assertAlmostEqual(surrogate_results.states[i]['tb'], new_results.states[i]['tb'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.states[i]['Vo'], new_results.states[i]['Vo'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.states[i]['Vsn'], new_results.states[i]['Vsn'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.states[i]['Vsp'], new_results.states[i]['Vsp'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.states[i]['qnB'], new_results.states[i]['qnB'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.states[i]['qnS'], new_results.states[i]['qnS'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.states[i]['qpB'], new_results.states[i]['qpB'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.states[i]['qpS'], new_results.states[i]['qpS'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.states[i]['t'], new_results.states[i]['t'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.states[i]['v'], new_results.states[i]['v'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.states[i]['EOD'], new_results.states[i]['EOD'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.inputs[i]['i'], new_results.inputs[i]['i'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.outputs[i]['t'], new_results.outputs[i]['t'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.outputs[i]['v'], new_results.outputs[i]['v'],delta = 3e-01)
            self.assertAlmostEqual(surrogate_results.event_states[i]['EOD'], new_results.event_states[i]['EOD'],delta = 3e-01)

# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Serialization of Surrogate Model")
    result = runner.run(l.loadTestsFromTestCase(TestSerialization)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
