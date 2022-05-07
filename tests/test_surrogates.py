# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
import warnings

from prog_models.models import *
from prog_models.exceptions import ProgModelInputException


class TestSurrogate(unittest.TestCase):
    def test_surrogate_improper_input(self):
        m = ThrownObject()
        def load_eqn(t = None, x = None):
            return m.InputContainer({})
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate(None)
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([])
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], method = 'invalid')
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], save_pts = [10])
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], trim_data_to = -1)
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], trim_data_to = 'invalid')
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], trim_data_to = 1.5)
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], states = ['invalid'])
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], states = ['x', 'invalid', 'v'])
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], inputs = ['invalid'])
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], outputs = ['invalid'])
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], outputs = ['invalid', 'x'])    
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], events = ['invalid'])
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], events = ['falling', 'impact', 'invalid'])
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], stability_tol = -1)
        with self.assertRaises(ProgModelInputException):
            m.generate_surrogate([load_eqn], stability_tol = 'invalid')
    
    def test_surrogate_basic_thrown_object(self):
        m = ThrownObject()
        def load_eqn(t = None, x = None):
            return m.InputContainer({})
        
        surrogate = m.generate_surrogate([load_eqn], dt = 0.1, save_freq = 0.25, threshold_keys = 'impact')
        self.assertEqual(surrogate.dt, 0.25)

        self.assertListEqual(surrogate.states, m.states + m.outputs + m.events)
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, m.outputs)
        self.assertListEqual(surrogate.events, m.events)

        options = {
            'threshold_keys': 'impact',
            'save_freq': 0.25,
            'dt': 0.25
        }
        result = m.simulate_to_threshold(load_eqn, **options)
        surrogate_results = surrogate.simulate_to_threshold(load_eqn, **options)

        MSE = m.calc_error(surrogate_results.times, surrogate_results.inputs, surrogate_results.outputs)
        self.assertLess(MSE, 10) # Pretty good approx

        self.assertAlmostEqual(surrogate_results.times[-1], result.times[-1], delta=0.26)
        for i in range(min(len(result.times), len(surrogate_results.times))):
            self.assertListEqual(list(surrogate_results.inputs[i].keys()), list(surrogate_results.inputs[i].keys()))
            self.assertAlmostEqual(surrogate_results.states[i]['x'], result.states[i]['x'], delta=8)
            self.assertEqual(surrogate_results.states[i]['x'], surrogate_results.outputs[i]['x'])
            self.assertAlmostEqual(surrogate_results.states[i]['v'], result.states[i]['v'], delta=1)
            self.assertAlmostEqual(surrogate_results.states[i]['falling'], result.event_states[i]['falling'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['falling'], surrogate_results.event_states[i]['falling'])
            self.assertAlmostEqual(surrogate_results.states[i]['impact'], result.event_states[i]['impact'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['impact'], surrogate_results.event_states[i]['impact'])

    def test_surrogate_basic_battery(self):
        m = BatteryElectroChem(process_noise = 0, measurement_noise = 0)
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
            return m.InputContainer({'i': i})
        
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
            return m.InputContainer({'i': i})
        load_functions = [future_loading_1, future_loading_2]

        options_surrogate = {
            'save_freq': 1, # For DMD, this value is the time step for which the surrogate model is generated
            'dt': 0.1, # For DMD, this value is the time step of the training data
            'trim_data_to': 0.7, # Value between 0 and 1 that determines the fraction of data resulting from simulate_to_threshold that is used to train DMD surrogate model
        }

        surrogate = m.generate_surrogate(load_functions,**options_surrogate)
        self.assertListEqual(surrogate.states, m.states + m.outputs + m.events)
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, m.outputs)
        self.assertListEqual(surrogate.events, m.events)

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
            return m.InputContainer({'i': i})

        result = m.simulate_to_threshold(future_loading, **options_sim)
        surrogate_results = surrogate.simulate_to_threshold(future_loading,**options_sim)

        self.assertAlmostEqual(surrogate_results.times[-1], result.times[-1], delta=50)
        MSE = m.calc_error(surrogate_results.times, surrogate_results.inputs, surrogate_results.outputs)
        self.assertLess(MSE, 300000)

        # Intermediate states dont match very well - skip tests
    
    def test_surrogate_subsets(self):
        m = ThrownObject()
        def load_eqn(t = None, x = None):
            return m.InputContainer({})

        # Perfect subset
        surrogate = m.generate_surrogate([load_eqn], dt = 0.1, save_freq = 0.25, threshold_keys = 'impact', states=['x', 'v'])
        self.assertEqual(surrogate.dt, 0.25)

        self.assertListEqual(surrogate.states, m.states + m.outputs + m.events + m.inputs)
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, m.outputs)
        self.assertListEqual(surrogate.events, m.events)

        options = {
            'threshold_keys': 'impact',
            'save_freq': 0.25,
            'dt': 0.25
        }
        result = m.simulate_to_threshold(load_eqn, **options)
        surrogate_results = surrogate.simulate_to_threshold(load_eqn, **options)
        self.assertAlmostEqual(surrogate_results.times[-1], result.times[-1], delta=0.26)
        for i in range(min(len(result.times), len(surrogate_results.times))):
            self.assertListEqual(list(surrogate_results.inputs[i].keys()), list(surrogate_results.inputs[i].keys()))
            self.assertAlmostEqual(surrogate_results.states[i]['x'], result.states[i]['x'], delta=8)
            self.assertEqual(surrogate_results.states[i]['x'], surrogate_results.outputs[i]['x'])
            self.assertAlmostEqual(surrogate_results.states[i]['v'], result.states[i]['v'], delta=1)
            self.assertAlmostEqual(surrogate_results.states[i]['falling'], result.event_states[i]['falling'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['falling'], surrogate_results.event_states[i]['falling'])
            self.assertAlmostEqual(surrogate_results.states[i]['impact'], result.event_states[i]['impact'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['impact'], surrogate_results.event_states[i]['impact'])
        
        # State subset
        surrogate = m.generate_surrogate([load_eqn], dt = 0.1, save_freq = 0.25, threshold_keys = 'impact', states = ['v'])
        self.assertListEqual(surrogate.states, ['v'] + m.outputs + m.events + m.inputs)
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, m.outputs)
        self.assertListEqual(surrogate.events, m.events)

        surrogate = m.generate_surrogate([load_eqn], dt = 0.1, save_freq = 0.25, threshold_keys = 'impact', states = 'v')
        self.assertListEqual(surrogate.states, ['v'] + m.outputs + m.events + m.inputs)
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, m.outputs)
        self.assertListEqual(surrogate.events, m.events)

        options = {
            'threshold_keys': 'impact',
            'save_freq': 0.25,
            'dt': 0.25
        }
        result = m.simulate_to_threshold(load_eqn, **options)
        surrogate_results = surrogate.simulate_to_threshold(load_eqn, **options)
        self.assertAlmostEqual(surrogate_results.times[-1], result.times[-1], delta=0.26)
        for i in range(min(len(result.times), len(surrogate_results.times))):
            self.assertListEqual(list(surrogate_results.inputs[i].keys()), list(surrogate_results.inputs[i].keys()))
            self.assertAlmostEqual(surrogate_results.states[i]['x'], result.states[i]['x'], delta=8)
            self.assertEqual(surrogate_results.states[i]['x'], surrogate_results.outputs[i]['x'])
            self.assertAlmostEqual(surrogate_results.states[i]['v'], result.states[i]['v'], delta=1)
            self.assertAlmostEqual(surrogate_results.states[i]['falling'], result.event_states[i]['falling'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['falling'], surrogate_results.event_states[i]['falling'])
            self.assertAlmostEqual(surrogate_results.states[i]['impact'], result.event_states[i]['impact'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['impact'], surrogate_results.event_states[i]['impact'])

        # Events subset
        surrogate = m.generate_surrogate([load_eqn], dt = 0.1, save_freq = 0.25,      threshold_keys = 'impact', events = ['impact'])
        surrogate = m.generate_surrogate([load_eqn], dt = 0.1, save_freq = 0.25,      threshold_keys = 'impact', events = 'impact')
        self.assertListEqual(surrogate.states, m.states + m.outputs + ['impact'] + m.inputs)
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, m.outputs)
        self.assertListEqual(surrogate.events, ['impact'])

        options = {
            'threshold_keys': 'impact',
            'save_freq': 0.25,
            'dt': 0.25
        }
        result = m.simulate_to_threshold(load_eqn, **options)
        surrogate_results = surrogate.simulate_to_threshold(load_eqn, **options)
        self.assertAlmostEqual(surrogate_results.times[-1], result.times[-1], delta=0.26)
        for i in range(min(len(result.times), len(surrogate_results.times))):
            self.assertListEqual(list(surrogate_results.inputs[i].keys()), list(surrogate_results.inputs[i].keys()))
            self.assertAlmostEqual(surrogate_results.states[i]['x'], result.states[i]['x'], delta=8)
            self.assertEqual(surrogate_results.states[i]['x'], surrogate_results.outputs[i]['x'])
            self.assertAlmostEqual(surrogate_results.states[i]['v'], result.states[i]['v'], delta=1.25)
            self.assertAlmostEqual(surrogate_results.states[i]['impact'], result.event_states[i]['impact'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['impact'], surrogate_results.event_states[i]['impact'])

        # Outputs - Empty
        surrogate = m.generate_surrogate([load_eqn], dt = 0.1, save_freq = 0.25,      threshold_keys = 'impact', outputs = [])
        self.assertListEqual(surrogate.states, m.states + m.events + m.inputs)
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, [])
        self.assertListEqual(surrogate.events, m.events)

        options = {
            'threshold_keys': 'impact',
            'save_freq': 0.25,
            'dt': 0.25
        }
        result = m.simulate_to_threshold(load_eqn, **options)
        surrogate_results = surrogate.simulate_to_threshold(load_eqn, **options)
        self.assertAlmostEqual(surrogate_results.times[-1], result.times[-1], delta=0.26)
        for i in range(min(len(result.times), len(surrogate_results.times))):
            self.assertListEqual(list(surrogate_results.inputs[i].keys()), list(surrogate_results.inputs[i].keys()))
            self.assertAlmostEqual(surrogate_results.states[i]['x'], result.states[i]['x'], delta=8)
            self.assertAlmostEqual(surrogate_results.states[i]['v'], result.states[i]['v'], delta=1)
            self.assertAlmostEqual(surrogate_results.states[i]['falling'], result.event_states[i]['falling'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['falling'], surrogate_results.event_states[i]['falling'])
            self.assertAlmostEqual(surrogate_results.states[i]['impact'], result.event_states[i]['impact'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['impact'], surrogate_results.event_states[i]['impact'])

    def test_surrogate_options(self):
        m = ThrownObject()
        def load_eqn(t = None, x = None):
            return m.InputContainer({})
        
        # treat warnings as exceptions
        warnings.filterwarnings("error")

        try:
            surrogate = m.generate_surrogate([load_eqn], dt = 0.1, save_freq = 0.25, threshold_keys = 'impact', states = ['v'])
            self.fail('warning not raised')
        except Warning:
            pass

        try:
            # Set sufficiently large failure tolerance
            surrogate = m.generate_surrogate([load_eqn], dt = 0.1, save_freq = 0.25, threshold_keys = 'impact', states = ['v'], stability_tol=1e99)
        except Warning:
            self.fail('Warning raised')
        
        # Reset Warnings
        warnings.filterwarnings("default")

    def test_surrogate_use_error_cases(self):
        m = ThrownObject()
        def load_eqn(t = None, x = None):
            return m.InputContainer({})
        
        surrogate = m.generate_surrogate([load_eqn], dt = 0.1, save_freq = 0.25, threshold_keys = 'impact')

        with self.assertWarns(Warning):
            surrogate.simulate_to_threshold(load_eqn, dt = 0.05)

# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Surrogate")
    result = runner.run(l.loadTestsFromTestCase(TestSurrogate)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
