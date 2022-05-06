# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest

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
            self.assertDictEqual(surrogate_results.inputs[i], surrogate_results.inputs[i])
            self.assertAlmostEqual(surrogate_results.states[i]['x'], result.states[i]['x'], delta=8)
            self.assertEqual(surrogate_results.states[i]['x'], surrogate_results.outputs[i]['x'])
            self.assertAlmostEqual(surrogate_results.states[i]['v'], result.states[i]['v'], delta=1)
            self.assertAlmostEqual(surrogate_results.states[i]['falling'], result.event_states[i]['falling'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['falling'], surrogate_results.event_states[i]['falling'])
            self.assertAlmostEqual(surrogate_results.states[i]['impact'], result.event_states[i]['impact'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['impact'], surrogate_results.event_states[i]['impact'])

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
