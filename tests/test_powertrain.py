# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest

from prog_models.models import ESC, DCMotor, Powertrain


class TestPowertrain(unittest.TestCase):
    def setUp(self):
        # set stdout (so it wont print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__
    
    def test_powertrain(self):
        esc = ESC()
        motor = DCMotor()
        powertrain = Powertrain(esc, motor)
        def future_loading(t, x=None):
            return powertrain.InputContainer({
                'duty': 1,
                'v': 23
            })
        
        (times, inputs, states, outputs, event_states) = powertrain.simulate_to(2, future_loading, dt=2e-5, save_freq=0.1)
        # Add additional tests

# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Powertrain model")
    result = runner.run(l.loadTestsFromTestCase(TestPowertrain)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
