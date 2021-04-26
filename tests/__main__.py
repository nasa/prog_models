# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
from .test_base_models import TestModels
from .test_examples import TestExamples
from .test_centrifugal_pump import TestCentrifugalPump
from .test_pneumatic_valve import TestPneumaticValve
from .test_battery import TestBattery

from io import StringIO 
import sys
import unittest
from examples import sim as sim_example

def _test_ex():
    # set stdout (so it wont print)
    _stdout = sys.stdout
    sys.stdout = StringIO()

    # Run example
    sim_example.run_example()

    # Reset stdout 
    sys.stdout = _stdout

if __name__ == '__main__':
    from timeit import timeit
    print("\nExample Runtime: ", timeit(_test_ex, number=10))

    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Base Models")
    result = runner.run(l.loadTestsFromTestCase(TestModels)).wasSuccessful()

    print("\n\nTesting Examples")
    result = runner.run(l.loadTestsFromTestCase(TestExamples)).wasSuccessful() and result

    print("\n\nTesting Centrifugal Pump model")
    result = runner.run(l.loadTestsFromTestCase(TestCentrifugalPump)).wasSuccessful() and result

    print("\n\nTesting Pneumatic Valve model")
    result = runner.run(l.loadTestsFromTestCase(TestPneumaticValve)).wasSuccessful() and result

    print("\n\nTesting Battery models")
    result = runner.run(l.loadTestsFromTestCase(TestBattery)).wasSuccessful() and result

    if not result:
        raise Exception("Failed test")