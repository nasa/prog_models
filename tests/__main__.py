# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
from .test_base_models import TestModels
from .test_examples import TestExamples
from .test_centrifugal_pump import TestCentrifugalPump
from .test_pneumatic_valve import TestPneumaticValve
from .test_battery import TestBattery

from io import StringIO 
import sys
import unittest
from examples import sim_example

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
    print("\n\nTesting Base Models")
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestModels))

    print("\n\nTesting Examples")
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestExamples))

    print("\n\nTesting Centrifugal Pump model")
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestCentrifugalPump))

    print("\n\nTesting Pneumatic Valve model")
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestPneumaticValve))

    print("\n\nTesting Battery models")
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestBattery))