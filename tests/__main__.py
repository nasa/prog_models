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

    print("\n\nTesting individual exectution of test files")

    # Run tests individually to test them and make sure they can be executed individually
    was_successful = True
    try:
        exec(open("tests/test_base_models.py").read())
    except Exception:
        was_successful = False

    try:
        exec(open("tests/test_examples.py").read())
    except Exception:
        was_successful = False
        
    try:
        exec(open("tests/test_battery.py").read())
    except Exception:
        was_successful = False

    try:
        exec(open("tests/test_centrifugal_pump.py").read())
    except Exception:
        was_successful = False

    try:
        exec(open("tests/test_pneumatic_valve.py").read())
    except Exception:
        was_successful = False

    if not was_successful:
        raise Exception("Failed test")
