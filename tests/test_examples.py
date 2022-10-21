# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

from importlib import import_module
from io import StringIO 
from os.path import dirname, join
import pkgutil
import sys
import unittest

sys.path.append(join(dirname(__file__), ".."))  # needed to access examples
from examples import *

EXAMPLES_SKIPPED = ['dataset', 'sim_battery_eol']

def make_test_function(example):
    def test(self):
        ex = import_module("examples." + example)

        with unittest.mock.patch('matplotlib.pyplot.show'):
            ex.run_example()
    return test


class TestExamples(unittest.TestCase):
    def setUp(self):
        # set stdout (so it wont print)
        sys.stdout = StringIO()

    def tearDown(self):
        # reset stdout
        sys.stdout = sys.__stdout__

# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    # Create tests for each example
    for _, name, _ in pkgutil.iter_modules(['examples']):
        if name not in EXAMPLES_SKIPPED:
            test_func = make_test_function(name)
            setattr(TestExamples, 'test_{0}'.format(name), test_func)   


    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Examples")
    result = runner.run(l.loadTestsFromTestCase(TestExamples)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
