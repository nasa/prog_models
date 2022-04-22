# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

import sys
from os.path import dirname, join
sys.path.append(join(dirname(__file__), ".."))

import unittest
from io import StringIO 
from examples import *
from unittest.mock import patch
import pkgutil
from importlib import import_module

examples_skipped = ['dataset', 'sim_battery_eol']

def make_test_function(example):
    def test(self):
        ex = import_module("examples." + example)

        with patch('matplotlib.pyplot.show'):
            ex.run_example()
    return test


class TestExamples(unittest.TestCase):
    def setUp(self):
        # set stdout (so it wont print)
        self._stdout = sys.stdout
        sys.stdout = StringIO()

    def tearDown(self):
        # reset stdout
        sys.stdout = self._stdout

# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    # Create tests for each example
    for _, name, _ in pkgutil.iter_modules(['examples']):
        if name not in examples_skipped:
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
