# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from nbformat import read, NO_CONVERT
from nbconvert.preprocessors import ExecutePreprocessor

class TestTutorials(unittest.TestCase):
    def test_tutorial_ipynb(self):
        with open('./tutorial.ipynb') as file:
            ExecutePreprocessor(timeout=600, kernel_name='python3').preprocess(read(file, NO_CONVERT))

def run_tests():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Tutorials")
    result = runner.run(l.loadTestsFromTestCase(TestTutorials)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    run_tests()
    