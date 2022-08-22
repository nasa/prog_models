# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
import warnings
import importlib.util


class TestTutorials(unittest.TestCase):

    def test_tutorial_ipynb(self):
        if importlib.util.find_spec('testbook') is None:
            warnings.warn('testbook not imported')
        else:
            with testbook('./tutorial.ipynb', execute=True) as tb:
                from testbook import testbook
                self.assertEqual(tb.__class__.__name__, "TestbookNotebookClient")

def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Tutorials")
    try:
        result = runner.run(l.loadTestsFromTestCase(TestTutorials)).wasSuccessful()
    except Exception as e:
        if not result:
            raise Exception("Failed test")
          

if __name__ == '__main__':
    main()
    