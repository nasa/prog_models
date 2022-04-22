# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import sys
import unittest

class TestManual(unittest.TestCase):
    # Test successful download of datasets
    def test_nasa_battery_download(self):
        from prog_models.datasets import nasa_battery
        # RW1 = "https://ti.arc.nasa.gov/c/27/" - a functional URL
        (desc, data) = nasa_battery.load_data(1) # is simply passing okay? a lot of values returned from desc, data
    def test_nasa_cmapss_download(self):
        from prog_models.datasets import nasa_cmapss
        # URL = "https://ti.arc.nasa.gov/c/6/" - a functional URL
        (desc, data) = nasa_cmapss.load_data(1)

# This allows the module to be executed directly
def run_tests():
    unittest.main()

def main():
    # This ensures that the directory containing ProgModelTemplate is in the python search directory
    from os.path import dirname, join
    sys.path.append(join(dirname(__file__), ".."))

    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Manual")
    result = runner.run(l.loadTestsFromTestCase(TestManual)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()

