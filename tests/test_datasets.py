# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import sys
import unittest

class TestDatasets(unittest.TestCase):
    # Bad URL tests
    def test_nasa_battery_bad_url_download(self):
        from prog_models.datasets import nasa_battery
        BAD_URL = "BADURLTEST"
        nasa_battery.urls = {'RW1':"https://"+BAD_URL}
        with self.assertRaises(ConnectionError):
            (desc, data) = nasa_battery.load_data(1)
    def test_nasa_cmapss_bad_url_download(self):
        from prog_models.datasets import nasa_cmapss
        BAD_URL = "https://"+"BADURLTEST"
        nasa_cmapss.URL = BAD_URL
        with self.assertRaises(ConnectionError):
            (train, test, results) = nasa_cmapss.load_data(1)
    # Testing for successful download located in manual testing files; test_manual.py

# This allows the module to be executed directly
def run_tests():
    unittest.main()

def main():
    # This ensures that the directory containing ProgModelTemplate is in the python search directory
    from os.path import dirname, join
    sys.path.append(join(dirname(__file__), ".."))

    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Datasets")
    result = runner.run(l.loadTestsFromTestCase(TestDatasets)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
