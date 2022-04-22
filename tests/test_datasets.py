import io
import sys
import unittest

class TestDatasets(unittest.TestCase):
    # TEST BATTERY ERROR EMSSAGE
    # Need internet connection off to pass
    # Can't pass in gibberish URL because load_data maps an int to a url: can't set url externally of function
    
    def test_nasa_battery_download_error(self):
        from prog_models.datasets import nasa_battery
        with self.assertRaises(ConnectionError):
            # RW1 = "https://ti.arc.nasa.gov/c/27/" - a functional URL
            (desc, data) = nasa_battery.load_data(1)
    def test_nasa_cmapss_download_error(self):
        from prog_models.datasets import nasa_cmapss
        with self.assertRaises(ConnectionError):
            # URL = "https://ti.arc.nasa.gov/c/6/" - a functional URL
            (desc, data) = nasa_cmapss.load_data(1)

    def test_nasa_cmpass_download_error(self):
        pass

# This allows the module to be executed directly
def run_tests():
    unittest.main()

def main():
    # This ensures that the directory containing ProgModelTemplate is in the python search directory
    from os.path import dirname, join
    sys.path.append(join(dirname(__file__), ".."))

    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Base Models")
    result = runner.run(l.loadTestsFromTestCase(TestDatasets)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
