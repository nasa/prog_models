import sys
import unittest

class TestDatasets(unittest.TestCase):
    # Bad URL tests
    def test_nasa_battery_bad_url_download(self):
        from prog_models.datasets import nasa_battery
        BAD_URL = "BADURLTEST"
        nasa_battery.urls = {'RW1':"https://"+BAD_URL}
        (desc, data) = nasa_battery.load_data(1)

    @unittest.skip # FOR MANUAL TESTS
    def test_nasa_battery_download(self):
        from prog_models.datasets import nasa_battery
        # RW1 = "https://ti.arc.nasa.gov/c/27/" - a functional URL
        (desc, data) = nasa_battery.load_data(1) # is simply passing okay? a lot of values returned from desc, data
    @unittest.skip # FOR MANUAL TESTS
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
    print("\n\nTesting Base Models")
    result = runner.run(l.loadTestsFromTestCase(TestDatasets)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
