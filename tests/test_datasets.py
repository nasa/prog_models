import io
import sys
import unittest

class TestDatasets(unittest.TestCase):
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
