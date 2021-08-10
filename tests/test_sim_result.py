# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from prog_models.sim_result import SimResult, CachedSimResult

class TestSimResult(unittest.TestCase):
    def test_sim_result(self):
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = SimResult(time, state)
        self.assertListEqual(list(result), state)
        self.assertListEqual(result.times, time)
        for i in range(5):
            self.assertEqual(result.time(i), time[i])
            self.assertEqual(result[i], state[i])

        try:
            tmp = result[NUM_ELEMENTS]
            self.fail("Should be out of range error")
        except IndexError:
            pass

        try:
            tmp = result.time(NUM_ELEMENTS)
            self.fail("Should be out of range error")
        except IndexError:
            pass
    
    def test_cached_sim_result(self):
        def f(x):
            return x * 2
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = CachedSimResult(f, time, state)
        self.assertFalse(result.is_cached())
        self.assertListEqual(result.times, time)
        for i in range(5):
            self.assertEqual(result.time(i), time[i])
            self.assertEqual(result[i], state[i]*2)
        self.assertTrue(result.is_cached())

        try:
            tmp = result[NUM_ELEMENTS]
            self.fail("Should be out of range error")
        except IndexError:
            pass

        try:
            tmp = result.time(NUM_ELEMENTS)
            self.fail("Should be out of range error")
        except IndexError:
            pass

# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    # This ensures that the directory containing ProgModelTemplate is in the python search directory
    import sys
    from os.path import dirname, join
    sys.path.append(join(dirname(__file__), ".."))

    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Sim Result")
    result = runner.run(l.loadTestsFromTestCase(TestSimResult)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()