# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from prog_models.sim_result import SimResult, LazySimResult

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
    
    def test_pickle(self):
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = SimResult(time, state)
        import pickle
        pickle.dump(result, open('model_test.pkl', 'wb'))
        result2 = pickle.load(open('model_test.pkl', 'rb'))
        self.assertEqual(result, result2)

    def test_extend(self):
        NUM_ELEMENTS = 5 # Creating two result objects
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = SimResult(time, state)
        NUM_ELEMENTS = 10
        time = list(range(NUM_ELEMENTS))
        state = [i * 10.0 for i in range(NUM_ELEMENTS)]
        result2 = SimResult(time, state)
        self.assertEqual(result.times, [0, 1, 2, 3, 4])
        self.assertEqual(result2.times, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(result.data, [0.0, 2.5, 5.0, 7.5, 10.0]) # Assert data is correct before extending
        self.assertEqual(result2.data, [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
        
        result.extend(result2) # Extend result with result2
        self.assertEqual(result.times, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(result.data, [0.0, 2.5, 5.0, 7.5, 10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    
    def test_index(self):
        NUM_ELEMENTS = 5 # Creating two result objects
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = SimResult(time, state)
        # other_arg = {2.5:1}
        # other_arg = {1:2.5}
        other_arg = 2.5

        self.assertEqual(result.index(other_arg), 1)

    def test_pop(self):
        NUM_ELEMENTS = 5 # Creating two result objects
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = SimResult(time, state)

        result.pop(2) # Test specified index
        self.assertEqual(result.data, [0.0, 2.5, 7.5, 10.0])
        result.pop() # Test default index -1 (last element)
        self.assertEqual(result.data, [0.0, 2.5, 7.5])
        self.assertRaises(IndexError, result.pop, 5) # Test specifying an invalid index value
        self.assertRaises(IndexError, result.pop, 3)
        self.assertRaises(TypeError, result.pop, "5") # Test specifying an invalid index type
        self.assertRaises(TypeError, result.pop, [0,1])
        self.assertRaises(TypeError, result.pop, {})
        self.assertRaises(TypeError, result.pop, set())
        self.assertRaises(TypeError, result.pop, 1.5)

    def test_remove(self):
        NUM_ELEMENTS = 5 # Creating two result objects
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = SimResult(time, state)

        # print(result.times) # [0, 1, 2, 3, 4]
        # print(result.data)  # [0.0, 2.5, 5.0, 7.5, 10.0]
        result.remove(0) # Test specified index
        self.assertEqual(result.times, [1, 2, 3, 4])
        self.assertEqual(result.data, [2.5, 5.0, 7.5, 10.0])

        # result.remove(1) # Test specified index
        # self.assertEqual(result.times, [2, 3, 4]) # Passes
        # self.assertEqual(result.data, [2.5, 5.0, 7.5, 10.0]) # wrong behavior? ValueError 

        # because ValueError, any type can be passed to be removed
        # self.assertRaises(TypeError, result.remove, ) # Test no index specified
        # self.assertRaises(TypeError, result.remove, "5") # Tests specifying an invalid index type
        # self.assertRaises(TypeError, result.remove, [0,1])
        # self.assertRaises(TypeError, result.remove, {})
        # self.assertRaises(TypeError, result.remove, set())
        # self.assertRaises(TypeError, result.remove, 1.5)

        # self.assertRaises(IndexError, result.remove, 5) # Test specifying an invalid value
        # self.assertRaises(IndexError, result.remove, 3)

    def test_clear(self):
        NUM_ELEMENTS = 5 # Creating two result objects
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = SimResult(time, state)
        self.assertEqual(result.times, [0, 1, 2, 3, 4])
        self.assertEqual(result.data, [0.0, 2.5, 5.0, 7.5, 10.0])
        self.assertRaises(TypeError, result.clear, True)

        result.clear()
        self.assertEqual(result.times, [])
        self.assertEqual(result.data, [])

    def test_time(self):
        NUM_ELEMENTS = 5 # Creating two result objects
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = SimResult(time, state)
        self.assertEqual(result.time(0), result.times[0])
        self.assertEqual(result.time(1), result.times[1])
        self.assertEqual(result.time(2), result.times[2])
        self.assertEqual(result.time(3), result.times[3])
        self.assertEqual(result.time(4), result.times[4])

        self.assertRaises(TypeError, result.time, ) # Test no input given
        self.assertRaises(TypeError, result.time, "0") # Tests specifying an invalid index type 
        self.assertRaises(TypeError, result.time, [0,1])
        self.assertRaises(TypeError, result.time, {})
        self.assertRaises(TypeError, result.time, set())
        self.assertRaises(TypeError, result.time, 1.5)

    def test_plot(self):
        NUM_ELEMENTS = 5 # Creating two result objects
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = SimResult(time, state)
        # INCOMPLETE
    
    def test_cached_sim_result(self):
        def f(x):
            return x * 2
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = LazySimResult(f, time, state)
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

        # Catch bug that occured where lazysimresults weren't actually different
        # This occured because the underlying arrays of time and state were not copied (see PR #158)
        result = LazySimResult(f, time, state)
        result2 = LazySimResult(f, time, state)
        self.assertTrue(result == result2)
        self.assertEqual(len(result), len(result2))
        result.extend(LazySimResult(f, time, state))
        self.assertFalse(result == result2)
        self.assertNotEqual(len(result), len(result2))

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
