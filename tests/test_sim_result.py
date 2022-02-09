# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest

from attr import Attribute
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
        self.assertEqual(result.index(10.0), 4)
        self.assertEqual(result.index(2.5), 1)
        self.assertEqual(result.index(0.0), 0)
        self.assertRaises(ValueError, result.index, 6.0) # Other argument doesn't exist
        self.assertRaises(ValueError, result.index, -1) # Non-existent data value
        self.assertRaises(ValueError, result.index, "7.5") # Data specified incorrectly as string
        self.assertRaises(ValueError, result.index, None) # Not type errors because its simply looking for an object in list
        self.assertRaises(ValueError, result.index, [1, 2])
        self.assertRaises(ValueError, result.index, {})
        self.assertRaises(ValueError, result.index, set())

    def test_pop(self):
        NUM_ELEMENTS = 5 # Creating two result objects
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = SimResult(time, state)

        result.pop(2) # Test specified index
        self.assertEqual(result.data, [0.0, 2.5, 7.5, 10.0])
        result.pop() # Test default index -1 (last element)
        self.assertEqual(result.data, [0.0, 2.5, 7.5])
        result.pop(-1) # Test argument of index -1 (last element)
        self.assertEqual(result.data, [0.0, 2.5])
        result.pop(0) # Test argument of 0
        self.assertEqual(result.data, [2.5])
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
        # Testing model taken from events.py
        YELLOW_THRESH, RED_THRESH, THRESHOLD = 0.15, 0.1, 0.05
        from prog_models.models import BatteryElectroChemEOD
        class MyBatt(BatteryElectroChemEOD):
            events = BatteryElectroChemEOD.events + ['EOD_warn_yellow', 'EOD_warn_red', 'EOD_requirement_threshold']
            def event_state(self, state):
                event_state = super().event_state(state)
                event_state['EOD_warn_yellow'] = (event_state['EOD']-YELLOW_THRESH)/(1-YELLOW_THRESH) 
                event_state['EOD_warn_red'] = (event_state['EOD']-RED_THRESH)/(1-RED_THRESH)
                event_state['EOD_requirement_threshold'] = (event_state['EOD']-THRESHOLD)/(1-THRESHOLD)
                return event_state
            def threshold_met(self, x):
                t_met =  super().threshold_met(x)
                event_state = self.event_state(x)
                t_met['EOD_warn_yellow'] = event_state['EOD_warn_yellow'] <= 0
                t_met['EOD_warn_red'] = event_state['EOD_warn_red'] <= 0
                t_met['EOD_requirement_threshold'] = event_state['EOD_requirement_threshold'] <= 0
                return t_met
        def future_loading(t, x=None):
            if (t < 600): i = 2
            elif (t < 900): i = 1
            elif (t < 1800): i = 4
            elif (t < 3000): i = 2     
            else: i = 3
            return {'i': i} 
        m = MyBatt()
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_loading, threshold_keys=['EOD'], print = False)
        plot_test = event_states.plot() # Plot doesn't raise error

    def test_not_implemented(self):
        # Not implemented functions, should raise errors
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = SimResult(time, state)
        self.assertRaises(NotImplementedError, result.append)
        self.assertRaises(NotImplementedError, result.count)
        self.assertRaises(NotImplementedError, result.insert)
        self.assertRaises(NotImplementedError, result.reverse)

    # Tests for LazySimResult
    def test_lazy_data_fcn(self):
        def f(x):
            return x * 2
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = LazySimResult(f, time, state)
        self.assertEqual(result.data, [0.0, 5.0, 10.0, 15.0, 20.0])

    def test_lazy_clear(self):
        def f(x):
            return x * 2
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = LazySimResult(f, time, state)
        self.assertEqual(result.times, [0, 1, 2, 3, 4])
        self.assertEqual(result.data, [0.0, 5.0, 10.0, 15.0, 20.0])
        self.assertEqual(result.states, [0.0, 2.5, 5.0, 7.5, 10.0])
        self.assertRaises(TypeError, result.clear, True)

        result.clear()
        self.assertEqual(result.times, [])
        self.assertEqual(result.data, [])
        self.assertEqual(result.states, [])

    def test_lazy_extend(self):
        def f(x):
            return x * 2
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = LazySimResult(f, time, state)

        def f2(x):
            return x * 5
        NUM_ELEMENTS = 10
        time2 = list(range(NUM_ELEMENTS))
        state2 = [i * 5 for i in range(NUM_ELEMENTS)]
        result2 = LazySimResult(f2, time2, state2)
        self.assertEqual(result.times, [0, 1, 2, 3, 4]) # Assert data is correct before extending
        self.assertEqual(result.data, [0.0, 5.0, 10.0, 15.0, 20.0])
        self.assertEqual(result.states, [0.0, 2.5, 5.0, 7.5, 10.0])
        self.assertEqual(result2.times, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(result2.data, [0, 25, 50, 75, 100, 125, 150, 175, 200, 225])
        self.assertEqual(result2.states, [0, 5, 10, 15, 20, 25, 30, 35, 40, 45])

        result.extend(result2)
        self.assertEqual(result.times, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # Assert data is correct after extending
        self.assertEqual(result.data, [0.0, 5.0, 10.0, 15.0, 20.0, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        self.assertEqual(result.states, [0.0, 2.5, 5.0, 7.5, 10.0, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45])

    def test_lazy_pop(self):
        def f(x):
            return x * 2
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = LazySimResult(f, time, state)

        result.pop(1) # Test specified index
        self.assertEqual(result.times, [0, 2, 3, 4])
        self.assertEqual(result.data, [0.0, 10.0, 15.0, 20.0])
        self.assertEqual(result.states, [0.0, 5.0, 7.5, 10.0])
        result.pop() # Test default index -1 (last element)
        self.assertEqual(result.times, [0, 2, 3])
        self.assertEqual(result.data, [0.0, 10.0, 15.0])
        self.assertEqual(result.states, [0.0, 5.0, 7.5])
        result.pop(-1) # Test argument of index -1 (last element)
        self.assertEqual(result.times, [0, 2])
        self.assertEqual(result.data, [0.0, 10.0])
        self.assertEqual(result.states, [0.0, 5.0])
        result.pop(0) # Test argument of 0
        self.assertEqual(result.times, [2])
        self.assertEqual(result.data, [10.0])
        self.assertEqual(result.states, [5.0])
        # Test erroneous input
        self.assertRaises(IndexError, result.pop, 5) # Test specifying an invalid index value
        self.assertRaises(IndexError, result.pop, 3)
        self.assertRaises(TypeError, result.pop, "5") # Test specifying an invalid index type
        self.assertRaises(TypeError, result.pop, [0,1])
        self.assertRaises(TypeError, result.pop, {})
        self.assertRaises(TypeError, result.pop, set())
        self.assertRaises(TypeError, result.pop, 1.5)

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

    def test_lazy_not_implemented(self):
        # Not implemented functions, should raise errors
        def f(x):
            return x * 2
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = LazySimResult(f, time, state)
        self.assertRaises(NotImplementedError, result.append)
        self.assertRaises(NotImplementedError, result.count)
        self.assertRaises(NotImplementedError, result.insert)
        self.assertRaises(NotImplementedError, result.reverse)


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
