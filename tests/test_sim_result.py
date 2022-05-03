# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from prog_models import sim_result

from prog_models.sim_result import SimResult, LazySimResult

class TestSimResult(unittest.TestCase):
    def test_sim_result(self):
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(NUM_ELEMENTS)]
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

        self.assertRaises(ValueError, result.extend, 0) # Passing non-LazySimResult types to extend method
        self.assertRaises(ValueError, result.extend, [0,1])
        self.assertRaises(ValueError, result.extend, {})
        self.assertRaises(ValueError, result.extend, set())
        self.assertRaises(ValueError, result.extend, 1.5)

    def test_extended_by_lazy(self):
        NUM_ELEMENTS = 5 
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = SimResult(time, state) # Creating one SimResult object
        def f(x):
            return x * 2
        NUM_ELEMENTS = 10
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result2 = LazySimResult(f, time, state) # Creating one LazySimResult object

        self.assertEqual(result.times, [0, 1, 2, 3, 4])
        self.assertEqual(result2.times, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(result.data, [0.0, 2.5, 5.0, 7.5, 10.0]) # Assert data is correct before extending
        self.assertEqual(result2.data, [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])
        result.extend(result2) # Extend result with result2
        self.assertEqual(result.times, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(result.data, [0.0, 2.5, 5.0, 7.5, 10.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])

    def test_pickle_lazy(self):
        def f(x):
            return x * 2
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        lazy_result = LazySimResult(f, time, state) # Ordinary LazySimResult with f, time, state
        sim_result = SimResult(time, state) # Ordinary SimResult with time,state

        converted_lazy_result = SimResult(lazy_result.times, lazy_result.data)
        self.assertNotEqual(sim_result, converted_lazy_result) # converted is not the same as the original SimResult

        import pickle # try pickle'ing
        pickle.dump(lazy_result, open('model_test.pkl', 'wb'))
        pickle_converted_result = pickle.load(open('model_test.pkl', 'rb'))
        self.assertEqual(converted_lazy_result, pickle_converted_result)
    
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
        NUM_ELEMENTS = 5
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

        result.remove(5.0) # Positional defaults to removing data
        self.assertEqual(result.times, [0, 1, 3, 4])
        self.assertEqual(result.data, [0.0, 2.5, 7.5, 10.0])
        result.remove(d = 0.0) # Testing named removal of data
        self.assertEqual(result.times, [1, 3, 4])
        self.assertEqual(result.data, [2.5, 7.5, 10.0])
        result.remove(t = 3) # Testing named removal of time
        self.assertEqual(result.times, [1, 4])
        self.assertEqual(result.data, [2.5, 10.0])
        result.remove(t = 1) 
        self.assertEqual(result.times, [4])
        self.assertEqual(result.data, [10.0])

        self.assertRaises(ValueError, result.remove, ) # If nothing specified, raise ValueError
        self.assertRaises(ValueError, result.remove, None, None) # Passing both as None
        self.assertRaises(ValueError, result.remove, 0.0, 1) # Passing arguments to both
        self.assertRaises(ValueError, result.remove, 7.5) # Test nonexistent data value
        self.assertRaises(ValueError, result.remove, -1) # Type checking negated as index searches for element in list
        self.assertRaises(ValueError, result.remove, "5") # Thus all value types allowed to be searched
        self.assertRaises(ValueError, result.remove, [0,1])
        self.assertRaises(ValueError, result.remove, {})
        self.assertRaises(ValueError, result.remove, set())

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

    def test_namedtuple_access(self):
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
        named_results = m.simulate_to_threshold(future_loading, threshold_keys=['EOD'], print = False)
        times = named_results.times
        inputs = named_results.inputs
        states = named_results.states
        outputs = named_results.outputs
        event_states = named_results.event_states

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
        self.assertFalse(result.is_cached())
        self.assertEqual(result.data, [0.0, 5.0, 10.0, 15.0, 20.0])
        self.assertTrue(result.is_cached())

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
        self.assertEqual(result.data, [0.0, 5.0, 10.0, 15.0, 20.0, 0, 25, 50, 75, 100, 125, 150, 175, 200, 225])
        self.assertEqual(result.states, [0.0, 2.5, 5.0, 7.5, 10.0, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45])

    def test_lazy_extend_cache(self):
        def f(x):
            return x * 2
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result1 = LazySimResult(f, time, state)
        result2 = LazySimResult(f, time, state)

        # Case 1
        result1.extend(result2)
        self.assertFalse(result1.is_cached()) # False
        
        # Case 2
        result1 = LazySimResult(f, time, state) # Reset result1
        store_test_data = result1.data # Access result1 data
        result1.extend(result2) 
        self.assertFalse(result1.is_cached()) # False

        # Case 3
        result1 = LazySimResult(f, time, state) # Reset result1
        store_test_data = result2.data # Access result2 data
        result1.extend(result2) 
        self.assertFalse(result1.is_cached()) # False

        # Case 4
        result1 = LazySimResult(f, time, state) # Reset result1
        result2 = LazySimResult(f, time, state) # Reset result2
        store_test_data1 = result1.data # Access result1 data
        store_test_data2 = result2.data # Access result2 data
        result1.extend(result2) 
        self.assertTrue(result1.is_cached()) # True

    def test_lazy_extend_error(self):
        def f(x):
            return x * 2
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = LazySimResult(f, time, state)

        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        sim_result = SimResult(time, state)

        self.assertRaises(ValueError, result.extend, sim_result) # Passing a SimResult to LazySimResult's extend
        self.assertRaises(ValueError, result.extend, 0) # Passing non-LazySimResult types to extend method
        self.assertRaises(ValueError, result.extend, [0,1])
        self.assertRaises(ValueError, result.extend, {})
        self.assertRaises(ValueError, result.extend, set())
        self.assertRaises(ValueError, result.extend, 1.5)

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

    def test_lazy_remove(self):
        def f(x):
            return x * 2
        NUM_ELEMENTS = 10
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = LazySimResult(f, time, state)

        result.remove(5.0) # Unnamed default positional argument removal of data value
        self.assertEqual(result.times, [0, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(result.data, [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0] )
        self.assertEqual(result.states, [0.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5])
        result.remove(d = 0.0) # Named argument removal of data value
        self.assertEqual(result.times, [2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(result.data, [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0] )
        self.assertEqual(result.states, [5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5])
        result.remove(t = 7) # Named argument removal of times value
        self.assertEqual(result.times, [2, 3, 4, 5, 6, 8, 9])
        self.assertEqual(result.data, [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 45.0] )
        self.assertEqual(result.states, [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 22.5])
        result.remove(s = 12.5) # Named argument removal of states value
        self.assertEqual(result.times, [2, 3, 4, 6, 8, 9])
        self.assertEqual(result.data, [10.0, 15.0, 20.0, 30.0, 40.0, 45.0] )
        self.assertEqual(result.states, [5.0, 7.5, 10.0, 15.0, 20.0, 22.5])

        self.assertRaises(ValueError, result.remove, ) # Test no values specified
        self.assertRaises(ValueError, result.remove, 90.0, 2) # Test two values specified positionally
        self.assertRaises(ValueError, result.remove, 90.0, 2, 15.0) # Test three values specified positionally
        self.assertRaises(ValueError, result.remove, d=90.0, t=2) # Test d,t values specified by name
        self.assertRaises(ValueError, result.remove, t=2, s=15.0) # Test s,t values specified by name
        self.assertRaises(ValueError, result.remove, d=90.0, s=15.0) # Test d,s values specified by name
        self.assertRaises(ValueError, result.remove, d=90.0, t=2, s=15.0) # Test three values specified by name
        self.assertRaises(ValueError, result.remove, 90.0) # Test nonexistent data value
        self.assertRaises(ValueError, result.remove, d=90.0) # Test nonexistent data value
        self.assertRaises(ValueError, result.remove, t=90.0) # Test nonexistent times value
        self.assertRaises(ValueError, result.remove, s=90.0) # Test nonexistent states value
        self.assertRaises(ValueError, result.remove, -1) # Type checking negated as index searches for element in list
        self.assertRaises(ValueError, result.remove, "5") # Thus all value types allowed to be searched
        self.assertRaises(ValueError, result.remove, [0,1])
        self.assertRaises(ValueError, result.remove, {})
        self.assertRaises(ValueError, result.remove, set())

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

    def test_lazy_to_simresult(self):
        def f(x):
            return x * 2
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [i * 2.5 for i in range(NUM_ELEMENTS)]
        result = LazySimResult(f, time, state)

        converted_result = result.to_simresult()
        self.assertTrue(isinstance(converted_result, SimResult)) # Ensure type is SimResult
        self.assertEqual(converted_result.times, result.times) # Compare to original LazySimResult
        self.assertEqual(converted_result.data, result.data)
        self.assertEqual(converted_result.times, [0, 1, 2, 3, 4]) # Compare to expected values
        self.assertEqual(converted_result.data, [0.0, 5.0, 10.0, 15.0, 20.0])
        
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
