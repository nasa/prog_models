# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import numpy as np
import pickle
import sys
import unittest
import pandas as pd

from prog_models.models import BatteryElectroChemEOD
from prog_models.sim_result import SimResult, LazySimResult
from prog_models.utils.containers import DictLikeMatrixWrapper


class TestSimResult(unittest.TestCase):
    """def setUp(self):
        set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__"""

    def test_sim_result(self):
        # Variables
        time = list(range(5))
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
        frame = pd.DataFrame(state)
        frame.insert(0, "time", time)
        result = SimResult(time, state)
        # Checks values from SimResult object and static variables
        self.assertListEqual(list(result), state)
        self.assertListEqual(result.times, time)
        self.assertTrue(frame.equals(result.frame))
        for i in range(5):
            self.assertEqual(result.get_time(i), time[i])
            self.assertEqual(result[i], state[i])
        self.assertTrue(frame.equals(result.frame))
        try:
            tmp = result[5]
            self.fail("Should be out of range error")
        except IndexError:
            pass
        try:
            tmp = result.times[5]
            self.fail("Should be out of range error")
        except IndexError:
            pass

    def test_pickle(self):
        # Variables
        time = list(range(5))  # list of int, 0 to 4
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
        result = SimResult(time, state)
        pickle.dump(result, open('model_test.pkl', 'wb'))
        result2 = pickle.load(open('model_test.pkl', 'rb'))
        self.assertEqual(result, result2)
        self.assertTrue(result.frame.equals(result2.frame))

    def test_extend(self):
        # Variables
        time = list(range(5))  # list of int from 0 to 4
        state = [{'a': i * 2.5, 'b': i * 2.5} for i in range(5)]
        result = SimResult(time, state)
        time2 = list(range(10))  # list of int from 0 to 9
        state2 = [{'a': i * 5, 'b': i * 5} for i in range(10)]
        result2 = SimResult(time2, state2)
        time_extended = time + time2
        state_extended = state + state2
        result_df = pd.DataFrame(state)
        result_df.insert(0, "time", time)
        result2_df = pd.DataFrame(state2)
        result2_df.insert(0, "time", time2)
        result_extend_df = pd.concat([result_df, result2_df], axis=0).reset_index(drop=True)

        self.assertEqual(result.times, time)
        self.assertEqual(result2.times, time2)
        self.assertEqual(result.data, state)  # Assert data is correct before extending
        self.assertEqual(result2.data, state2)

        result.extend(result2)  # Extend result with result2
        self.assertEqual(result.times, time_extended)
        self.assertEqual(result.data, state_extended)
        self.assertTrue(result.frame.equals(result_extend_df))

        self.assertRaises(ValueError, result.extend, 0)  # Passing non-LazySimResult types to extend method
        self.assertRaises(ValueError, result.extend, [0, 1])
        self.assertRaises(ValueError, result.extend, {})
        self.assertRaises(ValueError, result.extend, set())
        self.assertRaises(ValueError, result.extend, 1.5)

    def test_extended_by_lazy(self):
        # Variables
        time = list(range(5))  # list of int, 0 to 4
        state = [{'a': i * 2.5, 'b': i * 2.5} for i in range(5)]
        time2 = list(range(10))  # list of int, 0 to 9
        state2 = [{'a': i * 5, 'b': i * 5} for i in range(10)]
        data2 = [{'a': i * 10, 'b': i * 10} for i in range(10)]
        result_df = pd.DataFrame(state)
        result_df.insert(0, "time", time)
        result2_df = pd.DataFrame(state2)
        result2_df.insert(0, "time", time2)
        result_extend_df = pd.concat([result_df, result2_df], axis=0).reset_index(drop=True)
        result = SimResult(time, state)  # Creating one SimResult object

        def f(x):
            return {k: v * 2 for k, v in x.items()}

        result2 = LazySimResult(f, time2, state2)  # Creating one LazySimResult object
        # confirming the data in result and result2 are correct
        self.assertEqual(result.times, time)
        self.assertEqual(result2.times, time2)
        self.assertEqual(result.data, state)  # Assert data is correct before extending
        self.assertEqual(result2.data, data2)
        result.extend(result2)  # Extend result with result2
        # check data when result is extended with result2
        self.assertEqual(result.times, time + time2)
        self.assertEqual(result.data, state + data2)
        self.assertTrue(result.frame.equals(result_extend_df))

    def test_pickle_lazy(self):
        def f(x):
            return {k: v * 2 for k, v in x.items()}

        # Variables
        time = list(range(5))  # list of int, 0 to 4
        state = [{'a': i * 2.5, 'b': i * 2.5} for i in range(5)]
        lazy_result = LazySimResult(f, time, state)  # Ordinary LazySimResult with f, time, state
        sim_result = SimResult(time, state)  # Ordinary SimResult with time,state

        converted_lazy_result = SimResult(lazy_result.times, lazy_result.data)
        self.assertFalse(sim_result.frame.equals(converted_lazy_result))
        self.assertNotEqual(sim_result, converted_lazy_result)  # converted is not the same as the original SimResult

        pickle.dump(lazy_result, open('model_test.pkl', 'wb'))
        pickle_converted_result = pickle.load(open('model_test.pkl', 'rb'))
        self.assertEqual(converted_lazy_result, pickle_converted_result)
        self.assertTrue(converted_lazy_result.frame.equals(pickle_converted_result.frame))

    def test_index(self):
        # Variables
        time = list(range(5))  # list of int, 0 to 4
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
        result = SimResult(time, state)

        self.assertEqual(result.index({'a': 10, 'b': 20}), 4)
        self.assertEqual(result.index({'a': 2.5, 'b': 5}), 1)
        self.assertEqual(result.index({'a': 0, 'b': 0}), 0)
        self.assertRaises(ValueError, result.index, 6.0)  # Other argument doesn't exist
        self.assertRaises(ValueError, result.index, -1)  # Non-existent data value
        self.assertRaises(ValueError, result.index, "7.5")  # Data specified incorrectly as string
        self.assertRaises(ValueError, result.index,
                          None)  # Not type errors because its simply looking for an object in list
        self.assertRaises(ValueError, result.index, [1, 2])
        self.assertRaises(ValueError, result.index, {})
        self.assertRaises(ValueError, result.index, set())

    def test_pop(self):
        # Variables
        time = list(map(float, range(5)))
        state = [{'a': i * 2.5, 'b': i * 5.0} for i in range(5)]
        result = SimResult(time, state)
        result_df = pd.DataFrame(state)
        result_df.insert(0, "time", time)

        result.pop(2)  # Test specified index
        state.remove({'a': 5.0, 'b': 10})  # update state by removing value
        self.assertEqual(result.data, state)
        # removing row from DataFrame
        temp_df = result_df.T
        temp_df.pop(2)
        result_df = temp_df.T.reset_index(drop=True)
        self.assertTrue(result.frame.equals(result_df))
        result.pop()  # Test default index -1 (last element)
        state.pop()  # pop state, removes last item
        # removing row from DataFrame
        temp_df = result_df.T
        temp_df.pop(3)
        result_df = temp_df.T.reset_index(drop=True)
        self.assertEqual(result.data, state)
        self.assertTrue(result.frame.equals(result_df))
        result.pop(-1)  # Test argument of index -1 (last element)
        state.pop()  # pop state, removes last item
        # removing row from DataFrame
        temp_df = result_df.T
        temp_df.pop(2)
        result_df = temp_df.T.reset_index(drop=True)
        self.assertEqual(result.data, state)
        self.assertTrue(result.frame.equals(result_df))
        result.pop(0)  # Test argument of 0
        state.pop(0)  # pop state, removes first item
        # removing row from DataFrame
        temp_df = result_df.T
        temp_df.pop(0)
        result_df = temp_df.T.reset_index(drop=True)
        self.assertTrue(result.frame.equals(result_df))
        self.assertEqual(result.data, state)
        self.assertRaises(IndexError, result.pop, 5)  # Test specifying an invalid index value
        self.assertRaises(IndexError, result.pop, 3)
        self.assertRaises(TypeError, result.pop, "5")  # Test specifying an invalid index type
        self.assertRaises(TypeError, result.pop, [0, 1])
        self.assertRaises(TypeError, result.pop, {})
        self.assertRaises(TypeError, result.pop, set())
        self.assertRaises(TypeError, result.pop, 1.5)

    def test_to_numpy(self):
        # Variables
        time = list(range(10))  # list of int, 0 to 9
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(10)]
        result = SimResult(time, state)
        np_result = result.to_numpy()
        self.assertIsInstance(np_result, np.ndarray)
        self.assertEqual(np_result.shape, (10, 2))
        self.assertEqual(np_result.dtype, np.dtype('float64'))
        self.assertTrue(np.all(np_result == np.array([[i * 2.5, i * 5] for i in range(10)])))

        # Subset of keys
        result = result.to_numpy(['b'])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 1))
        self.assertEqual(result.dtype, np.dtype('float64'))
        self.assertTrue(np.all(result == np.array([[i * 5] for i in range(10)])))

        # Now test when empty
        result = SimResult([], [])
        result = result.to_numpy()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 0))
        self.assertEqual(result.dtype, np.dtype('float64'))

        # Now test with StateContainer
        state = [DictLikeMatrixWrapper(['a', 'b'], x) for x in state]
        result = SimResult(time, state)
        result = result.to_numpy()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 2))
        self.assertEqual(result.dtype, np.dtype('float64'))
        self.assertTrue(np.all(result == np.array([[i * 2.5, i * 5] for i in range(10)])))

    def test_remove(self):
        # Variables
        time = list(range(5))  # list of int, 0 to 4
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
        result = SimResult(time, state)
        result_df = pd.DataFrame(state)
        result_df.insert(0, "time", time)

        result.remove({'a': 5.0, 'b': 10})  # Positional defaults to removing data
        # Update Variables
        time.remove(2)
        state.remove({'a': 5.0, 'b': 10})
        temp_df = result_df.T
        temp_df.pop(2)
        result_df = temp_df.T.reset_index(drop=True)
        self.assertTrue(result_df.equals(result.frame))
        self.assertEqual(result.times, time)
        self.assertEqual(result.data, state)
        result.remove(d={'a': 0.0, 'b': 0})  # Testing named removal of data
        # Update Variables
        time.remove(0)
        state.remove({'a': 0.0, 'b': 0})
        temp_df = result_df.T
        temp_df.pop(0)
        result_df = temp_df.T.reset_index(drop=True)
        self.assertTrue(result_df.equals(result.frame))
        self.assertEqual(result.times, time)
        self.assertEqual(result.data, state)
        result.remove(t=3)  # Testing named removal of time
        # Update Variables
        time.remove(3)
        state.remove({'a': 7.5, 'b': 15})
        temp_df = result_df.T
        temp_df.pop(1)
        result_df = temp_df.T.reset_index(drop=True)
        self.assertTrue(result_df.equals(result.frame))
        self.assertEqual(result.times, time)
        self.assertEqual(result.data, state)
        result.remove(t=1)
        # Update Variables
        time.remove(1)
        state.remove({'a': 2.5, 'b': 5})
        temp_df = result_df.T
        temp_df.pop(0)
        result_df = temp_df.T.reset_index(drop=True)
        self.assertTrue(result_df.equals(result.frame))
        self.assertEqual(result.times, time)
        self.assertEqual(result.data, state)

        self.assertRaises(ValueError, result.remove, )  # If nothing specified, raise ValueError
        self.assertRaises(ValueError, result.remove, None, None)  # Passing both as None
        self.assertRaises(ValueError, result.remove, 0.0, 1)  # Passing arguments to both
        self.assertRaises(ValueError, result.remove, 7.5)  # Test nonexistent data value
        self.assertRaises(ValueError, result.remove, -1)  # Type checking negated as index searches for element in list
        self.assertRaises(ValueError, result.remove, "5")  # Thus all value types allowed to be searched
        self.assertRaises(ValueError, result.remove, [0, 1])
        self.assertRaises(ValueError, result.remove, {})
        self.assertRaises(ValueError, result.remove, set())

    def test_clear(self):
        # Variables
        time = list(range(5))  # list of int, 0 to 4
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
        result = SimResult(time, state)
        result_df = pd.DataFrame(state)
        result_df.insert(0, "time", time)
        self.assertEqual(result.times, time)
        self.assertEqual(result.data, state)
        self.assertTrue(result.frame.equals(result_df))
        self.assertRaises(TypeError, result.clear, True)

        result.clear()
        self.assertTrue(result.frame.equals(pd.DataFrame()))
        self.assertEqual(result.times, [])
        self.assertEqual(result.data, [])

    def test_get_time(self):
        # Variables
        # Creating two result objects
        time = list(range(5))  # list of int, 0 to 4
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
        result = SimResult(time, state)
        self.assertEqual(result.get_time(0), result.times[0])
        self.assertEqual(result.get_time(1), result.times[1])
        self.assertEqual(result.get_time(2), result.times[2])
        self.assertEqual(result.get_time(3), result.times[3])
        self.assertEqual(result.get_time(4), result.times[4])

        self.assertRaises(TypeError, result.get_time, )  # Test no input given
        self.assertRaises(TypeError, result.get_time, "0")  # Tests specifying an invalid index type
        self.assertRaises(TypeError, result.get_time, [0, 1])
        self.assertRaises(TypeError, result.get_time, {})
        self.assertRaises(TypeError, result.get_time, set())
        self.assertRaises(TypeError, result.get_time, 1.5)

    def test_plot(self):
        # Testing model taken from events.py
        YELLOW_THRESH, RED_THRESH, THRESHOLD = 0.15, 0.1, 0.05

        class MyBatt(BatteryElectroChemEOD):
            events = BatteryElectroChemEOD.events + ['EOD_warn_yellow', 'EOD_warn_red', 'EOD_requirement_threshold']

            def event_state(self, state):
                event_state = super().event_state(state)
                event_state['EOD_warn_yellow'] = (event_state['EOD'] - YELLOW_THRESH) / (1 - YELLOW_THRESH)
                event_state['EOD_warn_red'] = (event_state['EOD'] - RED_THRESH) / (1 - RED_THRESH)
                event_state['EOD_requirement_threshold'] = (event_state['EOD'] - THRESHOLD) / (1 - THRESHOLD)
                return event_state

            def threshold_met(self, x):
                t_met = super().threshold_met(x)
                event_state = self.event_state(x)
                t_met['EOD_warn_yellow'] = event_state['EOD_warn_yellow'] <= 0
                t_met['EOD_warn_red'] = event_state['EOD_warn_red'] <= 0
                t_met['EOD_requirement_threshold'] = event_state['EOD_requirement_threshold'] <= 0
                return t_met

        def future_loading(t, x=None):
            if (t < 600):
                i = 2
            elif (t < 900):
                i = 1
            elif (t < 1800):
                i = 4
            elif (t < 3000):
                i = 2
            else:
                i = 3
            return {'i': i}

        m = MyBatt()
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_loading, threshold_keys=['EOD'],
                                                                                 print=False)
        plot_test = event_states.plot()  # Plot doesn't raise error

    def test_namedtuple_access(self):
        # Testing model taken from events.py
        YELLOW_THRESH, RED_THRESH, THRESHOLD = 0.15, 0.1, 0.05

        class MyBatt(BatteryElectroChemEOD):
            events = BatteryElectroChemEOD.events + ['EOD_warn_yellow', 'EOD_warn_red', 'EOD_requirement_threshold']

            def event_state(self, state):
                event_state = super().event_state(state)
                event_state['EOD_warn_yellow'] = (event_state['EOD'] - YELLOW_THRESH) / (1 - YELLOW_THRESH)
                event_state['EOD_warn_red'] = (event_state['EOD'] - RED_THRESH) / (1 - RED_THRESH)
                event_state['EOD_requirement_threshold'] = (event_state['EOD'] - THRESHOLD) / (1 - THRESHOLD)
                return event_state

            def threshold_met(self, x):
                t_met = super().threshold_met(x)
                event_state = self.event_state(x)
                t_met['EOD_warn_yellow'] = event_state['EOD_warn_yellow'] <= 0
                t_met['EOD_warn_red'] = event_state['EOD_warn_red'] <= 0
                t_met['EOD_requirement_threshold'] = event_state['EOD_requirement_threshold'] <= 0
                return t_met

        def future_loading(t, x=None):
            if (t < 600):
                i = 2
            elif (t < 900):
                i = 1
            elif (t < 1800):
                i = 4
            elif (t < 3000):
                i = 2
            else:
                i = 3
            return {'i': i}

        m = MyBatt()
        named_results = m.simulate_to_threshold(future_loading, threshold_keys=['EOD'], print=False)
        times = named_results.times
        inputs = named_results.inputs
        states = named_results.states
        outputs = named_results.outputs
        event_states = named_results.event_states

    def test_not_implemented(self):
        # Not implemented functions, should raise errors
        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(NUM_ELEMENTS)]
        result = SimResult(time, state)
        self.assertRaises(NotImplementedError, result.append)
        self.assertRaises(NotImplementedError, result.count)
        self.assertRaises(NotImplementedError, result.insert)
        self.assertRaises(NotImplementedError, result.reverse)

    # Tests for LazySimResult
    def test_lazy_data_fcn(self):
        def f(x):
            return {k: v * 2 for k, v in x.items()}

        # Variables
        time = list(range(5))
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
        state2 = [{'a': i * 5.0, 'b': i * 10} for i in range(5)]
        result = LazySimResult(f, time, state)
        result_df = pd.DataFrame(state)
        result_df.insert(0, "time", time)
        result_df2 = pd.DataFrame(state2)
        result_df2.insert(0, "time", time)

        self.assertFalse(result.is_cached())
        self.assertEqual(result.data, state2)
        self.assertTrue(result_df.equals(result.frame))
        self.assertTrue(result_df2.equals(result.get_frame_data()))
        self.assertTrue(result.is_cached())

    def test_lazy_clear(self):
        def f(x):
            return {k: v * 2 for k, v in x.items()}

        # Variables
        time = list(range(5))  # list of int, 0 to 4
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
        state2 = [{'a': i * 5.0, 'b': i * 10} for i in range(5)]
        result_df = pd.DataFrame(state)
        result_df.insert(0, "time", time)
        result_df2 = pd.DataFrame(state2)
        result_df2.insert(0, "time", time)
        result = LazySimResult(f, time, state)
        self.assertEqual(result.times, time)
        self.assertEqual(result.data, state2)
        self.assertEqual(result.states, state)
        self.assertTrue(result_df.equals(result.frame))
        self.assertTrue(result_df2.equals(result.get_frame_data()))
        self.assertRaises(TypeError, result.clear, True)

        result.clear()
        self.assertEqual(result.times, [])
        self.assertEqual(result.data, [])
        self.assertEqual(result.states, [])
        self.assertTrue(result.frame.equals(pd.DataFrame()))
        self.assertTrue(result.get_frame_data().equals(pd.DataFrame()))

    def test_lazy_extend(self):
        def f(x):
            return {k: v * 2 for k, v in x.items()}

        # Variables
        time = list(range(5))  # list of int, 0 to 4
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
        result = LazySimResult(f, time, state)
        time2 = list(range(10))  # list of int, 0 to 9
        state2 = [{'a': i * 5, 'b': i * 10} for i in range(10)]
        data2 = [{'a': i * 25, 'b': i * 50} for i in range(10)]
        data = [{'a': i * 5.0, 'b': i * 10} for i in range(5)]
        result_df = pd.DataFrame(state)
        result_df.insert(0, "time", time)
        result_df2 = pd.DataFrame(state2)
        result_df2.insert(0, "time", time2)
        result_data_df = pd.DataFrame(data)
        result_data_df.insert(0, "time", time)
        result_data_df2 = pd.DataFrame(data2)
        result_data_df2.insert(0, "time", time2)
        result_df_extended = pd.DataFrame(state+state2)
        result_df_extended.insert(0, "time", time+time2)
        result_data_df_extended = pd.DataFrame(data+data2)
        result_data_df_extended.insert(0, "time", time+time2)

        def f2(x):
            return {k: v * 5 for k, v in x.items()}

        result2 = LazySimResult(f2, time2, state2)
        self.assertEqual(result.times, time)  # Assert data is correct before extending
        self.assertEqual(result.data, data)
        self.assertEqual(result.states, state)
        self.assertEqual(result2.times, time2)
        self.assertEqual(result2.data, data2)
        self.assertEqual(result2.states, state2)
        self.assertTrue(result.frame.equals(result_df))
        self.assertTrue(result.get_frame_data().equals(result_data_df))
        self.assertTrue(result2.frame.equals(result_df2))
        self.assertTrue(result2.get_frame_data().equals(result_data_df2))

        result.extend(result2)
        self.assertEqual(result.times, time + time2)  # Assert data is correct after extending
        self.assertEqual(result.data, data + data2)
        self.assertEqual(result.states, state + state2)
        self.assertTrue(result.frame.equals(result_df_extended))
        self.assertTrue(result.get_frame_data().equals(result_data_df_extended))

    def test_lazy_extend_cache(self):
        def f(x):
            return {k: v * 2 for k, v in x.items()}

        # Variables
        time = list(range(5))
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
        data = [{'a': i * 5.0, 'b': i * 10} for i in range(5)]
        result1 = LazySimResult(f, time, state)
        result2 = LazySimResult(f, time, state)
        result_df = pd.DataFrame(state)
        result_df.insert(0, "time", time)
        result_data_df = pd.DataFrame(data)
        result_data_df.insert(0, "time", time)
        result_df_ext = pd.DataFrame(state+state)
        result_df_ext.insert(0, "time", time+time)
        result_data_df_ext = pd.DataFrame(data+data)
        result_data_df_ext.insert(0, "time", time+time)

        # Case 1
        result1.extend(result2)
        self.assertFalse(result1.is_cached())  # False

        # Case 2
        result1 = LazySimResult(f, time, state)  # Reset result1
        self.assertTrue(result1.frame.equals(result_df))
        store_test_data = result1.data  # Access result1 data
        result1.extend(result2)
        self.assertTrue(result1.frame.equals(result_df_ext))
        self.assertFalse(result1.is_cached())  # False

        # Case 3
        result1 = LazySimResult(f, time, state)  # Reset result1
        store_test_data = result2.data  # Access result2 data
        result1.extend(result2)
        self.assertFalse(result1.is_cached())  # False

        # Case 4
        result1 = LazySimResult(f, time, state)  # Reset result1
        result2 = LazySimResult(f, time, state)  # Reset result2
        store_test_data1 = result1.data  # Access result1 data
        store_test_data2 = result2.data  # Access result2 data
        result1.extend(result2)
        self.assertTrue(result1.is_cached())  # True
        self.assertTrue(result1.get_frame_data().equals(result_data_df_ext))

    def test_lazy_extend_error(self):
        def f(x):
            return {k: v * 2 for k, v in x.items()}

        # Variables
        time = list(range(5))  # list of int, - to 4
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
        result = LazySimResult(f, time, state)
        sim_result = SimResult(time, state)

        self.assertRaises(ValueError, result.extend, sim_result)  # Passing a SimResult to LazySimResult's extend
        self.assertRaises(ValueError, result.extend, 0)  # Passing non-LazySimResult types to extend method
        self.assertRaises(ValueError, result.extend, [0, 1])
        self.assertRaises(ValueError, result.extend, {})
        self.assertRaises(ValueError, result.extend, set())
        self.assertRaises(ValueError, result.extend, 1.5)

    def test_lazy_pop(self):
        def f(x):
            return {k: v * 2 for k, v in x.items()}

        # Variables
        time = list(map(float, range(5)))  # list of int, 0 to 4
        state = [{'a': i * 2.5, 'b': i * 5.0} for i in range(5)]
        data = [{'a': i * 5.0, 'b': i * 10.0} for i in range(5)]
        result = LazySimResult(f, time, state)
        result_df = pd.DataFrame(state)
        result_df.insert(0, "time", time)
        result_data_df = pd.DataFrame(data)
        result_data_df.insert(0, "time", time)

        result.pop(1)  # Test specified index
        time.remove(1)  # remove value '1' to check time values after pop
        # removing row from DataFrame
        temp_df = result_df.T
        temp_df.pop(1)
        result_df = temp_df.T.reset_index(drop=True)
        self.assertTrue(result.frame.equals(result_df))
        self.assertEqual(result.times, time)
        # removing row from fcn DataFrame
        temp_df = result_data_df.T
        temp_df.pop(1)
        result_data_df = temp_df.T.reset_index(drop=True)
        self.assertTrue(result.get_frame_data().equals(result_data_df))
        data.remove({'a': 5.0, 'b': 10})  # removes index 1 value from data list
        self.assertEqual(result.data, data)
        state.remove({'a': 2.5, 'b': 5})  # removes index 1 value from state list
        self.assertEqual(result.states, state)
        # removing row from fcn DataFrame
        self.assertTrue(result.get_frame_data().equals(result_data_df))

        result.pop()  # Test default index -1 (last element)
        time.pop()
        data.pop()
        state.pop()
        # removing row from fcn DataFrame
        temp_df = result_data_df.T
        temp_df.pop(3)
        result_data_df = temp_df.T.reset_index(drop=True)
        self.assertEqual(result.times, time)
        self.assertEqual(result.data, data)
        self.assertEqual(result.states, state)
        self.assertTrue(result.get_frame_data().equals(result_data_df))

        result.pop(-1)  # Test argument of index -1 (last element)
        time.pop(-1)
        data.pop(-1)
        state.pop(-1)
        # removing row from fcn DataFrame
        temp_df = result_data_df.T
        temp_df.pop(2)
        result_data_df = temp_df.T.reset_index(drop=True)
        self.assertEqual(result.times, time)
        self.assertEqual(result.data, data)
        self.assertEqual(result.states, state)
        self.assertTrue(result.get_frame_data().equals(result_data_df))
        result.pop(0)  # Test argument of 0
        time.pop(0)
        data.pop(0)
        state.pop(0)
        # removing row from fcn DataFrame
        temp_df = result_data_df.T
        temp_df.pop(0)
        result_data_df = temp_df.T.reset_index(drop=True)
        self.assertTrue(result.get_frame_data().equals(result_data_df))
        self.assertEqual(result.times, time)
        self.assertEqual(result.data, data)
        self.assertEqual(result.states, state)
        # Test erroneous input
        self.assertRaises(IndexError, result.pop, 5)  # Test specifying an invalid index value
        self.assertRaises(IndexError, result.pop, 3)
        self.assertRaises(TypeError, result.pop, "5")  # Test specifying an invalid index type
        self.assertRaises(TypeError, result.pop, [0, 1])
        self.assertRaises(TypeError, result.pop, {})
        self.assertRaises(TypeError, result.pop, set())
        self.assertRaises(TypeError, result.pop, 1.5)

    def test_cached_sim_result(self):
        def f(x):
            return {k: v * 2 for k, v in x.items()}

        NUM_ELEMENTS = 5
        time = list(range(NUM_ELEMENTS))
        state = [{'a': i * 2.5, 'b': i * 5.0} for i in range(NUM_ELEMENTS)]
        data = [{'a': i * 5.0, 'b': i * 10.0} for i in range(NUM_ELEMENTS)]
        result_df = pd.DataFrame(state)
        result_df.insert(0, "time", time)
        result_data_df = pd.DataFrame(data)
        result_data_df.insert(0, "time", time)
        result = LazySimResult(f, time, state)
        result_df_ext = pd.DataFrame(state+state)
        result_df_ext.insert(0, "time", time+time)
        self.assertFalse(result.is_cached())
        self.assertListEqual(result.times, time)
        for i in range(5):
            self.assertEqual(result.get_time(i), time[i])
            self.assertEqual(result[i], {k: v * 2 for k, v in state[i].items()})
        self.assertTrue(result.is_cached())

        try:
            tmp = result[NUM_ELEMENTS]
            self.fail("Should be out of range error")
        except IndexError:
            pass

        try:
            tmp = result.get_time(NUM_ELEMENTS)
            self.fail("Should be out of range error")
        except IndexError:
            pass

        # Catch bug that occurred where lazysimresults weren't actually different
        # This occurred because the underlying arrays of time and state were not copied (see PR #158)
        result = LazySimResult(f, time, state)
        result2 = LazySimResult(f, time, state)
        self.assertTrue(result == result2)
        self.assertTrue(result.frame.equals(result2.frame))
        self.assertEqual(len(result), len(result2))
        result.extend(LazySimResult(f, time, state))
        self.assertFalse(result == result2)
        self.assertNotEqual(len(result), len(result2))
        self.assertTrue(result.frame.equals(result_df_ext))

    def test_lazy_remove(self):
        def f(x):
            return {k: v * 2 for k, v in x.items()}

        # Variables
        time = list(range(10))  # list of int, 0 to 9
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(10)]
        result = LazySimResult(f, time, state)
        data = [{'a': i * 5.0, 'b': i * 10} for i in range(10)]
        result_df = pd.DataFrame(state)
        result_df.insert(0, "time", time)
        result_data_df = pd.DataFrame(state)
        result_data_df.insert(0, "time", time)

        result.remove({'a': 5.0, 'b': 10})  # Unnamed default positional argument removal of data value
        # Update Variables
        state.remove({'a': 2.5, 'b': 5})
        time.remove(1)
        data.remove({'a': 5.0, 'b': 10})
        # removing row from DataFrame
        temp_df = result_df.T
        temp_df.pop(1)
        result_df = temp_df.T.reset_index(drop=True)
        self.assertTrue(result.frame.equals(result_df))
        self.assertEqual(result.times, time)
        self.assertEqual(result.data, data)
        self.assertEqual(result.states, state)
        result.remove(d={'a': 0.0, 'b': 0})  # Named argument removal of data value
        # Update Variables
        state.remove({'a': 0.0, 'b': 0})
        time.remove(0)
        data.remove({'a': 0.0, 'b': 0})
        # removing row from DataFrame
        temp_df = result_df.T
        temp_df.pop(0)
        result_df = temp_df.T.reset_index(drop=True)
        self.assertTrue(result.frame.equals(result_df))
        self.assertEqual(result.times, time)
        self.assertEqual(result.data, data)
        self.assertEqual(result.states, state)
        result.remove(t=7)  # Named argument removal of times value
        # Update Variables
        state.remove({'a': 17.5, 'b': 35})
        time.remove(7)
        data.remove({'a': 35.0, 'b': 70})
        # removing row from DataFrame
        temp_df = result_df.T
        temp_df.pop(5)
        result_df = temp_df.T.reset_index(drop=True)
        self.assertTrue(result.frame.equals(result_df))
        self.assertEqual(result.times, time)
        self.assertEqual(result.data, data)
        self.assertEqual(result.states, state)
        result.remove(s={'a': 12.5, 'b': 25})  # Named argument removal of states value
        # Update Variables
        state.remove({'a': 12.5, 'b': 25})
        time.remove(5)
        data.remove({'a': 25, 'b': 50})
        # removing row from DataFrame
        temp_df = result_df.T
        temp_df.pop(3)
        result_df = temp_df.T.reset_index(drop=True)
        self.assertTrue(result.frame.equals(result_df))
        self.assertEqual(result.times, time)
        self.assertEqual(result.data, data)
        self.assertEqual(result.states, state)

        self.assertRaises(ValueError, result.remove, )  # Test no values specified
        self.assertRaises(ValueError, result.remove, 90.0, 2)  # Test two values specified positionally
        self.assertRaises(ValueError, result.remove, 90.0, 2, 15.0)  # Test three values specified positionally
        self.assertRaises(ValueError, result.remove, d=90.0, t=2)  # Test d,t values specified by name
        self.assertRaises(ValueError, result.remove, t=2, s=15.0)  # Test s,t values specified by name
        self.assertRaises(ValueError, result.remove, d=90.0, s=15.0)  # Test d,s values specified by name
        self.assertRaises(ValueError, result.remove, d=90.0, t=2, s=15.0)  # Test three values specified by name
        self.assertRaises(ValueError, result.remove, 90.0)  # Test nonexistent data value
        self.assertRaises(ValueError, result.remove, d=90.0)  # Test nonexistent data value
        self.assertRaises(ValueError, result.remove, t=90.0)  # Test nonexistent times value
        self.assertRaises(ValueError, result.remove, s=90.0)  # Test nonexistent states value
        self.assertRaises(ValueError, result.remove, -1)  # Type checking negated as index searches for element in list
        self.assertRaises(ValueError, result.remove, "5")  # Thus all value types allowed to be searched
        self.assertRaises(ValueError, result.remove, [0, 1])
        self.assertRaises(ValueError, result.remove, {})
        self.assertRaises(ValueError, result.remove, set())

    def test_lazy_not_implemented(self):
        # Not implemented functions, should raise errors
        def f(x):
            return {k: v * 2 for k, v in x.items()}

        # Variables
        time = list(range(5))  # list of int, 0 to 4
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
        result = LazySimResult(f, time, state)
        self.assertRaises(NotImplementedError, result.append)
        self.assertRaises(NotImplementedError, result.count)
        self.assertRaises(NotImplementedError, result.insert)
        self.assertRaises(NotImplementedError, result.reverse)

    def test_lazy_to_simresult(self):
        def f(x):
            return {k: v * 2 for k, v in x.items()}

        # Variables
        time = list(map(float, range(5)))  # list of int, 0 to 4
        state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
        data = [{'a': i * 5.0, 'b': i * 10} for i in range(5)]
        result_df = pd.DataFrame(state)
        result_df.insert(0, "time", time)
        result_data_df = pd.DataFrame(data)
        result_data_df.insert(0, "time", time)
        result = LazySimResult(f, time, state)

        converted_result = result.to_simresult()
        self.assertTrue(isinstance(converted_result, SimResult))  # Ensure type is SimResult
        self.assertEqual(converted_result.times, result.times)  # Compare to original LazySimResult
        self.assertEqual(converted_result.data, result.data)
        self.assertEqual(converted_result.times, time)  # Compare to expected values
        self.assertEqual(converted_result.data, data)
        self.assertTrue(converted_result.frame.equals(result.get_frame_data()))

    def test_monotonicity(self):
        # Variables
        time = list(range(5))

        # Test monotonically increasing, decreasing
        states = [{'a': 1 + i / 10, 'b': 2 - i / 5} for i in range(5)]
        result = SimResult(time, states)
        self.assertDictEqual(result.monotonicity(), {'a': 1.0, 'b': 1.0})
        self.assertTrue(result.monotonicity_df().equals(pd.DataFrame({'a': 1.0, 'b': 1.0}, index=[0])))

        # Test monotonicity between range [0,1]
        states = [{'a': i * (i % 3 - 1), 'b': i * (i % 3 - 1)} for i in range(5)]
        result = SimResult(time, states)
        self.assertDictEqual(result.monotonicity(), {'a': 0.25, 'b': 0.25})
        self.assertTrue(result.monotonicity_df().equals(pd.DataFrame({'a': 0.25, 'b': 0.25}, index=[0])))

        # # Test no monotonicity
        states = [{'a': i * (i % 2), 'b': i * (i % 2)} for i in range(5)]
        result = SimResult(time, states)
        self.assertDictEqual(result.monotonicity(), {'a': 0.0, 'b': 0.0})
        self.assertTrue(result.monotonicity_df().equals(pd.DataFrame(pd.DataFrame({'a': 0.0, 'b': 0.0}, index=[0]))))


# This allows the module to be executed directly
def run_tests():
    unittest.main()


def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Sim Result")
    result = runner.run(l.loadTestsFromTestCase(TestSimResult)).wasSuccessful()

    if not result:
        raise Exception("Failed test")


if __name__ == '__main__':
    main()
