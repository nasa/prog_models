# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
import unittest

from prog_models import *
from prog_models.models import *

class TestCalcError(unittest.TestCase):
    """
    Main Testing class for calc_error.

    Validating that values are correctly being passed into the new calc_error calls and that we are receiving expected results!
    """
    def test_calc_error(self):
        # Note, lowering time steps or increasing simulate threshold may cause this model to not run (takes too long)
        m = BatteryElectroChemEOD()

        options = {
            'save_freq': 200, # Frequency at which results are saved
            'dt': 1, # Time step
        }

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
            return m.InputContainer({'i': i})
    
        simulated_results = m.simulate_to(2000, future_loading, **options)

        # Running calc_error before setting incorrect parameters
        m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt=1)

        with self.assertWarns(UserWarning) as cm:
            m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt = 1, stability_tol = 10)
        self.assertEqual(
            "Configurable cutoff must be some float value in the domain (0, 1]."  
            " Received 10. Resetting value to 0.95",
            str(cm.warning)
        )

        # Initializing parameters to very erroneous values       
        m.parameters['qMax'] = 4000
        keys = ['qMax']

        # Before running estimate_params
        with self.assertRaises(ValueError):
            m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt=1)

        m.estimate_params([(simulated_results.times, simulated_results.inputs, simulated_results.outputs)], keys, dt=0.5)

        # After running estimate_params. Note that this would not change the outcome of the result
        with self.assertRaises(ValueError):
            m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt=1)    

        # Running through various dt values that could work
        for i in np.arange(0, 1, 0.1):
            with self.assertRaises(ValueError):
                m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt=i)

        for i in range(2, 10):
            with self.assertRaises(ValueError):
                m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt=i)
 
        # Creating duplicate model
        m1 = BatteryElectroChemEOD()

        orig_params = m1.parameters.copy()

        # Much bigger parameter initialization
        m.parameters['kp'] = m1.parameters['kp'] = 10000
        m.parameters['kn'] = m1.parameters['kn'] = 1000 
        m.parameters['qpMax'] = m1.parameters['qpMax'] = 4500
        m.parameters['qMax'] = m1.parameters['qMax'] = 9000
        keys = ['kp', 'kn', 'qpMax','qMax']

        change_params = m1.parameters.copy()

        simulated_results = m.simulate_to(2000, future_loading, **options)
        m1_sim_results = m1.simulate_to(2000, future_loading, **options)

        data = [(simulated_results.times, simulated_results.inputs, simulated_results.outputs)]
        data_m1 = [(m1_sim_results.times, m1_sim_results.inputs, m1_sim_results.outputs)]

        # Check out the warnings that are occurring here...
        # They are being spammed almost. Increasing save_frequency increases spam
        # Calling estimate_params does not change any of the parameters here because we are always accounting for exceptions...

        m.estimate_params(data, keys, method='Powell')
        m1.estimate_params(data_m1, keys, method='CG')

        updated_params = m1.parameters.copy()

        # Checking to make sure estimate_params actually changed values away from the original and to something else
        self.assertEqual(change_params, updated_params)
        self.assertNotEqual(orig_params, updated_params)
        
        with self.assertRaises(ValueError):
            m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt = 1)
        
        with self.assertRaises(ValueError):
            m1.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt = 1)

        # Checks to see if stability_tolerance throws error if model goes unstable after threshold
        with self.assertWarns(UserWarning) as cm:
            m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, 
                     dt = 1, stability_tol=0.7)
        self.assertEqual(
            'Model unstable- NaN reached in simulation (t=1800.0)',
            str(cm.warning)
        )

        with self.assertRaises(ValueError) as cm:
            m1.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, 
                     dt = 1, stability_tol=70)
        # self.assertEqual(
        #     'Model unstable- NAN reached in simulation (t=1800.0) before cutoff threshold. '
        #     'Cutoff threshold is 1900.0, or roughly 95.0% of the data',
        #     str(cm.exception)
        # )
        # Rerunning params estimate would not change the results
        m.estimate_params(data, keys, method='Powell')
        m1.estimate_params(data_m1, keys, method='CG')

        with self.assertRaises(ValueError):
            m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt = 1)
        
        with self.assertRaises(ValueError):
            m1.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt = 1)

        # Resetting parameters
        m.parameters['kp'] = m1.parameters['kp'] = 10000
        m.parameters['kn'] = m1.parameters['kn'] = 1000 
        m.parameters['qpMax'] = m1.parameters['qpMax'] = 4500
        m.parameters['qMax'] = m1.parameters['qMax'] = 4000
        m.estimate_params(data, keys, method='Powell', options={'maxiter': 250, 'disp': False})
        m1.estimate_params(data_m1, keys, method='CG', options={'maxiter': 250, 'disp': False})
        simulated_results = m.simulate_to(2000, future_loading, **options)
        m1_sim_results = m1.simulate_to(2000, future_loading, **options)

        data = [(simulated_results.times, simulated_results.inputs, simulated_results.outputs)]
        data_m1 = [(m1_sim_results.times, m1_sim_results.inputs, m1_sim_results.outputs)]

        converge1 = m1.parameters.copy()

        with self.assertRaises(ValueError):
            m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt = 1)
        
        with self.assertRaises(ValueError):
            m1.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt = 1)

        m.estimate_params(data, keys, method='Powell', options={'maxiter': 10000, 'disp': False})
        m1.estimate_params(data_m1, keys, method='CG', options={'maxiter': 10000, 'disp': False})

        converge2 = m1.parameters.copy()

        self.assertEqual(converge1, converge2)

        with self.assertRaises(ValueError):
            m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt = 1)
        
        with self.assertRaises(ValueError):
            m1.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt = 1)



    """
    Base tests that ensure first-level workability within calc_error
    """
    def test_MSE(self):
        m = ThrownObject()
        m2 = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        gt = m.parameters.copy()

        # Arbitrary Test to ensure that both models are behaving the same way
        self.assertEqual(m.calc_error(results.times, results.inputs, results.outputs),
                            m2.calc_error(results.times, results.inputs, results.outputs))
        
        resultsm2 = m2.simulate_to_threshold(save_freq = 0.5)
        m2.parameters['throwing_speed'] = 35
        key = ['throwing_speed']

        previous = m2.calc_error(resultsm2.times, resultsm2.inputs, resultsm2.outputs)

        self.assertNotEqual(m.calc_error(results.times, results.inputs, results.outputs),
                            previous)
        
        m2.estimate_params(times = resultsm2.times, inputs = resultsm2.inputs, outputs = resultsm2.outputs, keys = key, dt = 1)

        for i in key:
            # We can compare with gt because m and m2 originally were the same, and gt was a copy of m.
            self.assertAlmostEqual(m2.parameters[i], gt[i], 2)

        self.assertNotEqual(m.calc_error(results.times, results.inputs, results.outputs),
                    previous)
        
        self.assertLess(m2.calc_error(resultsm2.times, resultsm2.inputs, resultsm2.outputs), previous)

        # By changing the 
        for i in np.arange(0.1, 1, 0.1):
            m.calc_error(results.times, results.inputs, results.outputs, dt=i)

        for i in range(2, 10):
            m.calc_error(results.times, results.inputs, results.outputs, dt=i)

        # Unique Tests for calc_error
        m.parameters['throwing_speed'] = 41
        
        with self.assertRaises(ValueError) as cm:
            m.calc_error(results.times, results.inputs, results.outputs, dt = 0)
        self.assertEqual(
            'Keyword argument \'dt\' must a initialized to a value greater than 0. Currently passed in 0',
            str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            m.calc_error(results.times, results.inputs, results.outputs, dt = {1})
        self.assertEqual(
            'Keyword argument \'dt\' must be either a int, float, or double.',
            str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            m.calc_error(results.times, results.inputs, results.outputs, x0 = 1)
        self.assertEqual(
            "Keyword argument 'x0' must be initialized to a Dict or StateContainer, not a int.",
            str(cm.exception)
        )

        with self.assertWarns(UserWarning) as cm:
            m.calc_error(results.times, results.inputs, results.outputs, 
                     dt = 1, stability_tol=70)
        self.assertEqual(
            'Configurable cutoff must be some float value in the domain (0, 1]. Received 70. Resetting value to 0.95',
            str(cm.warning)
        )

        with self.assertRaises(TypeError) as cm:
            m.calc_error(results.times, results.inputs, results.outputs, stability_tol = {1})
        self.assertEqual(
            "Keyword argument 'stability_tol' must be either a int, float, or double.",
            str(cm.exception)
        )


    def test_multiple(self):
        m = ThrownObject()

        # The value of time1, time2, inputs, and outputs are arbitrary values

        times = [[0, 1, 2, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3]]
        inputs = [[{}]*9, [{}]*4]
        outputs = [[{'x': 1.83},
            {'x': 36.95},
            {'x': 62.36},
            {'x': 77.81},
            {'x': 83.45},
            {'x': 79.28},
            {'x': 65.3},
            {'x': 41.51},
            {'x': 7.91},],
            [
                {'x': 1.83},
                {'x': 36.95},
                {'x': 62.36},
                {'x': 77.81},
            ]]
        
        m.calc_error(times, inputs, outputs)

        # Will be testing length errors, will also be providing a 'runs' feedback whenever there are multiple give?
        incorrectTimes = [[0, 1, 2, 4, 5, 6, 7, 8, 9]]

        with self.assertRaises(ValueError) as cm:
            m.calc_error(incorrectTimes, inputs, outputs)
        self.assertEqual(
            "Times, inputs, and outputs must all be the same length. Current lengths: times = 1, inputs = 2, outputs = 2",
            str(cm.exception)
        )

        incorrectTimes = [[0, 1, 2, 4, 5, 6, 7, 8], [0, 1, 2, 3]]

        with self.assertRaises(ValueError) as cm:
            m.calc_error(incorrectTimes, inputs, outputs)
        self.assertEqual(
            "Times, inputs, and outputs must all be the same length. Current lengths at run 0: times = 8, inputs = 9, outputs = 9",
            str(cm.exception)
        )

        incorrectTimes = [[0, 1, 2, 4, 5, 6, 7, 8, 9], [0, 1, 2]]

        with self.assertRaises(ValueError) as cm:
            m.calc_error(incorrectTimes, inputs, outputs)
        self.assertEqual(
            "Times, inputs, and outputs must all be the same length. Current lengths at run 1: times = 3, inputs = 4, outputs = 4",
            str(cm.exception)
        )

    def test_type_errors(self):
        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        gt = m.parameters.copy()

        with self.assertRaises(TypeError):
            m.calc_error()

        # With Strings
        with self.assertRaises(TypeError):
            m.calc_error("1, 2, 3", "1, 2, 3", "1, 2, 3")

        # With a list in a string
        with self.assertRaises(TypeError):
            m.calc_error(['1', '2', '3'], ['1', '2', '3'], ['1', '2', '3'])
    
        # Passing in bool values
        with self.assertRaises(TypeError) as cm:
            m.calc_error([False, False, True], [False, False, True], [False, False, True])
        self.assertEqual(
            "Data must be a dictionary or numpy array, not <class 'bool'>",
            str(cm.exception)
        )

        # Passing in tuples where, where inputs and outputs do not contain dicts or StateContainers, rather contain ints
        with self.assertRaises(TypeError) as cm:
            m.calc_error((1, 2, 3), (2, 3, 4), (3, 4, 5))
        self.assertEqual(
            "Data must be a dictionary or numpy array, not <class 'int'>",
            str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            m.calc_error({1, 2, 3}, ({'1': 1}, {'2': 2}, {'3': 3}), ({'1': 1}, {'2': 2}, {'3': 3}))
        self.assertEqual(
            "Types passed in must be from the following list: np.ndarray, list, SimResult, or LazySimResult. Current types: times = set, inputs = tuple, and outputs = tuple",
            str(cm.exception)
        )

        # Cannot add additional wrappers around inputs and outputs since they need to be dictionaries.
        with self.assertRaises(ProgModelTypeError) as cm:
            m.calc_error([1], [[{}]], [[{'1':1}]])
        self.assertEqual(
            "Data must be a dictionary or numpy array, not <class 'list'>",
            str(cm.exception)
        )        

        times = [1, [0, 1, 2, 3]]
        inputs = [{}, [{}]*4]
        outputs = [{'x': 1.83},
            [
                {'x': 1.83},
                {'x': 36.95},
                {'x': 62.36},
                {'x': 77.81},
            ]]

        with self.assertRaises(ValueError) as cm:
            m.calc_error(times, inputs, outputs)
        
        hold = m.calc_error([[1]], [[{}]], [[{'1':1}]])


        # Not support
        hold2 = m.calc_error([[[1]]], [[{}]], [[{'1':1}]])

        # Not support
        hold3 = m.calc_error([[[[1]]]], [[{}]], [[{'1':1}]])

        self.assertEqual(hold, hold2)
        self.assertEqual(hold2, hold3)

        with self.assertRaises(ProgModelTypeError) as cm:
            m.calc_error([[1]], [[[{}]]], [[[{'1':1}]]])
        self.assertEqual(
            "Data must be a dictionary or numpy array, not <class 'list'>",
            str(cm.exception)
        )

        # Expecting this to pass
        m.calc_error((1, 2, 3), ({'1': 1}, {'2': 2}, {'3': 3}), ({'1': 1}, {'2': 2}, {'3': 3}))


        
    def test_RMSE(self):
        return

    def test_MAX_E(self):
        return

    def test_MAE(self):
        return

    def test_MAPE(self):
        return

def run_tests():
    unittest.main()
    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting EstimateParams Feature")
    result = runner.run(l.loadTestsFromTestCase(TestCalcError)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    import cProfile
    cProfile.run('main()', "output.dat")

    import pstats

    with open("output_time.txt", 'w') as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("output_calls.txt", 'w') as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()
