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

    # @unittest.skip
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
        self.assertEqual(
            'Model unstable- NAN reached in simulation (t=1800.0) before cutoff threshold. '
            'Cutoff threshold is 1900.0, or roughly 95.0% of the data',
            str(cm.exception)
        )
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

    # def test_MSE(self):
        # m = ThrownObject()
        # m2 = ThrownObject()
        # results = m.simulate_to_threshold(save_freq=0.5)
        # data = [(results.times, results.inputs, results.outputs)]
        # gt = m.parameters.copy()

    #     self.assertNotEqual(m.calc_error(results.times, results.inputs, results.outputs),
    #                         m2.calc_error(results.times, results.inputs, results.outputs))
        
    #     results = m2.simulate_to_threshold(save_freq = 0.5)

    #     self.assertEqual(m.calc_error(results.times, results.inputs, results.outputs),
    #                         m2.calc_error(results.times, results.inputs, results.outputs))
    def test_RMSE(self):
        m = ThrownObject()
        m2 = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        data = [(results.times, results.inputs, results.outputs)]
        gt = m.parameters.copy()


    def test_MAX_E(self):
        return

    def test_MAE(self):
        return

    def test_MAPE(self):
        return
    
    def test_DTW(self):
        m = LinearThrownObject()
        results = m.simulate_to_threshold(save_freq = 0.5)

        times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
        inputs = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        outputs = [{'x': 2.83}, {'x': 22.868664648480163}, {'x': 40.422881456712254}, {'x': 55.51861687290881}, {'x': 68.06865643702567}, 
                   {'x': 78.16641111234323}, {'x': 85.89550327176332}, {'x': 91.18868647545982}, {'x': 94.01376508127296}, 
                   {'x': 94.31711597903195}, {'x': 92.22337598299377}, {'x': 87.67210201789473}, {'x': 80.62858869729064}, {'x': 71.10796509926787}, 
                   {'x': 59.19579056829866}, {'x': 44.79567793740186}, {'x': 27.97245305860176}, {'x': 8.736607826437163}, {'x': -12.879687324031048}]
        # Does not work readily with a LinearThrown Object

        hold = m.calc_error(times, inputs, outputs, method = 'dtw')

        self.assertEqual(m.calc_error(times, inputs, outputs, method = 'dtw'), 4.8146507570483195)

        hold2 = m.calc_error(results.times, results.inputs, results.outputs, method = 'MAE')

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
    
        results = m.simulate_to(2000, future_loading, **options)

        # Does not work readily with a LinearThrown Object
        hold = m.calc_error(results.times, results.inputs, results.outputs, method = 'dtw', dt=1)

def run_tests():
    unittest.main()
    

def main():
    import cProfile, pstats
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Base Models")
    profiler = cProfile.Profile()

    profiler.enable()
    result = runner.run(l.loadTestsFromTestCase(TestCalcError)).wasSuccessful()
    profiler.disable()

    with open("output_time.txt", 'w') as f:
        p = pstats.Stats(profiler, stream=f)
        p.sort_stats("time").print_stats()

    with open("output_calls.txt", 'w') as f:
        p = pstats.Stats(profiler, stream=f)
        p.sort_stats("calls").print_stats()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
