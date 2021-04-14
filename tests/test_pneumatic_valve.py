# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from prog_models.models.pneumatic_valve import PneumaticValve

class TestPneumaticValve(unittest.TestCase):
    def test_pneumatic_valve(self):
        m = PneumaticValve(process_noise= 0)

        cycle_time = 20
        def future_loading(t, x=None):
            t = t % cycle_time
            if t < cycle_time/2:
                return {
                    'pL': 3.5e5,
                    'pR': 2.0e5,
                    # Open Valve
                    'uTop': False,
                    'uBot': True
                }
            else:
                return {
                    'pL': 3.5e5,
                    'pR': 2.0e5,
                    # Close Valve
                    'uTop': True,
                    'uBot': False
                }

        u0 = future_loading(0)
        x0 = m.initialize(u0)
        x0_test = {
            'x': 0,
            'v': 0,
            'Ai': 0,
            'r': 6000,
            'k': 48000,
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'condition': 1,
            'mTop': 0.067876043046,
            'mBot': 9.445596253581e-4,
            'pDiff': u0['pL'] - u0['pR'],
            'wb': 0,
            'wi': 0,
            'wk': 0,
            'wr': 0,
            'wt': 0
        }

        for key in m.states:
            self.assertAlmostEqual(x0[key], x0_test[key], 7)
        
        x = m.next_state(x0, future_loading(0), 0.1)
        x_test = {
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'Ai': 0,
            'condition': 1,
            'k': 48000,
            'mBot': 0.008534524658,
            'mTop': 0.048044198904,
            'r': 6000,
            'v': 0,
            'x': 0,
            'pDiff': u0['pL'] - u0['pR'],
            'wb': 0,
            'wi': 0,
            'wk': 0,
            'wr': 0,
            'wt': 0
        }
        for key in m.states:
            self.assertAlmostEqual(x[key], x_test[key], 7)

        z = m.output(x)
        z_test = {
            "Q": 0,
            "iB": True, 
            "iT": False,
            "pB": 0.91551734,
            "pT": 3.7319387914459899,
            "x": 0
        }
        for key in m.outputs:
            self.assertAlmostEqual(z[key], z_test[key], 7)

        m.parameters['x0']['wr'] = 1

        x0 = m.initialize(u0)
        x0_test = {
            'x': 0,
            'v': 0,
            'Ai': 0,
            'r': 6000,
            'k': 48000,
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'mTop': 0.067876043046,
            'mBot': 9.445596253581e-4,
            'pDiff': u0['pL'] - u0['pR'],
            'wb': 0,
            'wi': 0,
            'wk': 0,
            'wr': 1,
            'wt': 0
        }

        for key in m.states:
            self.assertAlmostEqual(x0[key], x0_test[key], 7)
        
        x = m.next_state(x0, future_loading(0), 0.1)
        x_test = {
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'Ai': 0,
            'k': 48000,
            'mBot': 8.5345246578678786736e-3,
            'mTop': 0.048044198903933490206,
            'r': 6000,
            'v': 0,
            'x': 0,
            'pDiff': u0['pL'] - u0['pR'],
            'wb': 0,
            'wi': 0,
            'wk': 0,
            'wr': 1,
            'wt': 0
        }
        for key in m.states:
            self.assertAlmostEqual(x[key], x_test[key], 7)

        config = {'dt': 0.01, 'horizon': 800, 'save_freq': 60}
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_loading, m.output(m.initialize(future_loading(0))), **config)# , 'save_freq': 60
        self.assertAlmostEqual(times[-1], 782.53, 0)