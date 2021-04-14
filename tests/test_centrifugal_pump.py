# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from prog_models.models.centrifugal_pump import CentrifugalPump

class TestCentrifugalPump(unittest.TestCase):
    def test_centrifugal_pump(self):
        pump = CentrifugalPump(process_noise= 0)

        cycle_time = 3600
        def future_loading(t, x=None):
            t = t % cycle_time
            if t < cycle_time/2.0:
                V = 471.2389
            elif t < cycle_time/2 + 100:
                V = 471.2389 + (t-cycle_time/2)
            elif t < cycle_time - 100:
                V = 571.2389
            else:
                V = 471.2398 - (t-cycle_time)

            return {
                'Tamb': 290,
                'V': V,
                'pdisch': 928654, 
                'psuc': 239179, 
                'wsync': V * 0.8
            }
        x0 = pump.initialize(future_loading(0))
        x0_test = {
            'w': 376.991118431,
            'wA': 0.00,
            'wRadial': 0.0,
            'wThrust': 0,
            'rThrust': 1.4e-6,
            'rRadial': 1.8e-6,
            'Tt': 290,
            'Tr': 290,
            'To': 290,
            'Q': 0.0,
            'A': 12.7084,
            'QLeak':  -8.303463132934355e-08
        }
        for key in pump.states:
            self.assertAlmostEqual(x0[key], x0_test[key], 7)
        
        x = pump.next_state(x0, future_loading(0), 1)
        x_test = {
            'w': 372.68973081274,
            'Q': 0.017417462,
            'Tt': 290.027256332,
            'Tr': 290.1065917275,
            'To': 290,
            'A': 12.7084,
            'rThrust': 1.4e-6,
            'rRadial': 1.8e-6,
            'wA': 0.0,
            'wThrust': 0,
            'wRadial': 0,
            'QLeak': -8.303463132934355e-08
        }
        for key in pump.states:
            self.assertAlmostEqual(x[key], x_test[key], 7)

        z = pump.output(x)
        z_test = {
            'Qout': 0.0174,
            'To': 290,
            'Tr': 290.1066,
            'Tt': 290.0273,
            'w': 372.6897
        }

        for key in pump.outputs:
            self.assertAlmostEqual(z[key], z_test[key], 4)

        pump.parameters['x0']['wA'] = 1e-2
        pump.parameters['x0']['wThrust'] = 1e-10
        (times, inputs, states, outputs, event_states) = pump.simulate_to_threshold(future_loading, pump.output(pump.initialize(future_loading(0),{})))
        self.assertAlmostEqual(times[-1], 23891)