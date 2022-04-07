# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.
from prog_models import PrognosticsModel
from math import pi, floor
import scipy.signal as signal

# selection of switching pattern based on rotor position (theta)
def commutation(theta):
    theta = theta * (180/pi) # convert rad to deg
    CL = [[1, -1, 0],
           [1, 0, -1],
           [0, 1, -1],
           [-1, 1, 0],
           [-1, 0, 1],
           [0, -1, 1]
        ]
    return CL[floor(theta/60)%6]


class ESC(PrognosticsModel):
    default_parameters = {
        'sawtooth_freq': 16000, # Hz

        # Motor Parameters
        'x0': {
            'v_a': 0,
            'v_b': 0,
            'v_c': 0,
            't': 0
        }
    }

    states = ['v_a', 'v_b', 'v_c', 't']
    inputs = ['duty', 'theta', 'v']
    outputs = states

    def initialize(self, u = None, z = None):
        return self.StateContainer(self.parameters['x0'])

    def next_state(self, x, u, dt):
        pw = max(signal.square(2 * pi * self.parameters['sawtooth_freq'] * x['t'], duty=u['duty']), 0)
        V = pw*u['v']
        SP = commutation(u['theta'])
        VP = [V * SP[i] for i in range(3)]
        return self.StateContainer({
            'v_a': VP[0],
            'v_b': VP[1],
            'v_c': VP[2],
            't': x['t'] + dt
        })

    def output(self, x):
        return self.OutputContainer(x)
