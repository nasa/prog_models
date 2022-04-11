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
    """
    Simple Electronic-Speed Controller (ESC) model for powertrain modeling.
    This model replicates the behavior of the speed controller with pulse-width modulation (PWM) and commutation matrix.
    Duty cycle simulated with a square wave using scipy signal.square function. 
    

    References:
    Matteo Corbetta, Chetan S. Kulkarni. An approach for uncertainty quantification and management of unmanned aerial vehicle health. 
    Annual Conference of the PHM Society, Scottsdale, AZ, 2019. http://papers.phmsociety.org/index.php/phmconf/article/view/847

    George E. Gorospe Jr, Chetan S. Kulkarni, Edward Hogge, Andrew Hsu, and Natalie Ownby. A Study of the Degradation of Electronic Speed Controllers forBrushless DC Motors.
    Asia Pacific Conference of the Prognostics and Health Management Society, 2017. https://ntrs.nasa.gov/citations/20200000579

    Events: (0)
        | None
    
    Inputs/Loading: (3)
        | duty :        Duty cycle [-], percentage the input is "on" (i.e., voltage is supplied). 0 = no voltage supply (always closed), 1 = 100% voltage supply (always open).
        | theta :       rotor position [rad].
        | v :           voltage [V], voltage input from Battery (after DC converter, should be constant).

    States: (4)
        | v_a :         3-phase voltage value, first phase, [V], input to the motor
        | v_b :         3-phase voltage value, second phase, [V], input to the motor
        | v_c :         3-phase voltage value, third phase [V], input to the motor
        | t :           time value [s].

    Outputs: (4)
        | v_a :         3-phase voltage value, first phase, [V], input to the motor
        | v_b :         3-phase voltage value, second phase, [V], input to the motor
        | v_c :         3-phase voltage value, third phase [V], input to the motor
        | t :           time value [s].

    Model Configuration Parameters:
        | sawtooth_freq :       Frequency of PWM signal [Hz], default value in default_parameters.
        | x0 :                  Initial state-vector containing v_a, v_b, v_c and t.
    
    """
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
