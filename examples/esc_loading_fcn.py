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

# Consts
DC_volt= 10  # in Volts
sawtooth_freq= 600  # Hz, sawtooth frequency
duty = 1
        
def esc_loading(t, x = {'theta': 0}):
    pw = max(signal.square(2 * pi * sawtooth_freq * t, duty=1), 0)
    V = pw*DC_volt
    SP = commutation(x['theta'])
    VP = [V * SP[i] for i in range(3)]

    return {
        'v_a': VP[0],
        'v_b': VP[1],
        'v_c': VP[2],
        't_l': 0
    }
