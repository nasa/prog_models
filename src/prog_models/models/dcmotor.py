# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from prog_models import PrognosticsModel
from math import pi
import numpy as np
from copy import deepcopy

# Support Functions
def backemf(theta):
    """Backemf for the current opsition of rotor (theta)"""
    theta = theta * (180/pi) # convert rad to deg
    if 0. <= theta <= 60:
        f_a = 1
        f_b = -1
        f_c = -(1/30)*theta+1
    elif 60 < theta <= 120:
        f_a = 1
        f_b = (1/30)*(theta-60)-1
        f_c = -1
    elif 120 < theta <= 180:
        f_a = -(1/30)*(theta-120)+1
        f_b = 1
        f_c = -1
    elif 180 < theta <= 240:
        f_a = -1
        f_b = 1
        f_c = (1/30)*(theta-180)-1
    elif 240 < theta <= 300:
        f_a = -1
        f_b = -(1/30)*(theta-240)+1
        f_c = 1
    else:
        f_a = (1/30)*(theta-300)-1
        f_b = -1
        f_c = 1

    return f_a,f_b,f_c

# Derived Paramaters
def update_L1(params):
    return {
        'L1': params['L'] - params['M']
    }

def update_R_L1(params):
    return {
        'negR_L1': -params['R']/params['L1']
    }

def update_BC(params):
    L1 = params['L1']
    J = params['J']
    return {
        'Bc': np.array([[(1/L1), 0, 0, 0],
                [0, (1/L1), 0, 0],
                [0, 0, (1/L1), 0],
                [0, 0, 0, -(1/J)],
                [0, 0, 0, 0]
                ])
    }

def update_AC(params):
    L1 = params['L1']
    Flx = params['K']
    J = params['J']
    B = params['B']
    Po = params['Po']
    negR_L1 = params['negR_L1']
    return {'Ac': np.array([[negR_L1, 0, 0, -(Flx/L1), 0],
            [0, negR_L1, 0, -(Flx/L1), 0],
            [0, 0, negR_L1, -(Flx/L1), 0],
            [((Flx)/J), ((Flx)/J), ((Flx)/J), -(B/J), 0],
            [0, 0, 0, (Po/2), 0]
            ])
    }


class DCMotor(PrognosticsModel):
    """
    Model of DC Motor

    Inputs/Loading: (4)
        | v_a, v_b, v_c: Voltages at a, b, c
        | t_l: Torque from load

    States: (5)
        | i_a, i_b, i_c: Currents provided to motor 
        | v_rot: Rotational velocity (rad/sec)
        | theta: Angle of motor (rad)

    Outputs: (2)
        | v_rot: Rotational velocity (rad/sec)
        | theta: Angle of motor (rad)

    Model Configuration Parameters:
        | process_noise : Process noise (applied at dx/next_state). 
                    Can be number (e.g., .2) applied to every state, a dictionary of values for each 
                    state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        | process_noise_dist : Optional, distribution for process noise (e.g., normal, uniform, triangular)
        | measurement_noise : Measurement noise (applied in output eqn)
                    Can be number (e.g., .2) applied to every output, a dictionary of values for each 
                    output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        | measurement_noise_dist : Optional, distribution for measurement noise (e.g., normal, uniform, triangular)
        | x0: Initial State
        | L: Inductance (H)
        | M: Mutual inductance (H)
        | R: Resistance (Ohm)
        | K: back emf constant / Torque constant (V/rad/sec)  
        | B: Friction in motor / Damping (Not a function of thrust) (Nm/(rad/s))
        | Po: no of poles in rotor 
        | J: Load moment of inertia (neglecting motor shaft inertia) (Kg*m^2)
    """
    states = ['i_a', 'i_b', 'i_c', 'v_rot', 'theta']
    inputs = ['v_a', 'v_b', 'v_c', 't_l']
    outputs = ['v_rot', 'theta']
    param_callbacks = {
        'L': [update_L1],
        'L1': [update_R_L1, update_BC, update_AC],
        'M': [update_L1],
        'R': [update_R_L1, update_AC],
        'J': [update_BC, update_AC],
        'K': [update_AC],
        'B': [update_AC],
        'Po': [update_AC],
    }

    default_parameters = {
        # motor parameters
        'L': 83e-6,  # (H) inductance
        'M': 0, # (H) Mutual inductance
        'R': 0.081, # (Ohm) Resistance
        'K': 0.0265, # (V/rad/sec)  back emf constant / Torque constant (Nm/A) 
        'B': 0, # Nm/(rad/s) Friction in motor / Damping (Not a function of thrust)
        'Po': 28, # no of poles in rotor

        # Load parameters 
        'J': 26.967e-6, # (Kg*m^2) Load moment of inertia (neglecting motor shaft inertia)

        # Matricies
        'Cc': np.array([[1, 1, 1, 1, 1]]),
        'Dc': np.array([0,0,0,0]),

        # Initial State
        'x0': {
            'i_a': 0,
            'i_b': 0,
            'i_c': 0,
            'v_rot': 0,
            'theta': 0
        }
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer(self.parameters['x0'])

    def next_state(self, x, u, dt):
        params = self.parameters

        U = np.array([[u[key]] for key in self.inputs])  

        # Convert to new context
        (F_a, F_b, F_c) = backemf(x['theta'])

        Ac = deepcopy(params['Ac'])
        Ac[0][3] *= F_a
        Ac[1][3] *= F_b
        Ac[2][3] *= F_c
        Ac[3][0] *= F_a
        Ac[3][1] *= F_b
        Ac[3][2] *= F_c
        # TODO(CT): Move F_* to U_vector

        dxdt = np.dot(Ac, x.matrix) + np.dot(params['Bc'], U) 
        x_new = x.matrix +  dxdt * dt
        x_new[4] = x_new[4] % (2 * pi)

        # Convert back
        return {
            key: value[0] for (key, value) in zip(self.states, x_new)
        }

    def output(self, x):
        return {
            'v_rot': x['v_rot'],
            'theta': x['theta']
        }    
        