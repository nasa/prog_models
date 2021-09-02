from prog_models import PrognosticsModel
from math import pi
import numpy as np
import scipy.signal as signal

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

def update_Cq(params):
    return {
        'C_q': params['c_q'] * params['rho'] * pow(params['D'], 5)
    }

def update_R_L1(params):
    return {
        'negR_L1': -params['R']/params['L1']
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
        | c_q coefficient of torque (APC data, derived) [dimensionless]
        | rho: (Kg/m^3)
        | D: Propeller diameter (m)
    """
    states = ['i_a', 'i_b', 'i_c', 'v_rot', 'theta']
    inputs = ['v_a', 'v_b', 'v_c', 't_l']
    outputs = ['v_rot', 'theta']
    param_callbacks = {
        'L': [update_L1],
        'L1': [update_R_L1],
        'M': [update_L1],
        'c_q': [update_Cq],
        'rho': [update_Cq],
        'D': [update_Cq],
        'R': [update_R_L1]
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

        # Motor Load parameters
        'c_q': 0.00542, # coefficient of torque (APC data, derived) [dimensionless]
        'rho': 1.225, # (Kg/m^3)
        'D': 0.381, # (m)

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
        return self.parameters['x0']

    def next_state(self, x, u, dt):
        params = self.parameters

        # Convert to new context
        (F_a, F_b, F_c) = backemf(x['theta'])
        R = params['R']
        L1 = params['L1']
        Flx = params['K']
        J = params['J']
        B = params['B']
        Po = params['Po']
        C_q = params['C_q']
        negR_L1 = params['negR_L1']
        X = np.array([x[key] for key in self.states])
        U = np.array([u[key] for key in self.inputs])

        # Raj's code
        Ac = np.array([[negR_L1, 0, 0, -((Flx*F_a)/L1), 0],
            [0, negR_L1, 0, -((Flx*F_b)/L1), 0],
            [0, 0, negR_L1, -((Flx*F_c)/L1), 0],
            [((Flx*F_a)/J), ((Flx*F_b)/J), ((Flx*F_c)/J), -((B/J)+(C_q*X[3]/J)), 0],
            [0, 0, 0, (Po/2), 0]
            ])
        Bc = np.array([[(1/L1), 0, 0, 0],
                    [0, (1/L1), 0, 0],
                    [0, 0, (1/L1), 0],
                    [0, 0, 0, -(1/J)],
                    [0, 0, 0, 0]
                    ])
        Cc = np.array([[1, 1, 1, 1, 1]])
        Dc = np.array([0,0,0,0])
        
        # contineous to discrete
        dsys = signal.cont2discrete([Ac,Bc,Cc,Dc],dt)
        Ad = dsys[0]
        Bd = dsys[1]
        Cd = dsys[2]
        Dd = dsys[3]

        Xp = np.dot(Ad, X) + np.dot(Bd, U)
        Xp[4] = Xp[4] % ( 2.0 * pi )

        # Convert back
        return {
            key: value for (key, value) in zip(self.states, Xp)
        }

    def output(self, x):
        return {
            'v_rot': x['v_rot'],
            'theta': x['theta']
        }    
        