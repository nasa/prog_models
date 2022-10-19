# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import math
import numpy as np

from prog_models import PrognosticsModel

# Derived Paramaters
def update_L1(params):
    return {
        'L1': params['L'] - params['M']
    }

def update_Kv(params):
    # (rad/s)/V, inverse of Kt
    return {
        'Kv':    1.0 / params['Kt']
    }

def update_Km(params):
    # Nm/sqrt(W), motor constant, will be calculated on next line
    return {
        'Km': params['Kt'] / math.sqrt(params['R'])
    }

def update_B(params):
    # Estimate B according to empirical model 0.0051 * Km^1.9485 from:
    # Jeong et al., Improvement of Electric Propulsion System Model for Performance Analysis of Large-Size Multicopter UAVs. MDPI Applied Sciences.
    # This paper was for "large multicopter UAVs." 
    return {
        'B': 0.0051 * params['Km']**1.9485
    }


class DCMotorSP(PrognosticsModel):
    """
    .. versionadded:: 1.4.0

    :term:`Model<model>` of single-phase DC Motor, as defined by the following equations:

    d i     1
    ---  = --- * ( V - E - R * i )
    dt      L1

    d omega    1
    ------- = --- * (T_e - T_l - B * omega)
      dt       Jt

    where:
        i       current, A
        omega   rotor speed, rad/s
        V       input voltage, V
        E       back-emf voltage, V
        R       armature resistance, Ohm
        Jt      total inertia (rotor + propeller or load), kg*m^2
        T_e     driving torque (electrical), Nm
        T_l     load torque (mechanical), Nm
        B       friction coefficient, Nm/(rad/s)
        t       time, s
    
    :term:`Events<event>`: (0)
        | None

    :term:`Inputs/Loading<input>`: (2)
        | v: Voltage
        | t_l: Torque from load

    :term:`States<state>`: (2)
        | i: current (A)
        | v_rot: Rotational velocity (rad/sec)

    :term:`Outputs<output>`: (1)
        | v_rot: Rotational velocity (rad/sec)

    Keyword Args
    ------------
        process_noise : Optional, float or Dict[str, float]
          :term:`Process noise<process noise>` (applied at dx/next_state). 
          Can be number (e.g., .2) applied to every state, a dictionary of values for each 
          state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        process_noise_dist : Optional, String
          distribution for :term:`process noise` (e.g., normal, uniform, triangular)
        measurement_noise : Optional, float or Dict[str, float]
          :term:`Measurement noise<measurement noise>` (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of values for each
          output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        measurement_noise_dist : Optional, String
          distribution for :term:`measurement noise` (e.g., normal, uniform, triangular)
        L : float
            Self-inductance (H)
        M: float
            Mutual inductance (H)
        R: float
            Armature Resistance (Ohm)
        Kt: float
            back emf constant / Torque constant (V/rad/sec)  
        B: float
            Friction in motor / Damping (Not a function of thrust) (Nm/(rad/s))
        J: float
            Load moment of inertia (neglecting motor shaft inertia) (Kg*m^2)
        x0 : dict[str, float]
            Initial :term:`state`

    """
    states  = ['i', 'v_rot']
    inputs  = ['v', 't_l']
    outputs = ['v_rot']

    param_callbacks = {
        'L': [update_L1],
        'M': [update_L1],
        'R': [update_Km, update_B],
        'Kt': [update_Kv, update_Km, update_B],
        'Km': [update_B]
        }

    default_parameters = dict(L=83.0e-6,
                              M=0.0,
                              R=0.081,
                              Kt=0.0265258,
                              J=2.69e-5,
                              Jp=1e-4,
                              x0={'i': 0.0, 'v_rot': 0.0}
                              )

    def dx(self, x: dict, u: dict):
        
        # Get parameters
        parameters     = self.parameters
        friction_coeff = parameters['B']
        inertia        = parameters['J'] + parameters['Jp']
        inductance     = parameters['L']

        # Get input
        load_torque    = u['t_l']
        voltage        = u['v']

        # Forcing on electrical and mechanical equations
        back_emf_v     = parameters['Kt'] * x['v_rot']
        el_torque      = parameters['Kt'] * x['i']

        # Rate of change (state-space) form
        didt    = 1.0 / inductance * (voltage - back_emf_v - parameters['R'] * x['i'])
        dvrotdt = 1.0 / inertia * (el_torque - load_torque - friction_coeff * x['v_rot'])

        return self.StateContainer(np.array([
            np.atleast_1d(didt),            # current
            np.atleast_1d(dvrotdt)          # rotor speed
        ]))

    def output(self, x : dict):
        rotor_speed = x['v_rot']
        return self.OutputContainer(np.array([
            np.atleast_1d(rotor_speed),
            ]))
