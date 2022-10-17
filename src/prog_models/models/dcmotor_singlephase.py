# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from prog_models import PrognosticsModel
import math
import numpy as np


# Derived Paramaters
def update_L1(params):
    return {
        'L1': params['L'] - params['M']
    }


def update_Kv(params):
    return {
        'Kv':    1.0 / params['Kt']
    }


def update_Km(params):
    return {
        'Km': params['Kt'] / math.sqrt(params['R'])
    }


def update_B(params):
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
            Resistance (Ohm)
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

    default_parameters = dict(L=83.0e-6,                      # H, self-inductance
                              M=0.0,                          # H, mutual inductance
                              R=0.081,                        # Ohm, armature resistance
                              Kt=0.0265258,                   # V/(rad/s), back emf constant // Nm/A, torque constant
                              Kv=400.0,                       # (rad/s)/V, inverse of Kt
                              Km=0.0265258/math.sqrt(0.081),  # Nm/sqrt(W), motor constant, will be calculated on next line
                              B=0.0051 * (0.0265258/math.sqrt(0.081))**1.9485,  # Nm/(rad/s), friction coefficient. Estimate B according to empirical model 0.0051 * Km^1.9485 from:
                                                                                # Jeong et al., Improvement of Electric Propulsion System Model for Performance Analysis of Large-Size Multicopter UAVs. MDPI Applied Sciences.
                                                                                # This paper was for "large multicopter UAVs." 
                              J=2.69e-5,                      # kg*m^2, rotor inertia
                              Jp=1e-4,
                              motor_weight=0.230,             # kg, total with wires and bullets
                              x0={'i': 0.0, 'v_rot': 0.0}     # [A, rad/s], initial state values
                              )

    state_limits = {
        'i': (-math.inf, math.inf),           # undefined -- current can be very high if for a short perior of time. Max current of 38A is for 180s
        'v_rot': (-math.inf, math.inf),       # upper limit to be defined based on parameters (equivalent to rotor speed at 0 load)
    }

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
