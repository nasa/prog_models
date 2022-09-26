"""
Vehicle Models
SWS Project

Matteo Corbetta
matteo.corbetta@nasa.gov
"""

# Import functions
# =================
# from imports_ import np
import numpy as np


# Vehicle models
# ==============
def DJIS1000(payload=0.0, gravity=9.81):
    """
    DJI S 1000 / Octocopter
    :param payload:         float, payload for the flight.
    :param gravity:         float, gravity value according to location
    :return:                dictionaries containing mass, geometry and dynamic properties of the vehicle
    """

    mass = dict(body_empty=1.3,         # kg, total empty mass
                max_payload=None,        # kg, admissible payload    
                arm=0.325,              # kg, arm mass (comprehensive of motors)
                total=None,             # kg, total mass (to be calculated using payload, num rotors and weight of all arms)
                body=None,              # kg, mass of central body
                payload=payload,        # kg, payload mass
                Ixx=None,               # kg*m^2, interia moments along local axis
                Iyy=None,               # kg*m^2, interia moments along local axis
                Izz=None)               # kg*m^2, interia moments along local axis

    geom = dict(num_rotors=8,           # #, number of rotors
                body_type='sphere',     # -, should be disk. For now, only sphere.
                body_radius=0.3775/2.0, # m, body radius
                body_height=0.1,        # m, height of the body
                arm_length=0.386)       # m, arm length
    
    dynamics = dict(num_states=12,      # -, number of states in state vector: [x, y, z, u, v, w, phi, theta, psi, p, q, r]
                    num_inputs=4,       # -, number of inputs: thrust, moments along three axes
                    num_outputs=3,      # -, number of output measures (position coordinates)
                    C = None,           # observation matrix (constant)
                    thrust2weight=5.0,  # -, thrust over weight ratio
                    max_speed=15.0,     # m/s, max speed
                    max_wind_speed=8.0, # m/s, max wind speed for safe flight
                    max_acceleration=None,  # m/s^2, max acceleration to be calculated using thrust2weight ratio
                    max_thrust=None,    # N, max thrust the power system an deliver
                    state_vars=['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r'],
                    input_vars=['thrust', 'torque-p', 'torque-q', 'torque-r'],
                    output_vars=['x', 'y', 'z'],
                    aero = dict(cd=0.4,   # drag coefficient of airframe (including rotors), found on an article (but not reliable)
                                ad=0.5 * (np.pi * geom['body_radius']**2.0),   # apparent face of the rotorcraft facing air for drag force (guess using the "rule of thumb" found on another paper)
                                ),
                    # kt=1.09e-2, # Rotor thrust constant
                    # kt = 1.e-4,
                    kt = 5.4e-5,
                    kq=5e-5,       # Rotor torque constant
                    Gamma=None,     # Thrust allocation matrix (constant, depending on kt, kq, and the UAV rotor configuration)
                    Gamma_inv=None,     # Inverse of thrust allocation matrix (constant, depending on kt, kq, and the UAV rotor configuration)
                    )
    
    mass['body']        = mass['body_empty'] + geom['num_rotors'] * mass['arm']
    mass['max_payload'] = 11.0 - mass['body']
    mass['total']       = mass['body'] + mass['payload'] 
    if payload > mass['max_payload']:   raise Warning("Payload for DJIS1000 exceeds its maximum recommended payload.")

    dynamics['max_thrust']       = dynamics['thrust2weight'] * (mass['body'] * gravity)
    dynamics['max_acceleration'] = dynamics['max_thrust'] / mass['total']

    # Generate observation matrix
    dynamics['C'] = np.zeros((dynamics['num_outputs'], dynamics['num_states']))
    for ii in range(dynamics['num_outputs']):   dynamics['C'][ii, ii] = 1.0

    return mass, geom, dynamics







    
if __name__ == '__main__':

    print('Vehicle models')

    