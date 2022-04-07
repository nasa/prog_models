# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from prog_models import PrognosticsModel

def update_Cq(params):
    return {
        'C_q': params['c_q'] * params['rho'] * pow(params['D'], 5)
    }


class Powertrain(PrognosticsModel):
    """
    Model Configuration Parameters:
        | process_noise : Process noise (applied at dx/next_state). 
                    Can be number (e.g., .2) applied to every state, a dictionary of values for each 
                    state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        | process_noise_dist : Optional, distribution for process noise (e.g., normal, uniform, triangular)
        | measurement_noise : Measurement noise (applied in output eqn)
                    Can be number (e.g., .2) applied to every output, a dictionary of values for each 
                    output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        | measurement_noise_dist : Optional, distribution for measurement noise (e.g., normal, uniform, triangular)
        | c_q coefficient of torque (APC data, derived) [dimensionless]
        | rho: (Kg/m^3)
        | D: Propeller diameter (m)
    """
    inputs = ['duty', 'v']
    states = ['v_a', 'v_b', 'v_c', 't', 'i_a', 'i_b', 'i_c', 'v_rot', 'theta']
    outputs = ['v_rot', 'theta']
    param_callbacks = {
        'c_q': [update_Cq],
        'rho': [update_Cq],
        'D': [update_Cq],
    }

    default_parameters = {
        # Load parameters 
        'c_q': 0.00542, # coefficient of torque (APC data, derived) [dimensionless]
        'rho': 1.225, # (Kg/m^3)
        'D': 0.381, # (m)
    }

    def __init__(self, esc, motor, **kwargs):
        super().__init__(**kwargs)
        self.esc = esc
        self.motor = motor

    def initialize(self, u=None, z=None):
        x0 = self.esc.initialize(u=u, z=z)
        x0.update(self.motor.initialize(u=u, z=z))

        return self.StateContainer(x0)

    def next_state(self, x, u, dt):
        x_esc = self.esc.StateContainer(x)
        u_esc = {
            'duty': u['duty'], 
            'theta': x['theta'],
            'v': u['v']}
        u_esc = self.esc.InputContainer(u_esc)
        x_esc = self.esc.next_state(x_esc, u_esc, dt)

        x_motor = self.motor.StateContainer(x)
        u_motor = {
            'v_a': x_esc['v_a'],
            'v_b': x_esc['v_b'],
            'v_c': x_esc['v_c'],
            't_l': self.parameters['C_q']*x['v_rot']**2
        }
        x_motor = self.motor.next_state(x_motor, u_motor, dt)

        x_esc.update(x_motor)

        return self.StateContainer(x_esc)

    def output(self, x):
        return self.OutputContainer(
            {
                'v_rot': x['v_rot'], 
                'theta': x['theta']})
