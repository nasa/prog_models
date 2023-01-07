# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
TODO(CT): DESCIRPTION
"""

from prog_models.models import DCMotor, ESC
from prog_models import PrognosticsModel, CompositeModel

# Load Model- used below
def update_Cq(params):
    return {
        'C_q': params['c_q'] * params['rho'] * pow(params['D'], 5)
    }


class PropellerLoad(PrognosticsModel):
    inputs = ['v_rot']
    states = ['t_l']
    outputs = ['t_l']

    param_callbacks = {
        'c_q': [update_Cq],
        'rho': [update_Cq],
        'D': [update_Cq],
    }

    default_parameters = {
        # Load parameters 
        'c_q': 5.42e-7, # coefficient of torque (APC data, derived) [dimensionless]
        'rho': 1.225, # (Kg/m^3)
        'D': 0.381, # (m)

        'x0': {
            't_l': 0,
        }
    }

    def next_state(self, _, u, dt):
        return self.StateContainer({'t_l': self.parameters['C_q']*u['v_rot']**2})
    
    def output(self, x):
        return x

def run_example():
    # First, lets define the composite models
    m_motor = DCMotor()
    m_esc = ESC()
    m_load = PropellerLoad()

    # Now let's combine them into a single composite model
    # This model will then behave as a single model
    m_composite = CompositeModel(
        (m_esc, m_load, m_motor), 
        connections = [
            ('DCMotor.theta', 'ESC.theta'),
            ('ESC.v_a', 'DCMotor.v_a'),
            ('ESC.v_b', 'DCMotor.v_b'),
            ('ESC.v_c', 'DCMotor.v_c'),
            ('Load.t_l', 'DCMotor.t_l'),
            ('DCMotor.v_rot', 'Load.v_rot')],
        outputs = {'DCMotor.v_rot', 'DCMotor.theta'})
    
    print('inputs: ', m_composite.inputs)
    print('states: ', m_composite.states)
    print('outputs: ', m_composite.outputs)
    x0 = m_composite.initialize()
    print(x0)
    print(m_composite.next_state(x0, {'ESC.v': 10, 'ESC.duty': 0.5}, 0.1))
    print(m_composite.output(x0))
    print(m_composite.event_state(x0))
    print(m_composite.threshold_met(x0))

if __name__ == '__main__':
    run_example()
