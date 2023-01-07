from prog_models.models import DCMotor, ESC
from prog_models import PrognosticsModel
from prog_models.composite_model import CompositeModel
m_motor = DCMotor()
m_esc = ESC()

def update_Cq(params):
    return {
        'C_q': params['c_q'] * params['rho'] * pow(params['D'], 5)
    }


class Load(PrognosticsModel):
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

m_load = Load()

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
print('\ninputs: ', m_composite.inputs)
print('states: ', m_composite.states)
print('outputs: ', m_composite.outputs)
x0 = m_composite.initialize()
print(x0)
print(m_composite.next_state(x0, {'ESC.v': 10, 'ESC.duty': 0.5}, 0.1))
print(m_composite.output(x0))
print(m_composite.event_state(x0))
print(m_composite.threshold_met(x0))
