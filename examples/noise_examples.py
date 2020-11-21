# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example defining and testing a new model. Can be run using the following command `python -m examples.new_model_example`
"""

# Deriv prog model was selected because the model can be described as x' = x + dx*dt
from prog_models.deriv_prog_model import DerivProgModel

# Model used in example
class ThrownObject(DerivProgModel):
    """
    Model that similates an object thrown into the air without air resistance
    """

    inputs = [] # no inputs, no way to control
    states = [
        'x', # Position (m) 
        'v'  # Velocity (m/s)
        ]
    outputs = [ # Anything we can measure
        'x' # Position (m)
    ]
    events = [
        'falling', # Event- object is falling
        'impact' # Event- object has impacted ground
    ]

    default_parameters = {
        'thrower_height': 1.83, # m
        'throwing_speed': 40, # m/s
        'g': -9.81, # Acceleration due to gravity in m/s^2
        'process_noise': 0.0 # Required by all models, amount of noise in each step
    }

    def initialize(self, u, z):
        self.max_x = 0.0
        return {
            'x': self.parameters['thrower_height'], # Thrown, so initial altitude is height of thrower
            'v': self.parameters['throwing_speed'] # Velocity at which the ball is thrown - this guy is an professional baseball pitcher
            }
    
    def dx(self, t, x, u):
        return self.apply_process_noise({
            'x': x['v'],
            'v': self.parameters['g'] # Acceleration of gravity
        })

    def output(self, t, x):
        return {
            'x': x['x']
        }

    def threshold_met(self, t, x):
        return {
            'falling': x['v'] < 0,
            'impact': x['x'] <= 0
        }

    def event_state(self, t, x): 
        self.max_x = max(self.max_x, x['x']) # Maximum altitude
        return {
            'falling': max(x['v']/self.parameters['throwing_speed'],0), # Throwing speed is max speed
            'impact': max(x['x']/self.max_x,0) # 1 until falling begins, then it's fraction of height
        }

def run_example():
    def future_load(t):
        return {}
    event = 'impact'

    # Ex1: No noise
    m = ThrownObject({'process_noise': 0})
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    print('Example without noise')
    print('\t- states: {}'.format(['{}s: {}'.format(round(t,2), x) for (t,x) in zip(times, states)])) 
    print('\t- impact time: {}s'.format(times[-1]))

    # Ex2: with noise - same noise applied to every state
    m = ThrownObject({'process_noise': 0.5})  # Noise with a std of 0.5 to every state
    print('\nExample without same noise for every state')
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    print('\t- states: {}'.format(['{}s: {}'.format(round(t,2), x) for (t,x) in zip(times, states)])) 
    print('\t- impact time: {}s'.format(times[-1]))

    # Ex3: noise- more noise on position than velocity
    m = ThrownObject({'process_noise': {'x': 0.25, 'v': 0.75}})  # Noise with a std of 0.2 to every state
    print('\nExample with more noise on position than velocity')
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    print('\t- states: {}'.format(['{}s: {}'.format(round(t,2), x) for (t,x) in zip(times, states)])) 
    print('\t- impact time: {}s'.format(times[-1]))

    # Ex4: noise- Ex3 but uniform
    m = ThrownObject({'process_noise_dist': 'uniform', 'process_noise': {'x': 0.25, 'v': 0.75}})  # Noise with a std of 0.2 to every state
    print('\nExample with more noise on position than velocity')
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    print('\t- states: {}'.format(['{}s: {}'.format(round(t,2), x) for (t,x) in zip(times, states)])) 
    print('\t- impact time: {}s'.format(times[-1]))

    # Ex5: OK, now for something a little more complicated. Let's try proportional noise on v only (more variation when it's going faster)
    def apply_proportional_process_noise(self, x, dt = 1):
        return {
            'x': x['x'], # No noise on state
            'v': x['v'] + dt*0.5*x['v']
        }
    m = ThrownObject({'process_noise': apply_proportional_process_noise})
    print('\nExample with proportional noise on velocity')
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    print('\t- states: {}'.format(['{}s: {}'.format(round(t,2), x) for (t,x) in zip(times, states)])) 
    print('\t- impact time: {}s'.format(times[-1]))


# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
