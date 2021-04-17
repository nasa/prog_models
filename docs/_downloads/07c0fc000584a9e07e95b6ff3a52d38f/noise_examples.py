# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example defining and testing a new model. Can be run using the following command `python -m examples.new_model_example`
"""

# Deriv prog model was selected because the model can be described as x' = x + dx*dt
from prog_models.prognostics_model import PrognosticsModel

# Model used in example
class ThrownObject(PrognosticsModel):
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

    # The Default parameters. Overwritten by passing parameters dictionary into constructor
    default_parameters = {
        'thrower_height': 1.83, # m
        'throwing_speed': 40, # m/s
        'g': -9.81, # Acceleration due to gravity in m/s^2
        'process_noise': 0.0 # amount of noise in each step
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
        return self.apply_measurement_noise({
            'x': x['x']
        })

    # This is actually optional. Leaving thresholds_met empty will use the event state to define thresholds.
    #  Threshold = Event State == 0. However, this implementation is more efficient, so we included it
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
    def future_load(t, x=None):
        return {}
    event = 'impact'

    # Ex1: No noise
    process_noise = 0
    model_config = {'process_noise': process_noise}
    m = ThrownObject(model_config)
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    print('Example without noise')
    print('\t- states: {}'.format(['{}s: {}'.format(round(t,2), x) for (t,x) in zip(times, states)])) 
    print('\t- impact time: {}s'.format(times[-1]))

    # Ex2: with noise - same noise applied to every state
    process_noise = 0.5
    model_config = {'process_noise': process_noise}
    m = ThrownObject(model_config)  # Noise with a std of 0.5 to every state
    print('\nExample without same noise for every state')
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    print('\t- states: {}'.format(['{}s: {}'.format(round(t,2), x) for (t,x) in zip(times, states)])) 
    print('\t- impact time: {}s'.format(times[-1]))

    # Ex3: noise- more noise on position than velocity
    process_noise = {'x': 0.25, 'v': 0.75}
    model_config = {'process_noise': process_noise}
    m = ThrownObject(model_config) 
    print('\nExample with more noise on position than velocity')
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    print('\t- states: {}'.format(['{}s: {}'.format(round(t,2), x) for (t,x) in zip(times, states)])) 
    print('\t- impact time: {}s'.format(times[-1]))

    # Ex4: noise- Ex3 but uniform
    process_noise = {'x': 0.25, 'v': 0.75}
    process_noise_dist = 'uniform'
    model_config = {'process_noise_dist': process_noise_dist, 'process_noise': process_noise}
    m = ThrownObject(model_config) 
    print('\nExample with more uniform noise')
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    print('\t- states: {}'.format(['{}s: {}'.format(round(t,2), x) for (t,x) in zip(times, states)])) 
    print('\t- impact time: {}s'.format(times[-1]))

    # Ex5: noise- Ex3 but triangle
    process_noise = {'x': 0.25, 'v': 0.75}
    process_noise_dist = 'triangular'
    model_config = {'process_noise_dist': process_noise_dist, 'process_noise': process_noise}
    m = ThrownObject(model_config) 
    print('\nExample with triangular process noise')
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    print('\t- states: {}'.format(['{}s: {}'.format(round(t,2), x) for (t,x) in zip(times, states)])) 
    print('\t- impact time: {}s'.format(times[-1]))

    # Ex6: Measurement noise
    # Everything we've done with process noise, we can also do with measurement noise.
    # Just use 'measurement_noise' and 'measurement_noise_dist' 
    measurement_noise = {'x': 0.25} # For each output
    measurement_noise_dist = 'uniform'
    model_config = {'measurement_noise_dist': measurement_noise_dist, 'measurement_noise': measurement_noise}
    m = ThrownObject(model_config) 
    print('\nExample with measurement noise')
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    print('\t- states: {}'.format(['{}s: {}'.format(round(t,2), x) for (t,x) in zip(times, states)])) 
    print('\t- outputs: {}'.format(['{}s: {}'.format(round(t,2), x) for (t,x) in zip(times, outputs)])) 
    print('\t- impact time: {}s'.format(times[-1]))
    print(' note the output is sometimes not the same as state- that is the measurement noise')

    # Ex7: OK, now for something a little more complicated. Let's try proportional noise on v only (more variation when it's going faster)
    # This can be used to do custom or more complex noise distributions
    def apply_proportional_process_noise(self, x, dt = 1):
        return {
            'x': x['x'], # No noise on state
            'v': x['v'] + dt*0.5*x['v']
        }
    process_noise = apply_proportional_process_noise
    model_config = {'process_noise': apply_proportional_process_noise}
    m = ThrownObject(model_config)
    print('\nExample with proportional noise on velocity')
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    print('\t- states: {}'.format(['{}s: {}'.format(round(t,2), x) for (t,x) in zip(times, states)])) 
    print('\t- impact time: {}s'.format(times[-1]))


# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
