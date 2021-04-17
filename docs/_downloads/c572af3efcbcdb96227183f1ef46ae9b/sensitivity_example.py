# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
"""
Example performing a sensitivity analysis on a new model. Can be run using the following command `python -m examples.sensitivity_example`
"""

# Deriv prog model was selected because the model can be described as x' = x + dx*dt
from prog_models.prognostics_model import PrognosticsModel
import numpy as np

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
        return {
            'x': x['v'],
            'v': self.parameters['g'] # Acceleration of gravity
        }

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
    # Demo model
    # Step 1: Create instance of model
    m = ThrownObject()

    # Step 2: Setup for simulation 
    def future_load(t, x=None):
        return {}

    # Step 3: Setup range on parameters considered
    thrower_height_range = np.arange(1.2, 2.1, 0.1)

    # Step 4: Sim for each 
    event = 'impact'
    eods = np.empty(len(thrower_height_range))
    for (i, thrower_height) in zip(range(len(thrower_height_range)), thrower_height_range):
        m.parameters['thrower_height'] = thrower_height
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':1e-3, 'save_freq':10})
        eods[i] = times[-1]

    # Step 5: Analysis
    print('For a reasonable range of heights, impact time is between {} and {}'.format(round(eods[0],3), round(eods[-1],3)))
    sensitivity = (eods[-1]-eods[0])/(thrower_height_range[-1] - thrower_height_range[0])
    print('  - Average sensitivity: {} s per cm height'.format(round(sensitivity/100, 6)))
    print("  - It seems impact time is not very sensitive to thrower's height")

    # Now lets repeat for throw speed
    throw_speed_range = np.arange(20, 40, 1)
    eods = np.empty(len(throw_speed_range))
    for (i, throw_speed) in zip(range(len(throw_speed_range)), throw_speed_range):
        m.parameters['throwing_speed'] = throw_speed
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':1e-3, 'save_freq':10})
        eods[i] = times[-1]

    print('\nFor a reasonable range of throwing speeds, impact time is between {} and {}'.format(round(eods[0],3), round(eods[-1],3)))
    sensitivity = (eods[-1]-eods[0])/(throw_speed_range[-1] - throw_speed_range[0])
    print('  - Average sensitivity: {} s per m/s speed'.format(round(sensitivity/100, 6)))
    print("  - It seems impact time is much more dependent on throwing speed")

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()