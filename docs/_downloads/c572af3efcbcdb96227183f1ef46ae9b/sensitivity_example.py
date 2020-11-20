# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
"""
Example performing a sensitivity analysis on a new model. Can be run using the following command `python -m examples.sensitivity_example`
"""

# Deriv prog model was selected because the model can be described as x' = x + dx*dt
from prog_models.deriv_prog_model import DerivProgModel
import numpy as np

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
    def future_load(t):
        return {}

    # Step 3: Setup range on parameters considered
    thrower_height_range = range(1, 2, 0.05)

    # Step 4: Sim for each 
    event = 'impact'
    eods = np.empty(len(thrower_height_range))
    for (i, thrower_height) in zip(range(len(thrower_height_range)), thrower_height_range):
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
        eods[i] = times[-1]

    # Step 5: Analysis
    print(zip(thrower_height_range, eods))

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()