# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
"""
Example of generating a models from constituant parts. Can be run using the following command `python -m examples.model_gen_example`

Model for this example is that of an object thrown into the air, predicting impact event
"""

# Deriv prog model was selected because the model can be described as x' = x + dx*dt
from prog_models.prognostics_model import PrognosticsModel

def run_example():
    # Step 1: Define keys
    keys = {
        'inputs': [], # no inputs, no way to control
        'states': [
            'x', # Position (m) 
            'v'  # Velocity (m/s)
            ],
        'outputs': [ # Anything we can measure
            'x' # Position (m)
        ],
        'events': [
            'falling', # Event- object is falling
            'impact' # Event- object has impacted ground
        ]
    }

    thrower_height = 1.83 # m
    throwing_speed = 40 # m/s
    # Step 2: Define initial state
    def initialize(u, z):
        return {
            'x': thrower_height, # Thrown, so initial altitude is height of thrower
            'v': throwing_speed # Velocity at which the ball is thrown - this guy is an professional baseball pitcher
            }
    
    # Step 3: Define dx equation
    def dx(t, x, u):
        return {
            'x': x['v'],
            'v': -9.81 # Acceleration of gravity
        }

    # Step 3: Define equation for calculating output/measuremetn
    def output(t, x):
        return {
            'x': x['x']
        }

    # Step 4: Define threshold equation
    def threshold_met(t, x):
        return {
            'falling': x['v'] < 0,
            'impact': x['x'] <= 0
        }

    # Step 5 (optional): Define event state equation- measurement of how close you are to threshold (0-1)
    def event_state(t, x): 
        event_state.max_x = max(event_state.max_x, x['x']) # Maximum altitude
        return {
            'falling': max(x['v']/throwing_speed,0), # Throwing speed is max speed
            'impact': max(x['x']/event_state.max_x,0) # 1 until falling begins, then it's fraction of height
        }
    event_state.max_x = 0
    
    # Step 6: Generate model
    m = PrognosticsModel.generate_model(keys, initialize, output, event_state_eqn = event_state, threshold_eqn=threshold_met, dx_eqn=dx)

    # Step 7: Setup for simulation 
    def future_load(t, x=None):
        return {}

    # Step 8: Simulate to impact
    event = 'impact'
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':thrower_height}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    
    # Print results
    for i in range(len(times)):
        print("Time: {}\n\tInput: {}\n\tState: {}\n\tOutput: {}\n\tEvent State: {}\n".format(round(times[i],2), inputs[i], states[i], outputs[i], event_states[i]))
    print('The object hit the ground in {} seconds'.format(round(times[-1],2)))

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()