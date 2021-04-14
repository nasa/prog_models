# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example defining and testing a new model. Can be run using the following command `python -m examples.new_model_example`
"""

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
        # apply_process_noise is used to add process noise to each step
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
    # Demo model
    # Step 1: Create instance of model
    m = ThrownObject()

    # Step 2: Setup for simulation 
    def future_load(t, x=None):
        return {}

    # Step 3: Simulate to impact
    event = 'impact'
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    
    # Print results
    for i in range(len(times)):
        print("Time: {}\n\tInput: {}\n\tState: {}\n\tOutput: {}\n\tEvent State: {}\n".format(round(times[i],2), inputs[i], states[i], outputs[i], event_states[i]))
    print('The object hit the ground in {} seconds'.format(round(times[-1],2)))

    # OK, now lets compare performance on different heavenly bodies. 
    # This requires that we update the cofiguration
    grav_moon = -1.62
    opts = {
        'g': grav_moon
    }
    # The first way to change the configuration is to pass in your desired config into construction of the model
    m = ThrownObject(options=opts)
    (times_moon, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})

    grav_mars = -3.711
    # You can also update the parameters after it's constructed
    m.parameters['g'] = grav_mars
    (times_mars, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})

    grav_venus = -8.87
    m.parameters['g'] = grav_venus
    (times_venus, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})

    print('Time to hit the ground: ')
    print('\tvenus: {}s'.format(round(times_venus[-1],2)))
    print('\tearth: {}s'.format(round(times[-1],2)))
    print('\tmars: {}s'.format(round(times_mars[-1],2)))
    print('\tmoon: {}s'.format(round(times_moon[-1],2)))

    # We can also simulate until any event is met by neglecting the threshold_keys argument
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, options={'dt':0.005, 'save_freq':1})
    threshs_met = m.threshold_met(times[-1], states[-1])
    for (key, met) in threshs_met.items():
        if met:
            event_occured = key
    print('\nThis event that occured first: ', event_occured)
    # It falls before it hits the gorund, obviously

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
