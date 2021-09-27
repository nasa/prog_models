# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from .. import PrognosticsModel


class ThrownObject(PrognosticsModel):
    """
    Model that similates an object thrown into the air without air resistance
    """

    inputs = []  # no inputs, no way to control
    states = [
        'x',  # Position (m) 
        'v'  # Velocity (m/s)
        ]
    outputs = [
        'x'  # Position (m)
    ]
    events = [
        'falling',  # Event- object is falling
        'impact'  # Event- object has impacted ground
    ]

    # The Default parameters. Overwritten by passing parameters dictionary into constructor
    default_parameters = {
        'thrower_height': 1.83,  # m
        'throwing_speed': 40,  # m/s
        'g': -9.81,  # Acceleration due to gravity in m/s^2
        'process_noise': 0.0  # amount of noise in each step
    }

    def initialize(self, u, z):
        self.max_x = 0.0
        return {
            'x': self.parameters['thrower_height'],  # Thrown, so initial altitude is height of thrower
            'v': self.parameters['throwing_speed']  # Velocity at which the ball is thrown - this guy is a professional baseball pitcher
            }
    
    def dx(self, x, u):
        return {'x': x['v'],
                'v': self.parameters['g']}  # Acceleration of gravity

    def output(self, x):
        return {'x': x['x']}

    # This is actually optional. Leaving thresholds_met empty will use the event state to define thresholds.
    #  Threshold = Event State == 0. However, this implementation is more efficient, so we included it
    def threshold_met(self, x):
        return {
            'falling': x['v'] < 0,
            'impact': x['x'] <= 0
        }

    def event_state(self, x): 
        self.max_x = max(self.max_x, x['x'])  # Maximum altitude
        return {
            'falling': max(x['v']/self.parameters['throwing_speed'],0),  # Throwing speed is max speed
            'impact': max(x['x']/self.max_x,0)  # 1 until falling begins, then it's fraction of height
        }
