# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from prog_models.models import Powertrain, ESC, DCMotor

esc = ESC()
motor = DCMotor()
powertrain = Powertrain(esc, motor)

def future_loading(t, x=None):
    return powertrain.InputContainer({
        'duty': 1,
        'v': 23
    })

(times, inputs, states, outputs, event_states) = powertrain.simulate_to(2, future_loading, dt=2e-5, save_freq=0.1, print=True)
