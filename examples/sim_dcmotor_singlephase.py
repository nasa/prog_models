# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of a DC motor being simulated for a set amount of time, using the single-phase dcmotor model. 
"""

from prog_models.models import dcmotor_singlephase

import math
import matplotlib.pyplot as plt
import numpy as np


def run_example():

    motor = dcmotor_singlephase.DCMotorSP()
    
    def future_loading(t, x=None):
        f = 0.5
        # load proportional to rotor speed
        if x is None:
            t_l = 0.0
        else:
            t_l = 1e-5 * x['v_rot']**2.0
        return motor.InputContainer({
            'v': 10.0 + 2.0 * math.sin(math.tau * f * t),
            't_l': t_l  # assuming constant load (simple)
            })

    simulated_results = motor.simulate_to(2.0, future_loading, dt=1e-3, save_freq=0.1, print=True)

    current_vals = np.array([simulated_results[2][i]['i'] for i in range(len(simulated_results[2]))])
    rotorspeed_vals = np.array([simulated_results[2][i]['v_rot'] for i in range(len(simulated_results[2]))])
    plt.figure()
    plt.subplot(211)
    plt.plot(current_vals, '-o')
    plt.subplot(212)
    plt.plot(rotorspeed_vals, '-o')
    plt.show()
    return


if __name__ == '__main__':
    print("Simulation of DC single-phase motor")

    run_example()