# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of generating a trajectory for a rotorcraft through a set of coarse waypoints 
"""
import numpy as np
import matplotlib.pyplot as plt

from prog_models.aux_fcns.traj_gen import trajectory_gen_fcn as traj_gen
from prog_models.models.uav_model import UAVGen 
from prog_models.loading_fcns.controller_test import ExampleController

def run_example(): 

    # Define vehicle information:
    vehicle_params = {
        'dt': 0.2,
        'vehicle_model': 'tarot18', # Define vehicle
    }

    # Initialize vehicle 
    vehicle = UAVGen(**vehicle_params)

    # Define coarse waypoints: waypoints must be defined with a dictionary of numpy arrays or as columns in a text file 
    # See documentation for specific information on inputting waypoints 
    waypoints = {}
    waypoints['lat_deg']   = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719, 37.09719, 37.09719, 37.09748, 37.09798, 37.09776, 37.09776])
    waypoints['lon_deg']   = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628, -76.38628, -76.38628, -76.3848, -76.38589, -76.38629, -76.38629])
    waypoints['alt_ft']    = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0, 0.0, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
    waypoints['time_unix'] = np.array([1544188336, 1544188358, 1544188360, 1544188377, 1544188394, 1544188411, 1544188428, 1544188496, 1544188539, 1544188584, 1544188601, 1544188635, 1544188652, 1544188672, 1544188692])

    # Define reference trajectory parameters, if desired
    ref_params = {
        'nurbs_order': 4
    }

    # Calculate reference trajectory 
    ref_traj = traj_gen(waypoints, vehicle, **ref_params)

    # Define controller
    ctrl = ExampleController(ref_traj,vehicle)

    # Set simulation options 
    options = {
        'dt': 0.3, ### THIS IS CURRENTLY REQUIRED - this needs help, issue with consistency in dt 
        'save_freq': vehicle_params['dt']
    }

    # Generate trajectory
    traj_results_1 = vehicle.simulate_to(100, ctrl, **options)

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
    