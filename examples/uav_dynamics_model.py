# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of generating a trajectory for a rotorcraft through a set of coarse waypoints 
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from prog_models.aux_fcns.traj_gen import trajectory_gen_fcn as traj_gen
from prog_models.models.uav_model import SmallRotorcraft
from prog_models.loading_fcns.controllers import LQR_I, LQR

def run_example(): 

    # Define vehicle information:
    vehicle_params = {
        'dt': 0.1, 
        'vehicle_model': 'tarot18', 
        'process_noise': 0,
        'measurement_noise': 0
    }

    # Initialize vehicle 
    vehicle = SmallRotorcraft(**vehicle_params)

    # EXAMPLE 1: 
    # Define coarse waypoints: waypoints must be defined as a Pandas DataFrame
    # See documentation for specific information on inputting waypoints 
    # Latitude, longitude, and altitude values are required; ETAs are optional (see Example 3)

    # Here, we specify waypoints in a dictionary and then convert it to a Pandas DataFrame
    waypoints = {}
    waypoints['lat_deg']   = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719, 37.09719, 37.09719, 37.09748, 37.09798, 37.09776, 37.09776])
    waypoints['lon_deg']   = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628, -76.38628, -76.38628, -76.3848, -76.38589, -76.38629, -76.38629])
    waypoints['alt_ft']    = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0, 0.0, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
    waypoints['time_unix'] = np.array([1544188336, 1544188358, 1544188360, 1544188377, 1544188394, 1544188411, 1544188428, 1544188496, 1544188539, 1544188584, 1544188601, 1544188635, 1544188652, 1544188672, 1544188692])

    # Convert waypoints to pandas dataframe
    waypoints_pd = pd.DataFrame(waypoints)

    # Define reference trajectory parameters, if desired
    ref_params = {
        'nurbs_order': 4
    }

    # Calculate reference trajectory 
    ref_traj = traj_gen(waypoints=waypoints_pd, vehicle=vehicle, **ref_params)

    # Define controller and build scheduled control 
    ctrl = LQR(ref_traj, vehicle)
    ctrl.build_scheduled_control(vehicle.linear_model, input_vector=[vehicle.parameters['steadystate_input']])

    # Set simulation options 
    options = {
        'dt': 0.1, 
        'save_freq': vehicle_params['dt']
    }

    # Simulate vehicle to fly trajectory 
    traj_results = vehicle.simulate_to_threshold(ctrl, **options)

    # Visualize Results
    vehicle.visualize_traj(pred=traj_results, ref=ref_traj)

    # Note: waypoints can be specified in a variety of ways and then converted to a Pandas DataFrame for use 
    # Ex: generate a csv file with columns lat_deg, lon_deg, alt_ft, then use pd.read_csv(filename) to convert 
    # this to a dataframe for use in the trajectory generation 

    # EXAMPLE 2: 
    # In this example, we define another trajectory through the same waypoints but with speeds defined instead of ETAs
    del waypoints['time_unix'] # Delete ETAs for this example

    # Convert dictionary to Pandas DataFrame
    waypoints_pd_speeds = pd.DataFrame(waypoints)

    # If ETAs are not provided, speeds must be defined in reference trajectory parameters 
    ref_params = {
        'nurbs_order': 4,
        'cruise_speed': 8.0,
        'ascent_speed': 2.0,
        'descent_speed': 3.0,
        'landing_speed': 2
    }

    # Re-calculate reference trajectory 
    ref_traj_speeds = traj_gen(waypoints_pd_speeds, vehicle, **ref_params)

    # Define controller and build scheduled control. This time we'll use LQR_I
    ctrl_speeds = LQR_I(ref_traj_speeds, vehicle)
    ctrl_speeds.build_scheduled_control(vehicle.linear_model, input_vector=[vehicle.parameters['steadystate_input']])

    # Set simulation options 
    options = {
        'dt': 0.1, 
        'save_freq': vehicle_params['dt']
    }

    # Simulate vehicle to fly trajectory 
    traj_results_speeds = vehicle.simulate_to_threshold(ctrl_speeds, **options)

    # Visualize results - notice these results are slightly difference, since the speeds through the waypoints, and therefore the trajectory, are different than Example 1 and 2
    vehicle.visualize_traj(pred=traj_results_speeds, ref=ref_traj_speeds)

    # EXAMPLE 3: 
    # In this example, we just want to simulate a specific portion of the reference trajectory
    # We will simulate the second cruise interval in Example 1, i.e. waypoints 10 - 13 (where the first waypoint is index 0).
    # We will use the reference trajectory (ref_traj) and controller (ctrl) already generated in Example 1

    # First, we'll re-define the ETAs in the waypoints dictionary (since we deleted them from the waypoints in Example 2)
    waypoints['time_unix'] = np.array([1544188336, 1544188358, 1544188360, 1544188377, 1544188394, 1544188411, 1544188428, 1544188496, 1544188539, 1544188584, 1544188601, 1544188635, 1544188652, 1544188672, 1544188692])

    # Extract time information for desired interval, starting at waypoint 10 and ending at waypoint 13
    start_time = waypoints['time_unix'][10] - waypoints['time_unix'][0]
    end_time = waypoints['time_unix'][13] - waypoints['time_unix'][0]
    sim_time = end_time - start_time

    # Define initial state, x0, based on reference trajectory at start_time 
    ind = np.where(ref_traj['t'] == start_time)
    x0 = {key: ref_traj[key][ind][0] for key in ref_traj.keys()}
    vehicle.parameters['x0'] = x0

    # Define simulation parameters - note that we must define t0 as start_time since we are not starting at the default of t = 0
    options = {
        'dt': 0.1, 
        'save_freq': vehicle_params['dt'],
        't0': start_time
    }

    # Simulate starting at this initial condition from start_time to end_time
    traj_results_interval = vehicle.simulate_to(sim_time, ctrl, **options)

    # Plot results with Example 1 results to show same 
    z_1 = [traj_results.outputs[iter]['z'] for iter in range(len(traj_results.times))]
    z_4 = [traj_results_interval.outputs[iter]['z'] for iter in range(len(traj_results_interval.times))]

    fig, ax = plt.subplots()
    ax.plot(traj_results.times, z_1, '-b', label='Example 1')
    ax.plot(traj_results_interval.times, z_4, '--r', label='Example 4')
    ax.legend()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
    