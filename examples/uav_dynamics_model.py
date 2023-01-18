# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of generating a trajectory for a rotorcraft through a set of coarse waypoints 
"""
from prog_models.models.uav_model.uav_model import UAVGen 

import numpy as np
import matplotlib.pyplot as plt

def run_example(): 

    # Example 1: generate trajectory from waypoints and ETAs 

    # Define coarse waypoints: waypoints must be defined with a dictionary of numpy arrays or as columns in a text file 
    # See documentation for specific information on inputting waypoints 
    waypoints = {}
    waypoints['lat_deg']   = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719, 37.09719, 37.09719, 37.09748, 37.09798, 37.09776, 37.09776])
    waypoints['lon_deg']   = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628, -76.38628, -76.38628, -76.3848, -76.38589, -76.38629, -76.38629])
    waypoints['alt_ft']    = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0, 0.0, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
    waypoints['time_unix'] = np.array([1544188336, 1544188358, 1544188360, 1544188377, 1544188394, 1544188411, 1544188428, 1544188496, 1544188539, 1544188584, 1544188601, 1544188635, 1544188652, 1544188672, 1544188692])

    # Define model parameters:
    params_1 = {
        'flight_plan': waypoints, # Specify waypoints 
        'dt': 0.1, # Define time step for model generation; this value is also used for simulate_to functionality 
        'vehicle_model': 'tarot18', # Define vehicle
    }

    # Create a model object, define noise
    uav_1 = UAVGen(**params_1)
    uav_1.parameters['process_noise'] = 0
    uav_1.initialize()
    
    # Define future loading function to return empty InputContainer, since there is no user-defined loading in trajectory generation 
    def future_loading_1(t, x=None):
        return uav_1.InputContainer({}) 

    # Set simulation options 
    options = {
        # 'dt': 0.1, # Note that this value isn't used internally, as simulate_to dt is defined as params_1['dt]; if dt is defined here as any other value, a warning will be returned
        'save_freq': params_1['dt']
    }

    # Generate trajectory
    traj_results_1 = uav_1.simulate_to_threshold(future_loading_1, **options)

    # Visualize results:
    # Plot reference trajectory and generated trajectory - use 'visualize_traj' function in the UAVGen class
    uav_1.visualize_traj(traj_results_1)
    
    # Plot internal states 
    traj_results_1.outputs.plot(keys = ['x', 'y', 'z'], ylabel = 'Cartesian position', title='Example 1 Predicted Outputs')
    traj_results_1.outputs.plot(keys = ['phi', 'theta', 'psi'], ylabel = 'Pitch, roll, and yaw', title='Example 1 Predicted Outputs')
    traj_results_1.outputs.plot(keys = ['vx', 'vy', 'vz'], ylabel = 'Velocities', title='Example 1 Predicted Outputs')
    traj_results_1.outputs.plot(keys = ['p', 'q', 'r'], ylabel = 'Angular velocities',title='Example 1 Predicted Outputs')


    # Example 2: generate trajectory from waypoints and speeds 
    # For this example, we define waypoints without ETAs, and instead add speed specifications 
    
    # Use same waypoints, but delete the ETAs
    del waypoints['time_unix']

    # Define model parameters - this time we must specify speeds since ETAs are not given as input
    params_2 = {
        'flight_plan': waypoints, # Specify waypoints 
        'dt': 0.1, 
        'vehicle_model': 'tarot18',
        'cruise_speed': 6.0, # Additionally specify speeds 
        'ascent_speed': 3.0,
        'descent_speed': 3.0, 
        'landing_speed': 1.5,
    }

    # Create a model object, define noise
    uav_2 = UAVGen(**params_2)
    uav_2.parameters['process_noise'] = 0

    # Define future loading function to return empty InputContainer, since there is no user-defined loading in trajectory generation 
    def future_loading_2(t, x=None):
        return uav_2.InputContainer({}) 

    # Generate trajectory
    traj_results_2 = uav_2.simulate_to_threshold(future_loading_2, **options)

    # Visualize results:
    # Plot reference trajectory and generated trajectory
    uav_2.visualize_traj(traj_results_2)


    # Example 3: generate trajectory using complex UAV model parameters and waypoint information from text file
    # The UAV model class has other optional parameters for more specific modeling. Here, we illustrate the use of a few of these parameters.

    # Define model parameters - this time we must specify speeds since ETAs are not given as input
    params_3 = {
        'flight_file': 'examples/uav_waypoints.txt', # Specify path of text file with waypoints 
        'dt': 0.1, 
        'vehicle_model': 'tarot18',
        'payload': 5.0, # kg, Add payload to vehicle
        'hovering_time': 10.0, # s, Add hovering time between each waypoint
        'final_time_buffer_sec': 15, # s, Defines acceptable time interval to reach final waypoint 
    }

    # Create a model object, define noise
    uav_3 = UAVGen(**params_3)
    uav_3.parameters['process_noise'] = 0

    # Define future loading function to return empty InputContainer, since there is no user-defined loading in trajectory generation 
    def future_loading_3(t, x=None):
        return uav_3.InputContainer({}) 

    # Generate trajectory
    traj_results_3 = uav_3.simulate_to_threshold(future_loading_3, **options)

    # Visualize results:
    # Plot reference trajectory and generated trajectory
    uav_3.visualize_traj(traj_results_3)
    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
    