
from prog_models.models.uav_model.uav_model import UAVGen 

import numpy as np

def run_example(): 

    # Example uses waypoints from flight LaRC_20181207
    flight_name = 'LaRC_20181207'

    # Define coarse waypoints: 
    waypoints = {}
    waypoints['lat_deg'] = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719, 37.09719, 37.09719, 37.09748, 37.09798, 37.09776, 37.09776])
    waypoints['lon_deg'] = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628, -76.38628, -76.38628, -76.3848, -76.38589, -76.38629, -76.38629])
    waypoints['alt_ft'] = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0, 0.0, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
    # waypoints['time_unix'] = np.array([1544188336, 1544188358, 1544188360, 1544188377, 1544188394, 1544188411, 1544188428, 1544188496, 1544188539, 1544188584, 1544188601, 1544188635, 1544188652, 1544188672, 1544188692])

    # Parameters:
    params = {
        'flight_plan': waypoints,
        'dt': 0.1, 
        'cruise_speed': 6.0,
        'ascent_speed': 3.0,
        'descent_speed': 3.0, 
        'landing_speed': 1.5
    }

    # Generate UAV instance with waypoints defined in dictionary 
    # traj_gen = UAVGen(flight_plan=waypoints)

    # Alternatively, waypoints can be specified in a table format and saved to a text file 
    # flight_file = 'examples/uav_waypoints.txt'
    # traj_gen = UAVGen(flight_file=flight_file)

    # Try with parameters:
    traj_gen = UAVGen(**params)
    
    # No noise for testing 
    traj_gen.parameters['process_noise'] = 0
    traj_gen.parameters['measurement_noise'] = 0 

    x0_test = traj_gen.initialize()

    def future_loading(t, x=None):
        return traj_gen.InputContainer({})  # Loading defined internally 

    options = {
        'dt': 0.1,
        # 'save_freq': 0.2
        'integration_method': 'euler'
    }

    simulated_results = traj_gen.simulate_to(200,future_loading, **options)

    debug = 1


# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()