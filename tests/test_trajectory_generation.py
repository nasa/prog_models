# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest
import numpy as np
from scipy.interpolate import interp1d

from prog_models.models.uav_model.uav_model import UAVGen 
from prog_models.exceptions import ProgModelInputException

class TestTrajGen(unittest.TestCase):
    def setUp(self):
        # set stdout (so it wont print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_trajgen_improper_input(self):
        with self.assertRaises(ProgModelInputException):
            # No waypoint information provided
            uav = UAVGen()
        with self.assertRaises(ProgModelInputException):
            # Wrong format of waypoint information provided
            uav = UAVGen(**{'flight_plan': [1, 2, 3]})
        with self.assertRaises(ProgModelInputException):
            # Wrong format of waypoint information provided
            uav = UAVGen(**{'flight_plan': np.array([1, 2, 3])})
        with self.assertRaises(ProgModelInputException):
            # Wrong format of waypoint information provided
            uav = UAVGen(**{'flight_plan': {'lat_deg': [1, 2, 3], 'lon_deg': [1, 2, 3], 'alt_m': [1, 2, 3]}})
        with self.assertRaises(ProgModelInputException):
            # Missing waypoint information
            uav = UAVGen(**{'flight_plan': {'lat_deg': np.array([1, 2, 3])}})
        with self.assertRaises(ProgModelInputException):
            # Missing waypoint information
            uav = UAVGen(**{'flight_plan': {'lon_deg': np.array([1, 2, 3]), 'alt_ft': [1, 2, 3]}})
        with self.assertRaises(ProgModelInputException):
            # Missing waypoint information
            uav = UAVGen(**{'flight_plan': {'lat_rad': np.array([1, 2, 3]), 'alt_m': np.array([1, 2, 3])}})
        with self.assertRaises(ProgModelInputException):
            # Wrong units for waypoints
            uav = UAVGen(**{'flight_plan': {'lat_nounit': np.array([1, 2, 3]), 'lat_nounit': np.array([1, 2, 3]), 'alt_nounit': np.array([1, 2, 3])}})
        with self.assertRaises(ProgModelInputException):
            # ETAs input incorrectly
            uav = UAVGen(**{'flight_plan': {'lat_deg': np.array([1, 2, 3]), 'lat_deg': np.array([1, 2, 3]), 'alt_deg': np.array([1, 2, 3]), 'time_unix': np.array([1, 2])}})
        with self.assertRaises(ProgModelInputException):
            # Waypoints have incorrect dimensions 
            uav = UAVGen(**{'flight_plan': {'lat_deg': np.array([1, 2, 3]), 'lat_deg': np.array([1, 2]), 'alt_deg': np.array([1, 2, 3]), 'time_unix': np.array([1, 2])}})
        
        flight_dict = {'lat_deg': np.array([1, 2, 3]), 'lon_deg': np.array([1, 2, 3]), 'alt_ft': np.array([1, 2, 3]), 'time_unix': np.array([1, 2, 3])}
        flight_dict_no_time = {'lat_deg': np.array([1, 2, 3]), 'lon_deg': np.array([1, 2, 3]), 'alt_ft': np.array([1, 2, 3])}
        with self.assertRaises(ProgModelInputException):
            # Specify both flight_plan and flight_file
            uav = UAVGen(**{'flight_plan': flight_dict, 'flight_file': 'filename'})
        with self.assertRaises(ProgModelInputException):
            # No ETAs or speeds provided
            uav = UAVGen(**{'flight_plan': flight_dict_no_time})
        with self.assertRaises(ProgModelInputException):
            # Incomplete speeds provided
            uav = UAVGen(**{'flight_plan': flight_dict_no_time, 'ascent_speed': 1})
        with self.assertRaises(ProgModelInputException):
            # Incomplete speeds provided
            uav = UAVGen(**{'flight_plan': flight_dict_no_time, 'descent_speed': 1, 'landing_speed': 1})
        with self.assertRaises(ProgModelInputException):
            # Incomplete speeds provided
            uav = UAVGen(**{'flight_plan': flight_dict_no_time, 'ascent_speed': 1, 'descent_speed': 1, 'cruise_speed': 1})
        with self.assertRaises(ProgModelInputException):
            # Not enough waypoint information provided to calculate ETAs with speed 
            uav = UAVGen(**{'flight_plan': {'lat_deg': np.array([1, 2]), 'lat_deg': np.array([1, 2]), 'alt_deg': np.array([1, 2])}, 'ascent_speed': 1, 'descent_speed': 1, 'cruise_speed': 1, 'landing_speed': 1})

    def test_trajectory_ETAs(self):
        # Test that trajectory generated with ETAs input is a good approximation of reference trajectory 
        dt = 0.1

        # Set-up for testing: coarse waypoints 
        waypoints = {}  
        waypoints['lat_deg'] = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719, 37.09719, 37.09719, 37.09748, 37.09798, 37.09776, 37.09776])
        waypoints['lon_deg'] = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628, -76.38628, -76.38628, -76.3848, -76.38589, -76.38629, -76.38629])
        waypoints['alt_ft'] = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0, 0.0, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
        waypoints['time_unix'] = np.array([1544188336, 1544188358, 1544188360, 1544188377, 1544188394, 1544188411, 1544188428, 1544188496, 1544188539, 1544188584, 1544188601, 1544188635, 1544188652, 1544188672, 1544188692])

        # Create a model object, define noise
        uav = UAVGen(**{'flight_plan': waypoints, 'dt': dt})
        uav.parameters['process_noise'] = 0

        # Define future loading function to return empty InputContainer, since there is no user-defined loading in trajectory generation 
        def future_loading(t, x=None):
            return uav.InputContainer({}) 

        # Generate trajectory
        traj_gen = uav.simulate_to_threshold(future_loading, save_freq = dt) 

        # Interpolate traj_gen results
        time_ref = np.arange(0,traj_gen.times[-1]-dt, dt)
        x_pred_temp = [traj_gen.outputs[iter]['x'] for iter in range(len(traj_gen.times))]
        x_pred = interp1d(traj_gen.times,x_pred_temp)(time_ref)
        y_pred_temp = [traj_gen.outputs[iter]['y'] for iter in range(len(traj_gen.times))]
        y_pred = interp1d(traj_gen.times,y_pred_temp)(time_ref)
        z_pred_temp = [traj_gen.outputs[iter]['z'] for iter in range(len(traj_gen.times))]
        z_pred = interp1d(traj_gen.times,z_pred_temp)(time_ref)

        for ii in range(len(time_ref)):
            self.assertAlmostEqual(x_pred[ii], uav.ref_traj.cartesian_pos[ii][0], delta=4)
            self.assertAlmostEqual(y_pred[ii], uav.ref_traj.cartesian_pos[ii][1], delta=4)
            self.assertAlmostEqual(z_pred[ii], uav.ref_traj.cartesian_pos[ii][2], delta=4)

    def test_trajectory_speeds(self):
        # Test that trajectory generated with speeds input is a good approximation of reference trajectory 
        # Also add a few other model parameters for testing 
        dt = 0.1

        # Set-up for testing: coarse waypoints 
        wypts = {}  
        wypts['lat_deg'] = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719, 37.09719, 37.09719, 37.09748, 37.09798, 37.09776, 37.09776])
        wypts['lon_deg'] = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628, -76.38628, -76.38628, -76.3848, -76.38589, -76.38629, -76.38629])
        wypts['alt_ft'] = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0, 0.0, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])

        params = {
            'flight_plan': wypts,
            'cruise_speed': 7.0, # Additionally specify speeds 
            'ascent_speed': 1.5,
            'descent_speed': 1.5, 
            'landing_speed': 2.5,
            'dt': dt,
            'payload': 8.0, 
            'hovering_time': 2.0
        }

        # Create a model object, define noise
        uav = UAVGen(**params)
        uav.parameters['process_noise'] = 0

        # Define future loading function to return empty InputContainer, since there is no user-defined loading in trajectory generation 
        def future_loading(t, x=None):
            return uav.InputContainer({}) 

        # Generate trajectory
        traj_gen = uav.simulate_to_threshold(future_loading, save_freq = dt) 

        # Interpolate traj_gen results
        time_ref = np.arange(0,traj_gen.times[-1]-dt, dt)
        x_pred_temp = [traj_gen.outputs[iter]['x'] for iter in range(len(traj_gen.times))]
        x_pred = interp1d(traj_gen.times,x_pred_temp)(time_ref)
        y_pred_temp = [traj_gen.outputs[iter]['y'] for iter in range(len(traj_gen.times))]
        y_pred = interp1d(traj_gen.times,y_pred_temp)(time_ref)
        z_pred_temp = [traj_gen.outputs[iter]['z'] for iter in range(len(traj_gen.times))]
        z_pred = interp1d(traj_gen.times,z_pred_temp)(time_ref)

        for ii in range(len(time_ref)):
            self.assertAlmostEqual(x_pred[ii], uav.ref_traj.cartesian_pos[ii][0], delta=4)
            self.assertAlmostEqual(y_pred[ii], uav.ref_traj.cartesian_pos[ii][1], delta=4)
            self.assertAlmostEqual(z_pred[ii], uav.ref_traj.cartesian_pos[ii][2], delta=4)
        
# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Trajectory Generation model")
    result = runner.run(l.loadTestsFromTestCase(TestTrajGen)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()

