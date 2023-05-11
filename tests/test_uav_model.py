# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest
import numpy as np
from scipy.interpolate import interp1d

from prog_models.aux_fcns.traj_gen import trajectory_gen_fcn as traj_gen
from prog_models.models.uav_model import UAVGen 
from prog_models.loading_fcns.controllers import LQR_I, LQR
from prog_models.exceptions import ProgModelInputException, ProgModelException
from prog_models.aux_fcns.traj_gen_utils import geometry as geom
from prog_models.loading_fcns.controllers import LQR_I, LQR

class TestUAVGen(unittest.TestCase):
    
    def setUp(self):
        # set stdout (so it wont print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_reference_trajectory_generation(self):

        # Define vehicle, necessary to pass to ref_traj
        vehicle = UAVGen()

        # Define waypoints 
        waypoints = {}
        waypoints['lat_deg']   = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719, 37.09719, 37.09719, 37.09748, 37.09798, 37.09776, 37.09776])
        waypoints['lon_deg']   = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628, -76.38628, -76.38628, -76.3848, -76.38589, -76.38629, -76.38629])
        waypoints['alt_ft']    = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0, 0.0, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
        waypoints['time_unix'] = np.array([1544188336, 1544188358, 1544188360, 1544188377, 1544188394, 1544188411, 1544188428, 1544188496, 1544188539, 1544188584, 1544188601, 1544188635, 1544188652, 1544188672, 1544188692])

        waypoints_no_time = {}
        waypoints_no_time['lat_deg']   = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719, 37.09719, 37.09719, 37.09748, 37.09798, 37.09776, 37.09776])
        waypoints_no_time['lon_deg']   = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628, -76.38628, -76.38628, -76.3848, -76.38589, -76.38629, -76.38629])
        waypoints_no_time['alt_ft']    = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0, 0.0, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
       
        # Incorrect number of inputs to traj_gen
        with self.assertRaises(TypeError):
            # No waypoint information provided
            ref_traj = traj_gen()
        with self.assertRaises(TypeError):
            # Only vehicle model (no waypoints) provided
            ref_traj = traj_gen(vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Only waypoints (no vehicle model) provided
            ref_traj = traj_gen(waypoints=waypoints)   

        # Waypoints defined incorrectly 
        # Wrong type for waypoints 
        with self.assertRaises(TypeError):
            # Waypoints defined incorrectly; must be dict or string specifying filename
            ref_traj = traj_gen(waypoints=True, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Waypoints defined incorrectly; must be dict or string specifying filename
            ref_traj = traj_gen(waypoints=np.array([1,2,3]), vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Waypoints defined with dictionary but values in dictionary are not numpy arrays, as required
            ref_traj = traj_gen(waypoints={'lat_deg': [1, 2, 3], 'lon_deg': [1, 2, 3]}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Waypoints defined with dictionary but values in dictionary are not numpy arrays, as required
            ref_traj = traj_gen(waypoints={'lat_deg': '1'}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Waypoints defined with dictionary but values in dictionary are not numpy arrays, as required
            ref_traj = traj_gen(waypoints={'lat_deg': np.array([1, 2, 3]), 'lon_deg': [1, 2, 3]}, vehicle=vehicle)
        
        # Missing information
        with self.assertRaises(TypeError):
            # Missing waypoint information 
            ref_traj = traj_gen(waypoints={'lon_deg': np.array([1, 2, 3]), 'alt_ft': np.array([1, 2, 3]), 'time_unix': np.array([1, 2, 3])}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Missing waypoint information 
            ref_traj = traj_gen(waypoints={'lat_deg': np.array([1, 2, 3]), 'alt_ft': np.array([1, 2, 3]), 'time_unix': np.array([1, 2, 3])}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Missing waypoint information 
            ref_traj = traj_gen(waypoints={'lat_deg': np.array([1, 2, 3]), 'lon_deg': np.array([1, 2, 3]), 'time_unix': np.array([1, 2, 3])}, vehicle=vehicle)
        
        # Wrong units
        with self.assertRaises(TypeError):
            # Wrong units provided for waypoints
            ref_traj = traj_gen(waypoints={'lat_nounit': np.array([1, 2, 3])}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Wrong units provided for waypoints
            ref_traj = traj_gen(waypoints={'lat_rad': np.array([1, 2, 3]), 'lon_nounit': np.array([1, 2, 3])}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Wrong units provided for waypoints
            ref_traj = traj_gen(waypoints={'lat_rad': np.array([1, 2, 3]), 'lon_rad': np.array([1, 2, 3]), 'alt_nounit': np.array([1, 2, 3])}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Wrong units provided for waypoints
            ref_traj = traj_gen(waypoints={'lat_deg': np.array([1, 2, 3]), 'lon_deg': np.array([1, 2, 3]), 'alt_m': np.array([1, 2, 3]), 'time_nounit': np.array([1, 2, 3])}, vehicle=vehicle)
        
        # Lengths of waypoint information don't match 
        with self.assertRaises(TypeError):
            # Incorrect length of waypoint information 
            ref_traj = traj_gen(waypoints={'lat_deg': np.array([1, 2]), 'lon_deg': np.array([1, 2, 3]), 'alt_m': np.array([1, 2, 3]), 'time_unix': np.array([1, 2, 3])}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Incorrect length of waypoint information 
            ref_traj = traj_gen(waypoints={'lat_deg': np.array([1, 2, 3]), 'lon_deg': np.array([1, 2]), 'alt_m': np.array([1, 2, 3]), 'time_unix': np.array([1, 2, 3])}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Incorrect length of waypoint information 
            ref_traj = traj_gen(waypoints={'lat_deg': np.array([1, 2]), 'lon_deg': np.array([1, 2, 3]), 'alt_m': np.array([1, 2, 3]), 'time_unix': np.array([1, 2, 3])}, vehicle=vehicle)

        # Checking correct combination of ETAs and speeds
        with self.assertRaises(TypeError):
            # No ETAs or speeds provided
            ref_traj = traj_gen(waypoints=waypoints_no_time, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Incomplete speeds provided
            ref_gen = traj_gen(waypoints=waypoints_no_time, vehicle=vehicle, **{'ascent_speed': 1})
        with self.assertRaises(TypeError):
            # Incomplete speeds provided
            ref_traj = traj_gen(waypoints=waypoints_no_time, vehicle=vehicle, **{'descent_speed': 1, 'landing_speed': 1})
        with self.assertRaises(TypeError):
            # Incomplete speeds provided
            ref_traj = traj_gen(waypoints=waypoints_no_time, vehicle=vehicle, **{'ascent_speed': 1, 'descent_speed': 1, 'cruise_speed': 1})
        ## TODO: these tests take a long time because they run the whole trajectory generation simulation, is there a more efficient way to do this? 
        # with self.assertWarns(UserWarning):
        #     # Specify both ETAs and speeds
        #     ref_traj = traj_gen(waypoints=waypoints, vehicle=vehicle, **{'cruise_speed': 1})
        # with self.assertWarns(UserWarning):
        #     # Specify both ETAs and speeds
        #     ref_traj = traj_gen(waypoints=waypoints, vehicle=vehicle, **{'ascent_speed': 1})

        # Test trajectory generation functionality 
        # Note: this test is slow, need to decide if we actually want to include it 
        # Convert waypoints to cartesian 
        # DEG2RAD = np.pi/180.0
        # FEET2MET = 0.3048
        # coord = geom.Coord(lat0=waypoints['lat_deg'][0]*DEG2RAD, lon0=waypoints['lon_deg'][0]*DEG2RAD, alt0=waypoints['alt_ft'][0]*FEET2MET)
        # x_ref, y_ref, z_ref = coord.geodetic2enu(waypoints['lat_deg']*DEG2RAD, waypoints['lon_deg']*DEG2RAD, waypoints['alt_ft']*FEET2MET)
        # time_ref = [waypoints['time_unix'][iter] - waypoints['time_unix'][0] for iter in range(len(waypoints['time_unix']))]

        # # Generate trajectory
        # vehicle.parameters['dt'] = 1
        # ref_traj_test = traj_gen(waypoints=waypoints, vehicle=vehicle)

        # # Check that generated trajectory is close to waypoints 
        # for ind, val in enumerate(time_ref):
        #     # Since dt is 1, the time values in ref_traj_test['t'] correspond to their index number
        #     self.assertAlmostEqual(x_ref[ind], ref_traj_test['x'][val], delta=45)
        #     self.assertAlmostEqual(x_ref[ind], ref_traj_test['x'][val], delta=45)
        #     self.assertAlmostEqual(x_ref[ind], ref_traj_test['x'][val], delta=45)

    def test_controllers(self):

        # Instantiate vehicle 
        vehicle = UAVGen()

        # Define waypoints 
        waypoints = {}
        waypoints['lat_deg']   = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719, 37.09719, 37.09719, 37.09748, 37.09798, 37.09776, 37.09776])
        waypoints['lon_deg']   = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628, -76.38628, -76.38628, -76.3848, -76.38589, -76.38629, -76.38629])
        waypoints['alt_ft']    = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0, 0.0, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
        waypoints['time_unix'] = np.array([1544188336, 1544188358, 1544188360, 1544188377, 1544188394, 1544188411, 1544188428, 1544188496, 1544188539, 1544188584, 1544188601, 1544188635, 1544188652, 1544188672, 1544188692])

        # Generate reference trajectory
        ref_traj = traj_gen(waypoints=waypoints, vehicle=vehicle)

        # Testing incorrect arguments:
        with self.assertRaises(TypeError):
            # Reference trajectory not defined as dict
            ctrl = LQR(x_ref='abc', vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Reference trajectory not defined as dict
            ctrl = LQR(x_ref=[1, 2, 3], vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Incorrect type for values in reference trajectory dict
            ctrl = LQR(x_ref={'x': [1, 2, 3]}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Incorrect type for values in reference trajectory dict
            ctrl = LQR(x_ref={'x': np.array([1, 2, 3]), 'y': [1, 2, 3]}, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # No vehicle model given
            ctrl = LQR(x_ref=ref_traj)
        with self.assertRaises(TypeError):
            # No reference trajectory given
            ctrl = LQR(vehicle=vehicle)

        # Test error of build_scheduled_control not called
        ctrl = LQR(x_ref=ref_traj, vehicle=vehicle)
        vehicle.parameters['ref_traj'] = ref_traj
        with self.assertRaises(TypeError):
            res = vehicle.simulate_to(100, ctrl)
    
    def test_vehicle(self):

        # Instantiate vehicle 
        vehicle = UAVGen()

        # Define waypoints 
        waypoints = {}
        waypoints['lat_deg']   = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719, 37.09719, 37.09719, 37.09748, 37.09798, 37.09776, 37.09776])
        waypoints['lon_deg']   = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628, -76.38628, -76.38628, -76.3848, -76.38589, -76.38629, -76.38629])
        waypoints['alt_ft']    = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0, 0.0, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
        waypoints['time_unix'] = np.array([1544188336, 1544188358, 1544188360, 1544188377, 1544188394, 1544188411, 1544188428, 1544188496, 1544188539, 1544188584, 1544188601, 1544188635, 1544188652, 1544188672, 1544188692])

        # Generate reference trajectory
        ref_traj = traj_gen(waypoints=waypoints, vehicle=vehicle)

        # Instantiate controller
        ctrl = LQR(ref_traj, vehicle)
        ctrl.build_scheduled_control(vehicle.linear_model, input_vector=[vehicle.vehicle_model.mass['total']*vehicle.parameters['gravity']])

        # Testing appropriate input parameters: 
        with self.assertRaises(ProgModelInputException):
            vehicle_wrong = UAVGen(**{'vehicle_model': 'fakemodel'})

        # Test exception if no reference trajectory specified in vehicle parameters
        with self.assertRaises(ProgModelException):
            res = vehicle.simulate_to(100, ctrl)



# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Trajectory Generation model")
    result = runner.run(l.loadTestsFromTestCase(TestUAVGen)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()

