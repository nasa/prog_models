# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

from prog_models.aux_fcns.traj_gen import trajectory_gen_fcn as traj_gen
from prog_models.models.uav_model import SmallRotorcraft 
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
        vehicle = SmallRotorcraft()

        # Define waypoints 
        waypoints_dict = {}
        waypoints_dict['lat_deg']   = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719])
        waypoints_dict['lon_deg']   = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628])
        waypoints_dict['alt_ft']    = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
        waypoints_dict['time_unix'] = np.array([1544188336, 1544188358, 1544188360, 1544188377, 1544188394, 1544188411, 1544188428, 1544188496, 1544188539])
        waypoints = pd.DataFrame(waypoints_dict)

        waypoints_dict_no_time = {}
        waypoints_dict_no_time['lat_deg']   = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719])
        waypoints_dict_no_time['lon_deg']   = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628])
        waypoints_dict_no_time['alt_ft']    = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
        waypoints_no_time = pd.DataFrame(waypoints_dict_no_time)       

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
            # Waypoints defined incorrectly; must be pandas dataframe
            ref_traj = traj_gen(waypoints=True, vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Waypoints defined incorrectly; must be pandas dataframe
            ref_traj = traj_gen(waypoints=np.array([1,2,3]), vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Waypoints defined incorrectly; must be pandas dataframe
            ref_traj = traj_gen(waypoints={'lat_deg': [1, 2, 3], 'lon_deg': [1, 2, 3]}, vehicle=vehicle)
        
        # Missing information
        with self.assertRaises(TypeError):
            # Missing waypoint information 
            ref_traj = traj_gen(waypoints=pd.DataFrame({'lon_deg': np.array([1, 2, 3]), 'alt_ft': np.array([1, 2, 3]), 'time_unix': np.array([1, 2, 3])}), vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Missing waypoint information 
            ref_traj = traj_gen(waypoints=pd.DataFrame({'lat_deg': np.array([1, 2, 3]), 'alt_ft': np.array([1, 2, 3]), 'time_unix': np.array([1, 2, 3])}), vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Missing waypoint information 
            ref_traj = traj_gen(waypoints=pd.DataFrame({'lat_deg': np.array([1, 2, 3]), 'lon_deg': np.array([1, 2, 3]), 'time_unix': np.array([1, 2, 3])}), vehicle=vehicle)
        
        # Wrong units
        with self.assertRaises(TypeError):
            # Wrong units provided for waypoints
            ref_traj = traj_gen(waypoints=pd.DataFrame({'lat_nounit': np.array([1, 2, 3])}), vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Wrong units provided for waypoints
            ref_traj = traj_gen(waypoints=pd.DataFrame({'lat_rad': np.array([1, 2, 3]), 'lon_nounit': np.array([1, 2, 3])}), vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Wrong units provided for waypoints
            ref_traj = traj_gen(waypoints=pd.DataFrame({'lat_rad': np.array([1, 2, 3]), 'lon_rad': np.array([1, 2, 3]), 'alt_nounit': np.array([1, 2, 3])}), vehicle=vehicle)
        with self.assertRaises(TypeError):
            # Wrong units provided for waypoints
            ref_traj = traj_gen(waypoints=pd.DataFrame({'lat_deg': np.array([1, 2, 3]), 'lon_deg': np.array([1, 2, 3]), 'alt_m': np.array([1, 2, 3]), 'time_nounit': np.array([1, 2, 3])}), vehicle=vehicle)

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

        # Test trajectory generation functionality is generating an accurate result
        # Convert waypoints to Cartesian 
        DEG2RAD = np.pi/180.0
        FEET2MET = 0.3048
        coord = geom.Coord(lat0=waypoints['lat_deg'][0]*DEG2RAD, lon0=waypoints['lon_deg'][0]*DEG2RAD, alt0=waypoints['alt_ft'][0]*FEET2MET)
        x_ref, y_ref, z_ref = coord.geodetic2enu(waypoints['lat_deg']*DEG2RAD, waypoints['lon_deg']*DEG2RAD, waypoints['alt_ft']*FEET2MET)
        time_ref = [waypoints['time_unix'][iter] - waypoints['time_unix'][0] for iter in range(len(waypoints['time_unix']))]

        # Generate trajectory
        vehicle.parameters['dt'] = 1
        ref_traj_test = traj_gen(waypoints=waypoints, vehicle=vehicle)

        # Check that generated trajectory is close to waypoints 
        for ind, val in enumerate(time_ref):
            # Since dt is 1, the time values in ref_traj_test['t'] correspond to their index number
            self.assertAlmostEqual(x_ref[ind], ref_traj_test['x'][val], delta=45)
            self.assertAlmostEqual(x_ref[ind], ref_traj_test['x'][val], delta=45)
            self.assertAlmostEqual(x_ref[ind], ref_traj_test['x'][val], delta=45)

    def test_controllers_and_vehicle(self):

        # Controller and vehicle tests are combined so we only have to generate the reference trajectory once 

        # Instantiate vehicle 
        vehicle = SmallRotorcraft()

        # Define waypoints 
        waypoints_dict = {}
        waypoints_dict['lat_deg']   = np.array([37.09776, 37.09776, 37.09776, 37.09798, 37.09748, 37.09665, 37.09703, 37.09719, 37.09719])
        waypoints_dict['lon_deg']   = np.array([-76.38631, -76.38629, -76.38629, -76.38589, -76.3848, -76.38569, -76.38658, -76.38628, -76.38628])
        waypoints_dict['alt_ft']    = np.array([-1.9682394, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 164.01995, 0.0])
        waypoints_dict['time_unix'] = np.array([1544188336, 1544188358, 1544188360, 1544188377, 1544188394, 1544188411, 1544188428, 1544188496, 1544188539])
        waypoints = pd.DataFrame(waypoints_dict)

        # Generate reference trajectory
        ref_traj = traj_gen(waypoints=waypoints, vehicle=vehicle)

        # Controller tests:
        # -----------------
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
        with self.assertRaises(TypeError):
            res = vehicle.simulate_to(100, ctrl)

        # Vehicle Tests
        # -------------
        # Use controller created above. First must build_scheduled_control
        ctrl.build_scheduled_control(vehicle.linear_model, input_vector=[vehicle.parameters['steadystate_input']])

        # Testing appropriate input parameters: 
        with self.assertRaises(ProgModelInputException):
            vehicle_wrong = SmallRotorcraft(**{'vehicle_model': 'fakemodel'})

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

