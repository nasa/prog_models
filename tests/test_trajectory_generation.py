# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest
import numpy as np

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

