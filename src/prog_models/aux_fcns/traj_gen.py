# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
from warnings import warn

from prog_models.prognostics_model import PrognosticsModel
from .traj_gen_utils import route, trajectory
from prog_models.exceptions import ProgModelInputException


def trajectory_gen_fcn(waypoints_info, **params):

    parameters = {  # Set to defaults
        # Flight information
        'flight_file': None, 
        'flight_plan': None,

        # Simulation parameters:
        'dt': 0.1, 
        'gravity': 9.81,
        'cruise_speed': None, 
        'ascent_speed': None, 
        'descent_speed': None,  
        'landing_speed': None, 
        'hovering_time': 0.0,
        'takeoff_time': 0.0, 
        'landing_time': 0.0, 
        'waypoint_weights': 10.0, 
        'adjust_eta': None, 
        'nurbs_basis_length': 2000, 
        'nurbs_order': 4, 

        # Vehicle parameters:
        'vehicle_max_speed': 15.0, #### NEED TO FIX THIS 
        'vehicle_max_roll': 0.7853981633974483, ### NEED TO FIX 
        'vehicle_max_pitch': 0.7853981633974483, #### NEED TO FIX 
        'vehicle_model': 'tarot18'
        # 'vehicle_payload': 0.0,
    }

    parameters.update(params)
    if isinstance(waypoints_info, dict):
        parameters['flight_plan'] = waypoints_info
    elif isinstance(waypoints_info, str):
        parameters['flight_file'] = waypoints_info
    else:
        raise ProgModelInputException("Waypoints have incorrect format. Must be defined as dictionary or string specifying text file.")

    # Get Flight Plan
    # ================
    # Option 1: fligh_plan, in form of dict of numpy arrays with way-points and time, is passed, while there's no file to load the flight plan
    if parameters['flight_plan'] is not None and parameters['flight_file'] == None:
        # Check for appropriate input:
        if not isinstance(parameters['flight_plan'], dict):
            raise ProgModelInputException("'flight_plan' must be a dictionary. Type {} was given".format(type(parameters['flight_plan'])))
        for flight_plan_element in parameters['flight_plan'].values():
            if not isinstance(flight_plan_element, np.ndarray):
                raise ProgModelInputException("'flight_plan' entries must be numpy arrays specifying waypoint information. Type {} was given".format(type(flight_plan_element)))
        
        # Extract data from flight plan: latitude, longitude, altitude, time stamps
        flightplan = trajectory.load.convert_dict_inputs(parameters['flight_plan'])
        lat = flightplan['lat_rad']
        lon = flightplan['lon_rad']
        alt = flightplan['alt_m']
        tstamps = flightplan['timestamp']
      
    # Option 2: a file with flight plan is passed
    elif parameters['flight_file'] != None and parameters['flight_plan'] == None:
        flightplan = trajectory.load.get_flightplan(fname=parameters['flight_file'])
        # Extract data from flight plan: latitude, longitude, altitude, time stamps
        lat, lon, alt, tstamps = flightplan['lat'], flightplan['lon'], flightplan['alt'], flightplan['timestamp']
      
    # Option 3: both file with flight plan and dictionary with flight plan are passed, throw an error. Only 1 flight plan is allowed
    elif parameters['flight_file'] != None and parameters['flight_plan'] != None:
        raise ProgModelInputException("Too many flight plan inputs - please input either flight_plan dictionary or flight_file, not both.")
      
    # Option 4: no flight plan is passed. Throw an error. 
    else:
        raise ProgModelInputException("No flight plan information supplied. Please provide flight_plan or flight_file.")
    

    # Generate route
    # ==============
    if len(tstamps) > 1:
        # ETAs specified: 
        # Check if speeds have been defined and warn user if so:
        if parameters['cruise_speed'] is not None or parameters['ascent_speed'] is not None or parameters['descent_speed'] is not None:
            warn("Speed values are ignored since ETAs were specified. To define speeds (cruise, ascent, descent) instead, do not specify ETAs.")
        route_ = route.build(lat=lat, lon=lon, alt=alt, departure_time=tstamps[0],
                                etas=tstamps,  # ETAs override any cruise/ascent/descent speed requirements. Do not feed etas if want to use desired speeds values.
                                vehicle_max_speed = parameters['vehicle_max_speed'],
                                parameters = parameters)
    else: 
        # ETAs not specified:  
        # Check that speeds have been provided:
        if parameters['cruise_speed'] is None or parameters['ascent_speed'] is None or parameters['descent_speed'] is None or parameters['landing_speed'] is None:
            raise ProgModelInputException("ETA or speeds must be provided. If ETAs are not defined, desired speed (cruise, ascent, descent, landing) must be provided.")  
        # Build route (set of way-points with associated time) using latitude, longitude, altitude, initial time stamp (takeoff time), and desired speed.
        route_ = route.build(lat=lat, lon=lon, alt=alt, departure_time=tstamps[0],
                                parameters = parameters) 

    # Generate trajectory
    # =====================
    ref_traj = trajectory.Trajectory(route=route_) # Generate trajectory object and pass the route (waypoints, eta) to it
    weight_vector=np.array([parameters['waypoint_weights'],]*len(route_.lat))      # Assign weights to each way-point. Increase value of 'waypoint_weights' to generate a sharper-corner trajectory
    ref_traj.generate(
                    dt=parameters['dt'],                                 # assign delta t for simulation
                    nurbs_order=parameters['nurbs_order'],               # nurbs order, higher the order, smoother the derivatiges of trajectory's position profile
                    gravity=parameters['gravity'],                       # m/s^2, gravity magnitude
                    weight_vector=weight_vector,                              # weight of waypoints, defined ealier from 'waypoint_weights'
                    nurbs_basis_length=parameters['nurbs_basis_length'], # how long each basis polynomial should be. Used to avoid numerical issues. This is rarely changed.
                    max_phi=parameters['vehicle_max_roll'],                   # rad, allowable roll for the aircraft
                    max_theta=parameters['vehicle_max_pitch'])                # rad, allowable pitch for the aircraft
    
    if parameters['vehicle_model'] == 'tarot18' or parameters['vehicle_model'] == 'djis1000':
        # x_ref  = np.concatenate((ref_traj.cartesian_pos, ref_traj.attitude, ref_traj.velocity, ref_traj.angular_velocity, np.expand_dims(ref_traj.time, axis=1)), axis=1).T
        x_ref = {}
        x_ref['x'] = ref_traj.cartesian_pos[:,0]
        x_ref['y'] = ref_traj.cartesian_pos[:,1]
        x_ref['z'] = ref_traj.cartesian_pos[:,2]
        x_ref['phi'] = ref_traj.attitude[:,0]
        x_ref['theta'] = ref_traj.attitude[:,1]
        x_ref['psi'] = ref_traj.attitude[:,2]
        x_ref['vx'] = ref_traj.velocity[:,0]
        x_ref['vy'] = ref_traj.velocity[:,1]
        x_ref['vz'] = ref_traj.velocity[:,2]
        x_ref['p'] = ref_traj.angular_velocity[:,0]
        x_ref['q'] = ref_traj.angular_velocity[:,1]
        x_ref['r'] = ref_traj.angular_velocity[:,2]
        x_ref['t'] = ref_traj.time
    else: 
        raise ProgModelInputException("Reference trajectory format is not yet configured for this vehicle type.")

    debug = 1
    return x_ref
