# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
from warnings import warn

from prog_models.prognostics_model import PrognosticsModel
from .traj_gen_utils import route, trajectory

def trajectory_gen_fcn(waypoints=None, vehicle=None, **params):
    """
    Function to generate a flyable trajectory from coarse waypoints using the NURBS algorithm 
        TODO: @Matteo, can you make this a bit more descriptive/correct?

    Required arguments:
    ------------------
        waypoints: dict[str, np.array] or str 
            This argument specifies coarse waypoints. 
            dict format must include keys: 1) 'lat_deg' or 'lat_rad', 2) 'lon_deg' or 'lon_rad', 3) 'alt_m' or 'alt_ft', and optionally can include 4) 'time_unix'
                Values for each key must be numpy arrays
            str format specifies the name of a file that includes waypoint information 
                Columns must have the following headers and be in this order: 1) 'lat_deg' or 'lat_rad', 2) 'lon_deg' or 'lon_rad', 3) 'alt_m' or 'alt_ft', and optionally can include 4) 'time_unix'

        vehicle: 
            This argument must be an instance of the UAVGen class (prog_model.models.uav_model.uav_model.UAVGen)

    Keyword arguments:
    -----------------
        cruise_speed : float
          m/s, avg speed in-between way-points; required only if ETAs are not specified
        ascent_speed : float
          m/s, vertical speed (up); required only if ETAs are not specified
        descent_speed : float
          m/s, vertical speed (down); required only if ETAs are not specified
        landing_speed : float
          m/s, landing speed when altitude < 10m
        hovering_time : Optional, float
          s, time to hover between waypoints
        takeoff_time : Optional, float
          s, additional takeoff time 
        landing_time: Optional, float
          s, additional landing time 
        waypoint_weights: Optional, float
          weights of the waypoints in nurbs calculation 
        adjust_eta: Optional, dict 
          Dictionary with keys ['hours', 'seconds'], to adjust route time
        nurbs_basis_length: Optional, float
          Length of the basis function in the nurbs algorithm
        nurbs_order: Optional, int
          Order of the nurbs curve

    Returns:
    -------
        ref_traj: dict[str, np.array]
            Reference state vector as a function of time 
    """

    # Check for waypoints and vehicle information
    if waypoints is None:
        raise TypeError("No waypoints or flight plan information were provided to generate reference trajectory.")
    if vehicle is None:
        raise TypeError("No vehicle model was provided to generate reference trajectory.")

    parameters = {  # Set to defaults

        # Simulation parameters:
        'cruise_speed': None,
        'ascent_speed': None,
        'descent_speed': None,
        'landing_speed': None,
        'hovering_time': 0.0,
        'takeoff_time': 0.0, 
        'landing_time': 0.0,
        'waypoint_weights': 20.0, # 10? 
        'adjust_eta': None, 
        'nurbs_basis_length': 2000, 
        'nurbs_order': 4, 
    }

    # Update parameters with any user-defined parameters 
    parameters.update(params)

    # Check if user has erroneously defined dt and provide warning 
    if 'dt' in params.keys():
        warn("Reference trajectory is generated with vehicle-defined dt value. dt = {} will used, and any user-define value will be ignored.".format(vehicle.parameters['dt']))

    # Add vehicle-specific parameters 
    parameters['dt'] = vehicle.parameters['dt']
    parameters['gravity'] = vehicle.parameters['gravity']
    parameters['vehicle_max_speed'] = vehicle.parameters['vehicle_max_speed']
    parameters['vehicle_max_roll'] = vehicle.parameters['vehicle_max_roll']
    parameters['vehicle_max_pitch'] = vehicle.parameters['vehicle_max_pitch']
    parameters['vehicle_model'] = vehicle.parameters['vehicle_model']

    # Get Flight Plan
    # ================
    # Option 1: waypoints in form of dict of numpy arrays 
    if isinstance(waypoints,dict): 
        for flight_plan_element in waypoints.values(): 
            if not isinstance(flight_plan_element, np.ndarray):
                raise TypeError("When specifying waypoints with type dict, must define lat/lon/alt using numpy arrays. Type {} was given".format(type(flight_plan_element)))
        
        # Extract data from flight plan: latitude, longitude, altitude, time stamps
        flightplan = trajectory.load.convert_dict_inputs(waypoints)
        lat = flightplan['lat_rad']
        lon = flightplan['lon_rad']
        alt = flightplan['alt_m']
        tstamps = flightplan['timestamp']
      
    # Option 2: a file with flight plan is passed
    elif isinstance(waypoints,str):
        flightplan = trajectory.load.get_flightplan(fname=waypoints)
        # Extract data from flight plan: latitude, longitude, altitude, time stamps
        lat, lon, alt, tstamps = flightplan['lat'], flightplan['lon'], flightplan['alt'], flightplan['timestamp']

    # Option 3: incorrect format is passed for waypoints 
    else:
        raise TypeError("Waypoints have incorrect format. Must be defined as dictionary or string specifying text file.")

    # Generate route
    # ==============
    if len(tstamps) > 1:
        # Case 1: ETAs specified 
        # Check if speeds have been defined and warn user if so:
        if parameters['cruise_speed'] is not None or parameters['ascent_speed'] is not None or parameters['descent_speed'] is not None:
            warn("Speed values are ignored since ETAs were specified. To define speeds (cruise, ascent, descent) instead, do not specify ETAs.")
        route_ = route.build(lat=lat, lon=lon, alt=alt, departure_time=tstamps[0],
                                etas=tstamps,  # ETAs override any cruise/ascent/descent speed requirements. Do not feed etas if want to use desired speeds values.
                                vehicle_max_speed = parameters['vehicle_max_speed'],
                                parameters = parameters)
    else: 
        # Case 2: ETAs not specified; speeds must be provided  
        # Check that speeds have been provided:
        if parameters['cruise_speed'] is None or parameters['ascent_speed'] is None or parameters['descent_speed'] is None or parameters['landing_speed'] is None:
            raise TypeError("ETA or speeds must be provided. If ETAs are not defined, desired speed (cruise, ascent, descent, landing) must be provided.")  
        # Build route (set of waypoints with associated time) using latitude, longitude, altitude, initial time stamp (takeoff time), and desired speed.
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
        raise TypeError("Reference trajectory format is not yet configured for this vehicle type.")

    return x_ref
