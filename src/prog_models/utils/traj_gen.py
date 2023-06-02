# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
import pandas as pd
from warnings import warn

from prog_models.utils.traj_gen_utils import route, trajectory

def trajectory_gen(waypoints=None, vehicle=None, **params):
    """
    Function to generate a flyable trajectory from coarse waypoints using the NURBS algorithm.
    The function uses the waypoints as anchor points and generates a smooth, time-parametrized position profile in cartesian coordinates, 
    px, py, and pz.
    Then, the position profile is derived to obtain velocity and acceleration profiles, and the latter is used to compute the Euler's angles
    to fly the desired trajectory.

    These time-parameterized profiles work as reference state the vehicle needs to follow to complete the trajectory.
    Non-uniform rational B-splines (NURBS) are parametric composite curves with convex hull and continuity up to the k-1 derivative 
    for a curve of degree k. The NURBS is a clamped B-spline, ensuring that the position profiles pass for the first and last waypoints.
    Given a set of n + 1 waypoints, a NURBS curve is defined as a piecewise curve described by parameter 
    u ∈ IR+ : 0 ≤ u ≤ n-k+2, where each section of the curve {[0,1],[1,2],...,[(n-k+1),(n-k+2)]} is of degree k.
    Waypoint importance is defined by "weights," which controls the distance of curve from that waypoint. However, the importance is not
    defined by the absolute value of the weights, but by the relative weight of a waypoint w.r.t. the weights of the surrounding waypoints.
    This may be a problem when many or all waypoints are important and should be passed by very closely by the vehicle.

    Therefore, the NURBS algorithm used here introduces some "fictitious" waypoints, in between the real ones, with lower weights.
    These fictitious waypoints allow the NURBS to maintain high relative weight for the real waypoints, allowing the curve to pass
    close by without sacrificing the surrounding waypoints.

    The NURBS algorithm does not automatically produce a constrained trajectory based on vehicle performance (maximum speed, acceleration, attitude rates, etc.).
    The feasibility of the trajectory is checked after it has been generated. If the check fails, the trajectory is corrected by lowering speed and acceleration profiles.

    Args:
        waypoints (pandas dataframe): specifies coarse waypoints 
            Columns must have the following headers: 1) 'lat_deg' or 'lat_rad', 2) 'lon_deg' or 'lon_rad', 3) 'alt_m' or 'alt_ft', and optionally can include 4) 'time_unix'
        vehicle (PrognosticsModel): an instance of a vehicle PrognosticsModel 
            Currently, only prog_model.models.uav_model.SmallRotorcraft is supported

    Keyword args:
        cruise_speed (float, optional): m/s, avg speed between waypoints
            Required only if ETAs are not specified
        ascent_speed (float, optional): m/s, vertical speed (up)
            Required only if ETAs are not specified
        descent_speed (float, optional): m/s, vertical speed (down)
            Required only if ETAs are not specified
        landing_speed (float, optional): m/s, landing speed when altitude < 10m
        hovering_time (float, optional): s, time to hover between waypoints
        takeoff_time (float, optional): s, additional takeoff time 
        landing_time (float, optional): s, additional landing time 
        waypoint_weights (float, optional): weights of the waypoints in NURBS calculation 
        adjust_eta: (dict, optional): specification to adjust route time 
          Dictionary with keys ['hours', 'seconds']
        nurbs_basis_length (float, optional): Length of the basis function in the NURBS algorithm
        nurbs_order (int, optional): Order of the NURBS curve

    Returns:
        ref_traj (dict[str, np.array]) 
            Reference state vector as a function of time 
    """

    # Check for waypoints and vehicle information
    if waypoints is None:
        raise TypeError("No waypoints or flight plan information were provided to generate reference trajectory.")
    if not isinstance(waypoints, pd.DataFrame):
        raise TypeError("Waypoints must be provided using Pandas DataFrame.")
    if vehicle is None:
        raise TypeError("No vehicle model was provided to generate reference trajectory.")

    # Waypoints must include at least two points
    if len(waypoints.index) <= 1:
        raise TypeError("The waypoint dataframe provided is not valid. Two or more waypoints must be provided.")

    parameters = {  # Set to defaults

        # Simulation parameters:
        'cruise_speed': None,
        'ascent_speed': None,
        'descent_speed': None,
        'landing_speed': None,
        'hovering_time': 0.0,
        'takeoff_time': 0.0, 
        'landing_time': 0.0,
        'waypoint_weights': 20.0,
        'adjust_eta': None, 
        'nurbs_basis_length': 2000, 
        'nurbs_order': 4, 
    }

    # Update parameters with any user-defined parameters 
    parameters.update(params)

    # Check if user has erroneously defined dt and provide warning 
    if 'dt' in params.keys():
        warn("Reference trajectory is generated with vehicle-defined dt value. dt = {} will used, and any user-defined value will be ignored.".format(vehicle.parameters['dt']))

    # Add vehicle-specific parameters 
    parameters['dt'] = vehicle.parameters['dt']
    parameters['gravity'] = vehicle.parameters['gravity']
    parameters['vehicle_max_speed'] = vehicle.parameters['vehicle_max_speed']
    parameters['vehicle_max_roll'] = vehicle.parameters['vehicle_max_roll']
    parameters['vehicle_max_pitch'] = vehicle.parameters['vehicle_max_pitch']
    parameters['vehicle_model'] = vehicle.parameters['vehicle_model']

    # Get Flight Plan
    # ================
    flightplan = trajectory.convert_df_inputs(waypoints)
    lat = flightplan['lat_rad']
    lon = flightplan['lon_rad']
    alt = flightplan['alt_m']
    tstamps = flightplan['timestamp']

    # Generate route
    # ==============
    if len(tstamps) > 1:
        # Case 1: ETAs specified 
        # Check if speeds have also been defined and warn user if so:
        has_speed = (parameters['cruise_speed'] is not None or parameters['ascent_speed'] is not None or parameters['descent_speed'] is not None)
        if has_speed:
            warn("Speed values are ignored since ETAs were specified. To define speeds (cruise, ascent, descent) instead, do not specify ETAs.")
        route_ = route.build(lat=lat, lon=lon, alt=alt, departure_time=tstamps[0],
                                etas=tstamps,  # ETAs override any cruise/ascent/descent speed requirements. Do not feed ETAs if want to use desired speeds values.
                                vehicle_max_speed = parameters['vehicle_max_speed'],
                                parameters = parameters)
    else: 
        # Case 2: ETAs not specified; speeds must be provided  
        # Check that speeds have been provided:
        has_all_speed = (parameters['cruise_speed'] is None or parameters['ascent_speed'] is None or parameters['descent_speed'] is None or parameters['landing_speed'] is None)
        if has_all_speed:
            raise TypeError("ETAs or speeds must be provided. If ETAs are not defined, desired speed (cruise, ascent, descent, landing) must be provided.")  
        # Build route (set of waypoints with associated time) using latitude, longitude, altitude, initial time stamp (takeoff time), and desired speed.
        route_ = route.build(lat=lat, lon=lon, alt=alt, departure_time=tstamps[0],
                                parameters = parameters) 

    # Generate trajectory
    # =====================
    ref_traj = trajectory.Trajectory(route=route_) # Generate trajectory object and pass the route (waypoints, ETA) to it
    weight_vector=np.array([parameters['waypoint_weights'],]*len(route_.lat))      # Assign weights to each waypoint. Increase value of 'waypoint_weights' to generate a sharper-corner trajectory
    ref_traj.generate(
                    dt=parameters['dt'],                                 # assign delta t for simulation
                    nurbs_order=parameters['nurbs_order'],               # NURBS order, higher the order, smoother the derivatiges of trajectory's position profile
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
