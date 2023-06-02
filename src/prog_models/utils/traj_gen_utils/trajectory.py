# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Auxiliary functions for trajectories and aircraft routes
"""

import datetime as dt
import numpy as np
import scipy.interpolate as interp
import datetime as dt

from prog_models.utils.traj_gen_utils import geometry as geom
from .nurbs import generate_3dnurbs, generate_intermediate_points, evaluate, knot_vector
<<<<<<< HEAD:src/prog_models/aux_fcns/traj_gen_utils/trajectory.py
from prog_models.aux_fcns.traj_gen_utils import load_trajectories as load
from prog_models.exceptions import ProgModelInputException


def check_and_adjust_eta_feasibility(lat, lon, alt, eta, vehicle_max_speed, vehicle_max_speed_vert, distance_method='greatcircle'):
    """
    Check that ETAs are feasible by verifying that the average speed necessary to 
    reach the way-point is not beyond the maximum speed of the vehicle.
    If so, the ETA for each way-point is increased progressively until the criterion is met.

    :param lat:                         rad, n x 1, latitude of way-points
    :param lon:                         rad, n x 1, longitude of way-points
    :param alt:                         m, n x 1, altitude of way-points
    :param eta:                         s, n x 1, ETAs of way-points
    :param vehicle_max_speed:           m/s, scalar, vehicle max horizontal speed 
    :param vehicle_max_speed_vert:      m/s, scalar, vehicle max speed ascending or descending
    :param distance_method:             string, method to calculate the distance between two points, either greatcircle (default), or vicenti.
    :return:                            s, n x 1, ETAs of way-points corrected to ensure feasibility.
    """
    # Check if the speeds required for ETAs are beyond the max vehicle speeds.
    n = len(lat)-1
    d_eta = np.diff(eta)
    for point in range(n):
        dh, dv = geom.geodetic_distance([lat[point], lat[point + 1]],
                                        [lon[point], lon[point + 1]],
                                        [alt[point], alt[point + 1]],
                                        method=distance_method, return_surf_vert=True)
        dv = dv[0]
        # If speed is larger than maximum (possible when both dh, dv>0), increment d_eta to reduce until desired (consider margin)
        while dh / d_eta[point] > vehicle_max_speed or dv / d_eta[point] > vehicle_max_speed_vert:
            d_eta[point] += 1.0
    return np.asarray(np.cumsum(np.insert(d_eta, 0, 0.0)))


def reshape_route_attribute(x, dim=None, msk=None):
    """
    Reshape route attribute to ensure right length

    :param x:           attribute to reshape
    :param dim:         scalar, int or None, desired length of attribute. Used only if x has no attribute "len"
    :param msk:         list or array of int, or None, mask to reshape attribute x. The function return x by repeating x[msk] at msk points.
    :return:            reshaped attribute
    """
    if not hasattr(x, "__len__"):   
        x = x * np.ones((dim, ))
    elif msk:
        x = np.insert(x, msk, x[msk])
    return x
=======
>>>>>>> 4d9d8342d633ebdbff383c33d781a213bc6899c8:src/prog_models/utils/traj_gen_utils/trajectory.py


def angular_vel_from_attitude(phi, theta, psi, delta_t=1):
    """
    Compute angular velocities from attitudes. The function computes the body angular rates p, q, r given the 
    desired Euler's angles parameterized by time: phi(t), theta(t), psi(t), and their corresponding rates.

    :param phi:         double, n x 1, first Euler's angle as a function of time.
    :param theta:       double, n x 1, second Euler's angle as a function of time.
    :param psi:         double, n x 1, third Euler's angle as a function of time.
    :param delta_t:     double, scalar, time step size, (default=1).
    :return p:          double, n x 1, body roll rate as a function of time
    :return q:          double, n x 1, body pitch rate as a function of time
    :return r:          double, n x 1, body yaw rate as a function of time
    """
    phidot   = np.insert(np.diff(phi) / delta_t, 0, 0.0)
    thetadot = np.insert(np.diff(theta) / delta_t, 0, 0.0)
    psidot   = np.insert(np.diff(psi) / delta_t, 0, 0.0)
    p        = phidot.copy()
    q        = phidot.copy()
    r        = phidot.copy()
    for ii in range(len(phi)):
        des_angular_vel = geom.body_ang_vel_from_eulers(phi[ii], theta[ii], psi[ii], phidot[ii], thetadot[ii], psidot[ii])
        p[ii], q[ii], r[ii] = des_angular_vel[0], des_angular_vel[1], des_angular_vel[2]
    return p, q, r


def gen_from_pos_profile(px, py, pz, t, wp_etas, wp_yaw, gravity=9.81, max_phi=45/180.0*np.pi, max_theta=45/180.0*np.pi):
    """
    Generate time-parameterized trajectory velocity, acceleration, Euler's angles, and body angular rates given the position profiles,
    the waypoints ETAs, and the yaw angle between waypoints. First two Euler's angles are limited to 45 degrees.

    :param px:          m, double, n x 1, x-position profile as a function of time
    :param py:          m, double, n x 1, y-position profile as a function of time
    :param pz:          m, double, n x 1, z-position profile as a function of time
    :param t:           s, double, n x 1, time vector
    :param wp_etas:     s, double, m x 1, ETAs at waypoints
    :param wp_yaw:      rad, double, m x 1, yaw angle between waypoints
    :param gravity:     m/s^2, double, scalar, gravity magnitude
    :param max_phi:     rad, double, scalar, maximum phi allowed. Default is 45 deg.
    :param max_theta:   rad, double, scalar, maximum theta allowed. Default is 45 deg.
    :return:            dictionary of results with keys: 'position', 'velocity', 'acceleration', 'attitude', 'angVel', 'time'
    """    
    
    delta_t = t[1]-t[0]
    
    # --------- Differentiate trajectory to obtain speed and acceleration ------ #
    # Velocity
    vx = np.gradient(px, delta_t)
    vy = np.gradient(py, delta_t)
    vz = np.gradient(pz, delta_t)
    # modify to start at 0
    vx[0] = 0.
    vy[0] = 0.
    vz[0] = 0.

    # Acceleration 
    ax = np.gradient(vx, delta_t)
    ay = np.gradient(vy, delta_t)
    az = np.gradient(vz, delta_t)
    # modify to start at 0
    ax[0] = 0.
    ay[0] = 0.
    az[0] = 0.

    # --------- Calculate angular kinematics based on acceleration and yaw ---------- #
    # Yaw interpolation has to be done with zero-degree polynomial to ensure we have constant yaw over a segment
    psi = generate_smooth_yaw(wp_etas, wp_yaw, t)

    # linearized angular kinematics
    phi   = 1.0 / (gravity + az) * (ax * np.sin(psi) - ay * np.cos(psi))    # 
    theta = 1.0 / (gravity + az) * (ax * np.cos(psi) + ay * np.sin(psi))    # 
    
    # Introduce limits on attitude angles
    phi   = np.fmax(np.fmin(phi, max_phi), -max_phi)
    theta = np.fmax(np.fmin(theta, max_theta), -max_theta)

    # Get body angular velocity from attitude
    p, q, r = angular_vel_from_attitude(phi, theta, psi, delta_t)

    # --------- Create dictionary with trajectory information ---------- #
    trajectory = {'position':       np.array([px, py, pz]).T,
                  'velocity':       np.array([vx, vy, vz]).T,
                  'acceleration':   np.array([ax, ay, az]).T,
                  'attitude':       np.array([phi, theta, psi]).T,
                  'angVel':         np.array([p, q, r]).T,
                  'time':           t}
    return trajectory

def generate_smooth_yaw(etas, yaw_points, time):
    """
    Generate smooth yaw angle as a function of time given ETAs and yaw between waypoints.
    The function first generates a step-wise yaw vector for each point in the time vector.
    Then, it takes the step-wise yaw curve and uses a NURBS with order 6 to generate a smooth transition at the waypoints.
    By so doing, the yaw as a function of time remains constant when between waypoints. When the next waypoint is approaching, 
    the yaw changes with a slow rate to avoid sharp acceleration.

    :param etas:            double, n x 1, ETAs at waypoints
    :param yaw_points:      double, n x 1, yaw between waypoints
    :param time:            double, m x 1, time vector to parameterize yaw curve.
    :return:                yaw curve as a function of time.
    """
    # Use interpolation with order 0 to create step-wise yaw values as a function of time.
    yaw_fun = interp.interp1d(etas, yaw_points, kind='zero', fill_value='extrapolate')
    psi = yaw_fun(time)

    # Smooth yaw out by using NURBS with order 6
    order = 6
    t = knot_vector(len(time)-1, order)
    p_vector       = np.zeros((2, len(time)))
    p_vector[0, :] = time
    p_vector[1, :] = psi
    weight_vector = np.ones_like(time)
    psi_smooth = evaluate(order, t, weight_vector, p_vector, basis_length=2000)
    psi_smooth = interp.interp1d(psi_smooth[0,:], psi_smooth[1,:], kind='linear', fill_value='extrapolate')(time)
    return psi_smooth

# Trajectory loading functions 
# ----------------------------
def convert_df_inputs(input_df):
    """
    Functions to extract user-defined waypoint information and convert to appropriate units for trajectory generation

    Flight plan is input by user in pandas dataframe format. 
    Column headers must contain: latitude ('lat_deg' or 'lat_rad'), longitude ('lon_deg' or 'lon_rad'), altitude ('alt_ft' or 'alt_m'), and (optional) time ('time_unix')
    
    :param input_df:      pandas dataframe with columns for waypoints latitude, longitude, altitude, and optional time        
    :return:              flight plan dictionary with keys: lat (in rad), lon (in rad), alt (in m), time_unix, time_stamp.
    """
    # Conversion values 
    DEG2RAD = np.pi/180.0
    FEET2MET = 0.3048

    # Check units and return exceptions if incorrect:
    if 'lat_deg' not in input_df.columns and 'lat_rad' not in input_df.columns:
        raise TypeError("Waypoints latitude must be defined in degrees (with lat_deg) or radians (with lat_rad).")
    elif 'lon_deg' not in input_df.columns and 'lon_rad' not in input_df.columns:
        raise TypeError("Waypoints longitude must be defined in degrees (with lon_deg) or radians (with lon_rad).")
    elif 'alt_ft' not in input_df.columns and 'alt_m' not in input_df.columns:
        raise TypeError("Waypoints altitude must be defined in feet (with alt_ft) or meters (with alt_m).")
    if len(input_df.columns) > 3 and 'time_unix' not in input_df.columns:
        raise TypeError("Waypoints input includes unrecognized values. Use lat_deg, lon_deg, alt_ft, and time_unix to specify.")

    # Convert units, if necessary
    if 'lat_deg' in input_df.columns:
        lat = (input_df['lat_deg'] * DEG2RAD).to_numpy()
        lon = (input_df['lon_deg'] * DEG2RAD).to_numpy()
    else: 
        lat = (input_df['lat_rad']).to_numpy()
        lon = (input_df['lon_rad']).to_numpy()
    if 'alt_ft' in input_df.columns:
        alt = (input_df['alt_ft'] * FEET2MET).to_numpy()
    else: 
        alt = (input_df['alt_m']).to_numpy()
        
    if 'time_unix' in input_df.columns:
        time_unix = (input_df['time_unix']).to_numpy()
        timestamps = [dt.datetime.fromtimestamp(time_unix[ii]) for ii in range(len(time_unix))]
    else: 
        # If no time stamp was available from file, add current time stamp and corresponding unix time.
        timestamps = [dt.datetime.now()]
        time_unix  = [timestamps[0].timestamp()]

    return {'lat_rad': lat, 'lon_rad': lon, 'alt_m': alt, 'timestamp': timestamps, 'time_unix': time_unix}
   

class Trajectory():
    """
    Trajectory class.

    This class generates a trajectory object that is used to create a smooth profile including
    position, velocity, acceleration, Euler's angles, and attitude rates.
    The profiles are used to generate the desired state of a vehicle that needs to be followed to complete the trajectory.
    """
    def __init__(self, 
                 route, 
                 gravity=9.81,  
                 dt=0.1, 
                 lat : float = 0.0, 
                 lon : float = 0.0, 
                 alt : float = 0.0, 
                 takeoff_time=None, 
                 etas=0, 
                 landing_alt=10.5,
                 cruise_speed : float = 6.0,
                 ascent_speed : float = 3.0,
                 descent_speed : float = 3.0,
                 landing_speed : float = 1.5):
        """
        Initialization of trajectory class.

        :param route:           object of class Route
        :param gravity:         m/s^2, gravity magnitude
        :param dt:              s, time step size to generate the trajectory. Default = 0.1 s
        """

        # Store relevant route variables
        self.route            = route
        self.geodetic_pos     = []
        self.cartesian_pos    = []
        self.velocity         = []
        self.acceleration     = []
        self.attitude         = []
        self.angular_velocity = []
        self.dt               = dt
        self.time             = []
        self.wind             = None
        self.gravity          = gravity
        self.nurbs_order      = []
        self.weight_vector    = []

        # Route properties
        # ----------------
        self.waypoints = {'lat': lat, 
                          'lon': lon, 
                          'alt': alt,
                          'takeoff': takeoff_time,
                          'eta': etas}
        self.landing_altitude = landing_alt
        self.speed_parameters = {'cruise_speed': cruise_speed,
                                 'ascent_speed': ascent_speed,
                                 'descent_speed': descent_speed,
                                 'landing_speed': landing_speed}
        
        self.set_eta(eta=etas, same_wp_hovering_time=1.0)



    def gen_cartesian_coordinates(self):
        """
        Generate waypoints cartesian coordinates in East-North-UP (ENU) reference frame given the 
        waypoints in geodetic coordinates (latitude, longitude, altitude).

        :return:        waypoint coordinates x, y, z (ENU), and ETAs in unix time.
        """
        # coord = geom.Coord(self.route.lat[0], self.route.lon[0], self.route.alt[0])
        # x, y, z = coord.geodetic2enu(self.route.lat, self.route.lon, self.route.alt)
        # eta_unix = np.asarray([self.route.eta[item].timestamp() for item in range(len(self.route.eta))])

        coord = geom.Coord(self.waypoints['lat'][0], self.waypoints['lon'][0], self.waypoints['alt'][0])
        x, y, z = coord.geodetic2enu(self.waypoints['lat'], self.waypoints['lon'], self.waypoints['alt'])
        if self.waypoints['eta'] is not None:
            eta_unix = np.asarray([self.waypoints['eta'][item].timestamp() for item in range(len(self.waypoints['eta']))])
        else:
            eta_unix = None
        return x, y, z, eta_unix

    def generate(self, dt, nurbs_order, weight_vector=None, gravity=None, nurbs_basis_length=1000, max_phi=45/180.0*np.pi, max_theta=45/180.0*np.pi):
        """
        Generate trajectory given the waypoints and desired velocity or ETAs.
        
        :param dt:                      s, double, scalar, time step size
        :param nurbs_order:             -, int, scalar, order of the NURBS curve
        :param weight_vector:           -, int, n x 1, weights to assign to waypoints. If None (as default), a fixed value will be assigned.
        :param gravity:                 m/s^2, double, scalar, gravity magnitude. If None (as default), the gravity magnitude used when the trajectory was initialized will be used. 
        :param nurbs_basis_length:      -, int, scalar, length of each basis function of the NURBS. Default=1000.
        :param max_phi:                 rad, double, scalar, maximum phi allowed. Default is 45 deg.
        :param max_theta:               rad, double, scalar, maximum theta allowed. Default is 45 deg.
        """
        print('\n\n**** Generating Trajectory using NURBS ***\n===================================')
        
        # Update parameters for future use
        if gravity is not None:     self.gravity = gravity # overwrite gravity
        self.nurbs_order = nurbs_order
        self.dt          = dt
        
        print('Generate cartesian coordinates', end=" ")
        # Generate cartesian coordinates
        x, y, z, eta_unix = self.gen_cartesian_coordinates()
        # self.route.x, self.route.y, self.route.z = x, y, z  # assign cartesian coordinates to route
        print('complete.')
        # compass = geom.gen_heading_angle(self.route.lat, self.route.lon, self.route.alt)
        compass = geom.gen_heading_angle(self.waypoints['lat'], self.waypoints['lon'], self.waypoints['alt'])
        
        # if weight_vector is None:   weight_vector = np.asarray([10,]*len(self.route.x))
        if weight_vector is None:   weight_vector = np.asarray([10,]*len(self.waypoints['lat']))
        self.weight_vector = weight_vector

        # GENERATE NURBS
        # ===============
        # Add fictitious waypoints to ensure the trajectory will pass through the true waypoints
        wpx, wpy, wpz, wyaw, eta, weight_vector = generate_intermediate_points(x, y, z, compass, eta_unix-eta_unix[0], self.weight_vector)
        nurbs_curve = generate_3dnurbs(wpx, wpy, wpz, eta, dt, self.nurbs_order, weight_vector=weight_vector, basis_length=nurbs_basis_length)
        
        # Generating higher-order derivatives for velocity, acceleration profile, attitude, etc.
        # ------------------------------------------------------------------------------------
        print("Generating kinematics from position profile ", end=" ")
        traj = gen_from_pos_profile(nurbs_curve['px'], nurbs_curve['py'], nurbs_curve['pz'], nurbs_curve['time'],
                                    eta, wyaw, gravity=self.gravity, max_phi=max_phi, max_theta=max_theta)
        print('complete.')

        # Adjust trajectory if maximum acceleration exceeds the limit allowed. This prevents the vehicle to crash because of
        # high accelerations during the flight
        # -------------------------------------------------------------------------------------------------------------------
        traj = self.__adjust_eta_given_max_acceleration(waypoints=np.concatenate((wpx.reshape((-1,1)), wpy.reshape((-1,1)), wpz.reshape((-1,1))), axis=1),
                                                        etas=eta,
                                                        waypoint_yaw=wyaw,
                                                        dt=dt,
                                                        trajectory_profile=traj,
                                                        weight_vector=weight_vector,
                                                        nurbs_basis_length=nurbs_basis_length,
                                                        maxiter=10,
                                                        max_phi=max_phi,
                                                        max_theta=max_theta)


        # Convert trajectory into geodetic coordinates
        # --------------------------------------------
        print("Convert into geodetic coordinates", end=" ")
        # traj['geodetic_pos'] = geom.transform_from_cart_to_geo(traj['position'], self.route.lat[0], self.route.lon[0], self.route.alt[0])
        traj['geodetic_pos'] = geom.transform_from_cart_to_geo(traj['position'], self.waypoints['lat'][0], self.waypoints['lon'][0], self.waypoints['alt'][0])
        print('complete.')

        # Store trajectory
        # --------------------------------------------
        self.geodetic_pos     = traj['geodetic_pos']
        self.cartesian_pos    = traj['position']
        self.velocity         = traj['velocity']
        self.acceleration     = traj['acceleration']
        self.attitude         = traj['attitude']
        self.angular_velocity = traj['angVel']
        self.time             = traj['time']


    def __adjust_eta_given_max_acceleration(self, 
                                            waypoints, 
                                            etas, 
                                            waypoint_yaw, 
                                            dt, 
                                            trajectory_profile, 
                                            weight_vector, 
                                            nurbs_basis_length, 
                                            maxiter=30, 
                                            max_phi=0.25*np.pi,
                                            max_theta=0.25*np.pi):
        n = waypoints.shape[0]
        keep_adjusting = True
        counter = 0
        while keep_adjusting:
            m = len(trajectory_profile['time'])
            counter += 1
            print(f'ITERATION {counter}: acceleraiton is still too high; adjusting ETA...', end=" ")
            new_eta = etas.copy()    # store eta to be updated
            keep_adjusting = False
            for i in range(n-1):
                
                dist1 = geom.euclidean_distance_point_vector(waypoints[i, :], trajectory_profile['position'])
                dist2 = geom.euclidean_distance_point_vector(waypoints[i+1, :], trajectory_profile['position'])
                
                delta_etas = etas[i+1] - etas[i]
                dtime1 = 100.0 * np.abs(etas[i] - trajectory_profile['time'])**3.0  # time is fundamental and more important than distance, so I'm using a higher power to take that into account
                dtime2 = 100.0 * np.abs(etas[i+1] - trajectory_profile['time'])**3.0

                accelerations_abs = np.abs(trajectory_profile['acceleration'][max(np.argmin(dist1 * dtime1), 0) : min(np.argmin(dist2 * dtime2), m), :])

                if accelerations_abs.size > 0:
                    # if np.amax(accelerations_abs) > max_acceleration:
                        # extra_time = (etas[i+1] - etas[i]) * eta_perc_increment  # add 50% of the differential ETA at the waypoints 
                    acc_max = np.amax(accelerations_abs)
                    if acc_max/delta_etas > 75.0:
                        # extra_time = (delta_etas * eta_perc_increment) * (delta_etas>1.0) + 1.0 * (delta_etas <= 1.0)
                        extra_time = acc_max/75.0 + 1.0
                        new_eta[i+1:] += extra_time
                        keep_adjusting = True

            # Generate new curve and new_eta becomes the reference eta
            # ---------------------------------------------------------
            nurbs_curve = generate_3dnurbs(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], new_eta, dt, self.nurbs_order, weight_vector=weight_vector, basis_length=nurbs_basis_length)
            trajectory_profile = gen_from_pos_profile(nurbs_curve['px'], nurbs_curve['py'], nurbs_curve['pz'], nurbs_curve['time'],
                                                      etas, waypoint_yaw, gravity=self.gravity, max_phi=max_phi, max_theta=max_theta)
            etas = new_eta.copy()
            print("complete.")

            # Exit the while loop and give up if in 20 iterations the trajectory acceleration has not been pushed under the limit
            if counter == maxiter:
                print("WARNING: max number of iterations reached, the trajectory still contains accelerations beyond the limit.")
                break

        return trajectory_profile

    def set_landing_waypoints(self, set_landing_eta=False):
        """
        Set waypoints at altitude defined by landing_altitude.
        By so doing, the trajectory can split the ascending and descending phases into segments, and assign the landing_speed to all segments that
        fall below the landing_alt mark.

        :param set_landing_eta:         Bool, whether to set the landing ETAs as well. Otherwise, they will be calculated later.
        :return:                        int, n x 1 array, indices of way-points that define the landing.
        """
        
        # get boolean flag where altitude is below the landing altitude, and get the corresponding indices
        idx_land     = np.asarray(self.waypoints['alt'] < self.landing_altitude)
        idx_land_pos = np.where(idx_land)[0]    
        
        # if there are waypoints below landing altitude: append a landing way-point (with altitude self.landing_altitude) accordingly.
        if idx_land_pos.size != 0:
            n_ = len(self.waypoints['lat'])
            counter = 0
            for item in idx_land_pos:
                if item == 0:   # if first element is below, just add a landing way-point
                    self.waypoints['lat'] = np.insert(self.waypoints['lat'], item + 1, self.waypoints['lat'][item])
                    self.waypoints['lon'] = np.insert(self.waypoints['lon'], item + 1, self.waypoints['lon'][item])
                    self.waypoints['alt'] = np.insert(self.waypoints['alt'], item+1, self.landing_altitude*1.0)
                    counter += 1
                elif item == n_-1: # if one before the last element, add landing way-points right before landing.
                    if self.waypoints['alt'][item+counter-1] > self.landing_altitude:
                        self.waypoints['lat'] = np.insert(self.waypoints['lat'], -1, self.waypoints['lat'][item+counter-1])
                        self.waypoints['lon'] = np.insert(self.waypoints['lon'], -1, self.waypoints['lon'][item+counter-1])
                        self.waypoints['alt'] = np.insert(self.waypoints['alt'], -1, self.landing_altitude*1.0)
                        counter += 1
                else:
                    if self.waypoints['alt'][item+counter] - self.waypoints['alt'][item+counter-1] < 0:     # descending
                        idx_delta = 0
                    else:   # ascending
                        idx_delta = +1
                    self.waypoints['lat'] = np.insert(self.waypoints['lat'], item+counter + idx_delta, self.waypoints['lat'][item+counter])
                    self.waypoints['lon'] = np.insert(self.waypoints['lon'], item+counter + idx_delta, self.waypoints['lon'][item+counter])
                    self.waypoints['alt'] = np.insert(self.waypoints['alt'], item+counter + idx_delta, self.landing_altitude*1.0)
                    counter += 1
                    if idx_delta == 0:  # if descended, needs to go back up
                        if self.waypoints['alt'][item+counter+1] > self.landing_altitude:
                            self.waypoints['lat'] = np.insert(self.waypoints['lat'], item+counter+1, self.waypoints['lat'][item+counter+1])
                            self.waypoints['lon'] = np.insert(self.waypoints['lon'], item + counter+1, self.waypoints['lon'][item + counter+1])
                            self.waypoints['alt'] = np.insert(self.waypoints['alt'], item + counter+1, self.landing_altitude*1.0)
                            counter += 1

        # Recalculate landing positions with new waypoints:
        idx_land = np.asarray(self.waypoints['alt'] < self.landing_altitude)
        idx_land_pos = np.where(idx_land)[0]

        # Interpolate ETA at landing waypoints linearly
        if set_landing_eta:
            eta_landing = []
            counter     = 0
            for idx in idx_land_pos:
                delta_alt   = self.waypoints['alt'][idx+1+counter] - self.waypoints['alt'][idx-1+counter]
                t_0         = self.waypoints['eta'][idx-1].timestamp()
                t_1         = self.waypoints['eta'][idx].timestamp()
                delta_t     = t_1 - t_0
                delta_ratio = delta_alt / delta_t
                t_land      = 1.0/delta_ratio * ( self.landing_altitude + delta_ratio * t_0 - self.waypoints['alt'][idx-1])
                counter    += 1
                eta_landing.append(dt.datetime.utcfromtimestamp(t_land) + dt.timedelta(hours=-8))   # -8 because it's California time
            self.eta = np.insert(self.eta, idx_land_pos, eta_landing)
        return idx_land_pos

    def set_eta(self, eta=None, hovering=0, add_takeoff_time=None, add_landing_time=None, same_wp_hovering_time=1.0):
        """
        Assign ETAs to way-points.
        If ETAS are provided (i.e., eta is not None), assign them. that's it.
        If they are not provided, compute them from desired speed.

        :param eta:                     s, n x 1 array or None. ETAs for all waypoints.
        :param hovering:                s, scalar or n x 1 array. Default = 0. hovering condition to add to the way-points.
        :param add_takeoff_time         s, scalar or None, add takeoff time
        :param add_landing_time         s, scalar or None, add landing time
        :param same_wp_hovering_time    s, scalar, ancillary variable to avoid spurious accelerations when the vehicle needs to hover on the same way-point. It adds time to the ETA, which lower accelerations.
        """
        # Assign ETAS
        # ============
        if eta is not None: # if ETA is provided, assign to self.eta and that's it.
            if hasattr(eta, "__len__") is False or len(eta)!=len(self.waypoints['lat']):
                raise ProgModelInputException("ETA must be vector array with same length as lat, lon and alt.")
                
            eta_unix = np.zeros_like(eta, dtype=np.float64)
            for i, eta_i in enumerate(eta):
                eta_unix[i] = dt.datetime.timestamp(eta_i) 
            # Check if speeds required for ETAs are above max speeds            
            # if self.cruise_speed is None:       cruise_speed_val = 6.0
            # else:                               cruise_speed_val = self.cruise_speed
            # if self.ascent_speed is None:       vert_speed_val = 3.0
            # else:                               vert_speed_val = self.ascent_speed

            # Get the new relative ETA given expected ETAs and distance between waypoints
            relative_eta_new = check_and_adjust_eta_feasibility(self.waypoints['lat'], self.waypoints['lon'], self.waypoints['alt'], eta_unix-eta_unix[0], self.speed_parameters['cruise_speed']*1.2, self.speed_parameters['ascent_speed']*1.2, distance_method='greatcircle')
            self.eta = np.asarray([dt.datetime.fromtimestamp(relative_eta_new[i] + eta_unix[0]) for i in range(len(eta))])

        else:   # if ETA is not provided, compute it from desired cruise speed and other speeds
            if self.cruise_speed == None:
                raise ProgModelInputException("If ETA is not provided, desired speed (cruise, ascent, descent) must be provided.")

            idx_land_pos = self.set_landing_waypoints(set_landing_eta=False)

            if add_takeoff_time is not None:    self.takeoff_time = add_takeoff_time
            if add_landing_time is not None:    self.landing_time = add_landing_time

            # Check speed dimensions
            n = len(self.lat)
            self.cruise_speed  = reshape_route_attribute(self.cruise_speed, dim=n-1, msk=idx_land_pos)
            self.ascent_speed  = reshape_route_attribute(self.ascent_speed, dim=n-1, msk=idx_land_pos)
            self.descent_speed = reshape_route_attribute(self.descent_speed, dim=n-1, msk=idx_land_pos)
            self.hovering      = reshape_route_attribute(hovering, dim=n-1, msk=idx_land_pos)
            if self.landing_speed is None:  self.landing_speed = self.descent_speed
            else:                           self.landing_speed = reshape_route_attribute(self.landing_speed, dim=n-1, msk=idx_land_pos)

            self.eta = self.compute_etas_from_speed(takeoff_time=self.takeoff_time, landing_time=self.landing_time, hovering=self.hovering,
                                                    same_wp_hovering_time=same_wp_hovering_time)

        return self.eta

    def compute_etas_from_speed(self, hovering, takeoff_time, landing_time, distance_method='greatcircle', cruise_speed=None, ascent_speed=None, descent_speed=None, same_wp_hovering_time=1.0, assign_eta=True):
        """
        Compute the ETAs for all way-points given the desired cruise and vertical speed.

        :param cruise_speed:        m/s, cruise speed between waypoints
        :param ascent_speed:        m/s, ascent speed between waypoints
        :param descent_speed:       m/s, descent speed between waypoints
        :param hovering:            s, extra time for hovering in between waypoints
        :param takeoff_time:        s, extra time needed to take off
        :param landing_time:        s, extra time needed to land
        :param distance_method:     string, method used to compute the distance between two points, either 'greatcircle' or 'vincenty'. default = 'greatcircle'
        :return:                    s, n x 1, ETAs for all way-points.
        """
        if len(self.lat) <= 2:
            raise ProgModelInputException("At least 3 waypoints are required to compute ETAS from speed. Only {} were given.".format(len(self.lat)))
        
        # define margin on cruise speed
        # ----------------------------
        # If calculated ETA produces a speed that is larger than desired speed, we can accommodate it as long as is within this margin (%)
        cruise_speed_margin = 0.1   # %, 'extra' speed we can tolerate on cruise.
        vert_speed_margin = 0.05    # %, 'extra' speed we can tolerate on ascent/descent

        # Reshape speed
        if cruise_speed is None:    
            cruise_speed  = self.cruise_speed
        else:                       
            cruise_speed      = reshape_route_attribute(cruise_speed, dim=len(self.lat)-1)
            self.cruise_speed = cruise_speed

        if ascent_speed is None:    
            ascent_speed  = self.ascent_speed
        else:                       
            ascent_speed      = reshape_route_attribute(ascent_speed, dim=len(self.lat)-1)
            self.ascent_speed = ascent_speed

        if descent_speed is None:   
            descent_speed = self.descent_speed
        else:                       
            descent_speed      = reshape_route_attribute(descent_speed, dim=len(self.lat)-1)
            self.descent_speed = descent_speed

        if self.landing_speed is None:  
            self.landing_speed = descent_speed.copy()

        if hovering is None:        
            hovering      = self.hovering
        else:                       
            hovering      = reshape_route_attribute(hovering, dim=len(self.lat)-1)
            self.hovering = hovering

        # Compute relative ETAs
        # -------------------
        alt_for_land = self.alt[1:]
        n     = len(self.lat)-1
        d_eta = np.zeros((n,))
        for point in range(n):
            dh, dv = geom.geodetic_distance([self.lat[point], self.lat[point+1]], 
                                            [self.lon[point], self.lon[point+1]], 
                                            [self.alt[point], self.alt[point+1]], 
                                            method=distance_method, return_surf_vert=True)
            dv = dv[0]

            # Identify correct vertical speed
            if   dv > 0 and alt_for_land[point] > self.landing_altitude:    vert_speed = ascent_speed[point]
            elif dv > 0 and alt_for_land[point] <= self.landing_altitude:   vert_speed = self.landing_speed[point]
            elif dv < 0 and alt_for_land[point] >= self.landing_altitude:   vert_speed = descent_speed[point]
            elif dv < 0 and alt_for_land[point] < self.landing_altitude:    vert_speed = self.landing_speed[point]
            else:                                                           vert_speed = 0. # not moving vertically.

            # Define the correct speed:
            if np.isclose(dh + dv, 0.0):
                d_eta[point] = same_wp_hovering_time  # if there's no vertical / horizontal speed (waypoints are identical) add a default hovering value to avoid extreme accelerations.
            else:
                if np.isclose(dh, 0.):      speed_sq = vert_speed**2.0
                elif np.isclose(dv, 0.):    speed_sq = cruise_speed[point]**2.0
                else:                       speed_sq = cruise_speed[point]**2.0 + vert_speed**2.0
                d_eta[point] = np.sqrt( (dh**2.0 + dv**2.0) / speed_sq )
                # If speed is larger than desired (possible when both dh, dv>0), increment d_eta to reduce until desired (consider margin)
                while dh/d_eta[point] > (cruise_speed[point]*(1.+cruise_speed_margin)) or dv/d_eta[point] > (vert_speed*(1.+vert_speed_margin)):
                    d_eta[point] += 1.0

            if hovering[point] != 0:    d_eta[point] += hovering[point]

        d_eta[0]  += takeoff_time
        d_eta[-1] += landing_time
        eta_array = np.asarray(np.cumsum(np.insert(d_eta, 0, 0.0)))
        if assign_eta:  self.eta = [dt.datetime.fromtimestamp(eta_array[ii] + + self.waypoints['takeoff_time'].timestamp()) for ii in range(len(eta_array))]
        return self.eta