# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Auxiliary functions for trajectories and aircraft routes
"""

import numpy as np
import scipy.interpolate as interp

from prog_models.aux_fcns.traj_gen_utils import geometry as geom
from .nurbs import generate_3dnurbs, generate_intermediate_points, evaluate, knot_vector
from prog_models.aux_fcns.traj_gen_utils import load_trajectories as load


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


class Trajectory():
    """
    Trajectory class.

    This class generates a trajectory object that is used to create a smooth profile including
    position, velocity, acceleration, Euler's angles, and attitude rates.
    The profiles are used to generate the desired state of a vehicle that needs to be followed to complete the trajectory.
    """
    def __init__(self, route, gravity=9.81,  dt=0.1):
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

    def gen_cartesian_coordinates(self):
        """
        Generate waypoints cartesian coordinates in East-North-UP (ENU) reference frame given the 
        waypoints in geodetic coordinates (latitude, longitude, altitude).

        :return:        waypoint coordinates x, y, z (ENU), and ETAs in unix time.
        """
        coord = geom.Coord(self.route.lat[0], self.route.lon[0], self.route.alt[0])
        x, y, z = coord.geodetic2enu(self.route.lat, self.route.lon, self.route.alt)
        eta_unix = np.asarray([self.route.eta[item].timestamp() for item in range(len(self.route.eta))])
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
        self.route.x, self.route.y, self.route.z = x, y, z  # assign cartesian coordinates to route
        print('complete.')
        compass = geom.gen_heading_angle(self.route.lat, self.route.lon, self.route.alt)
        
        if weight_vector is None:   weight_vector = np.asarray([10,]*len(self.route.x))
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
        # Convert trajectory into geodetic coordinates
        # --------------------------------------------
        print("Convert into geodetic coordinates", end=" ")
        traj['geodetic_pos'] = geom.transform_from_cart_to_geo(traj['position'], self.route.lat[0], self.route.lon[0], self.route.alt[0])
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
