# Auxiliary functions for trajectories and aircrraft routes
#
#
import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../utilities/'))

import prog_models.models.trajectory_generation.trajectory.load_trajectories as load
from prog_models.models.trajectory_generation.trajectory.route import Route, read_routes


import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
import datetime as dt

import geometry as geom
import utils
from .nurbs import generate_3dnurbs


def angular_vel_from_attitude(phi, theta, psi, delta_t=1):
    
    phidot   = np.insert(np.diff(phi) / delta_t, 0, 0.0)
    thetadot = np.insert(np.diff(theta) / delta_t, 0, 0.0)
    psidot   = np.insert(np.diff(psi) / delta_t, 0, 0.0)
    p        = phidot.copy()
    q        = phidot.copy()
    r        = phidot.copy()
    for ii in range(len(phi)):
        # des_angular_vel = geom.body_ang_vel_from_eulers(phidot[ii], thetadot[ii], psidot[ii])
        des_angular_vel = geom.body_ang_vel_from_eulers(phi[ii], theta[ii], psi[ii], phidot[ii], thetadot[ii], psidot[ii])
        p[ii], q[ii], r[ii] = des_angular_vel[0], des_angular_vel[1], des_angular_vel[2]
    return p, q, r



def gen_from_pos_profile(px, py, pz, t, wp_etas, wp_yaw, gravity=9.81):
    
    delta_t = t[1]-t[0]
    # --------- Differentiate trajectory to obtain speed and acceleration ------ #
    # Try to get velocity starting at 0
    px[1] = 0.
    py[1] = 0.
    pz[1] = 0.
    
    # velocity
    vx = np.gradient(px, delta_t)
    vy = np.gradient(py, delta_t)
    vz = np.gradient(pz, delta_t)

    ax = np.gradient(vx, delta_t)
    ay = np.gradient(vy, delta_t)
    az = np.gradient(vz, delta_t)

    # --------- Calculate angular kinematics based on acceleration and yaw ---------- #
    # Yaw interpolation has to be done with zero-degree polynomial to ensure we have constant yaw over a segment
    yawFun = interp.interp1d(wp_etas, wp_yaw, kind='zero', fill_value='extrapolate')
    psi    = yawFun(t)

    # linearized angular kinematics
    phi   = 1.0 / (gravity + az) * (ax * np.sin(psi) - ay * np.cos(psi))    # 
    theta = 1.0 / (gravity + az) * (ax * np.cos(psi) + ay * np.sin(psi))    # 
    
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
    

def horizontal_cruise_speed_from_cartesian_speed(vx_m, vy_m, vz_m, vx_std, vy_std, vz_std, nsamps=1000, return_climb_speed=False):
    norm_rv = utils.LHS(dist='normal')
    usamps  = norm_rv(ndims=1, nsamps=nsamps, loc=vx_m, scale=vx_std)
    vsamps  = norm_rv(ndims=1, nsamps=nsamps, loc=vy_m, scale=vy_std)
    # In case standard deviations are 0, replace nan with average value
    usamps[:, vx_std==0] = vx_m[vx_std==0]
    vsamps[:, vy_std==0] = vy_m[vy_std==0]
    
    if np.all(vz_std!=0):       wsamps = norm_rv(ndims=1, nsamps=nsamps, loc=vz_m, scale=vz_std)
    else:                       wsamps = np.repeat(vz_m.reshape((-1,1)), repeats=nsamps, axis=1).T
    cruise_speed_samps = np.sqrt(usamps**2.0 + vsamps**2.0)
    cruise_avg = np.mean(cruise_speed_samps, axis=0)
    cruise_std = np.std(cruise_speed_samps, axis=0)
    if return_climb_speed:      return cruise_avg, cruise_std, np.mean(wsamps, axis=0), np.std(wsamps, axis=0)
    else:                       return cruise_avg, cruise_std


def average_speed_between_wps(wp_etas, traj_timestamps, traj_cruise_speed_avg, traj_cruise_speed_std):
    ts            = np.asarray([wp_etas[item].timestamp() for item in range(len(wp_etas))])
    ts_traj       = np.asarray([traj_timestamps[item].timestamp() for item in range(len(traj_timestamps))])
    idx_waypoints = np.asarray([np.argmin(abs(ts[item] - ts_traj)) for item in range(len(ts))])    
    n = len(idx_waypoints)
    wp_speed_avg = np.zeros((n-1,))
    wp_speed_std = np.zeros((n-1,))
    # Average speed in between waypoints
    for ii in range(len(idx_waypoints)-1):
        wp_speed_avg[ii] = np.mean(traj_cruise_speed_avg[idx_waypoints[ii] : idx_waypoints[ii+1]])
        wp_speed_std[ii] = np.mean(traj_cruise_speed_std[idx_waypoints[ii] : idx_waypoints[ii+1]])
    return wp_speed_avg, wp_speed_std


def degrees_to_decimal_from_str(str_):
    degrees_ = float(str_[:str_.find('°')])
    minutes_ = float(str_[str_.find('°')+1:str_.find("'")]) / 60.0
    seconds_ = float(str_[str_.find("'")+1:str_.find('"')-1]) / 3600.0
    sign     = 1.0
    if str_[str_.rfind('"')+1:str_.rfind(',')] == 'W' or str_[str_.rfind('"')+1:str_.rfind(',')] == 'S':  sign = -1.0
    return sign * (degrees_ + minutes_ + seconds_)
    

def feet2meters_from_str(str_):
    return 0.3048 * float(str_)


def make_list(x, n=1):
    if type(x) != list: return [x,] * n
    else:               return x


def gen_timestamps_from_etas(etas, departure_timestamp):
    # departure_timestamp = dt.datetime.strptime(departure_datetime, "%Y-%m-%d %H:%M:%S")
    eta_timestamp       = [departure_timestamp, ]
    for item in etas[1:]:   
        eta_timestamp.append(eta_timestamp[0] + dt.timedelta(seconds=item))
    return eta_timestamp


def sample_generator_4d(etas, t0, x, y, z, n_samples, timestep, lat0, lon0, alt0, name):
    wb = utils.ProgressBar(n_samples, prefix='Sampling ' + str(name) + ' with speed uncertainty (linear interpolator) ', 
                     suffix=' complete.', print_length=70)
    traj_samples = []
    if type(etas[0][0]) == dt.datetime:
        etas = np.asarray([etas[ii][0].timestamp() for ii in range(etas.shape[0])]).reshape((-1,1))
    for samp in range(n_samples):
        t   = etas[:, samp] - etas[0,samp]
        t  += t0
        t_i = np.arange(start=t[0], stop=t[-1], step=timestep)
        x_i = interp.interp1d(t, x, kind='linear')(t_i)
        y_i = interp.interp1d(t, y, kind='linear')(t_i)
        z_i = interp.interp1d(t, z, kind='linear')(t_i)
        geodetic_pos = geom.transform_from_cart_to_geo(np.hstack((x_i.reshape((-1,1)), y_i.reshape((-1,1)), z_i.reshape((-1,1)))), 
                                                       lat0, lon0, alt0)
        traj_samples.append(dict(name=name, geodetic_pos=geodetic_pos, 
                                 px=x_i, py=y_i, pz=z_i, time=t_i))
        wb(samp)
    wb(n_samples)
    return traj_samples


def sample_generator_4d_fixed_t(etas, t0, x, y, z, n_samples, t_i, lat0, lon0, alt0, name):
    wb = utils.ProgressBar(n_samples, prefix='Sampling ' + str(name) + ' with speed uncertainty (linear, fixed time vector) ', suffix=' complete.', print_length=50)
    traj_samples = []
    ts_ = etas - np.repeat(etas[0,:].reshape((1,-1)), repeats=etas.shape[0], axis=0) + t0
    for samp in range(n_samples):
        t = ts_[:, samp]
        # Out of bounds points will be filled with nan automatically
        x_i = interp.interp1d(t, x, kind='linear', bounds_error=False)(t_i).reshape((-1, 1))
        y_i = interp.interp1d(t, y, kind='linear', bounds_error=False)(t_i).reshape((-1, 1))
        z_i = interp.interp1d(t, z, kind='linear', bounds_error=False)(t_i).reshape((-1, 1))
        geodetic_pos = geom.transform_from_cart_to_geo(np.hstack((x_i, y_i, z_i)), lat0, lon0, alt0)
        traj_samples.append(dict(name=name, geodetic_pos=geodetic_pos, px=x_i, py=y_i, pz=z_i, time=t_i))
        wb(samp)
    wb(n_samples)
    return traj_samples


class Trajectory():
    def __init__(self, name, route, gravity=9.81, airspeed_std=0., dt=0.1):
        self.name = name

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
        self.airspeed_std     = airspeed_std # m/s

    def set_airspeed_std(self, s):
        self.airspeed_std = s

    def gen_cartesian_coordinates(self):
        coord = geom.Coord(self.route.lat[0], self.route.lon[0], self.route.alt[0])
        x, y, z = coord.geodetic2enu(self.route.lat, self.route.lon, self.route.alt)
        eta_unix = np.asarray([self.route.eta[item].timestamp() for item in range(len(self.route.eta))])
        return x, y, z, eta_unix

    def generate(self, dt, nurbs_order, weight_vector=None, gravity=None, nurbs_basis_length=1000):
        
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
        compass = geom.gen_heading_angle(self.route.lat, self.route.lon)
        
        if weight_vector is None:   weight_vector = np.asarray([10,]*len(self.route.x))
        self.weight_vector = weight_vector

        # Generating nurbs
        # -----------------
        nurbs_curve = generate_3dnurbs(x, y, z, compass, eta_unix - eta_unix[0], 
                                       dt, self.nurbs_order, weightVector=self.weight_vector, basis_length=nurbs_basis_length)
        
        # Generating higher-order derivatives for velocity, acceleration profile, attitude, etc.
        # ------------------------------------------------------------------------------------
        print("Generating kinematics from position profile ", end=" ")
        traj = gen_from_pos_profile(nurbs_curve['px'], nurbs_curve['py'], nurbs_curve['pz'], nurbs_curve['time'],
                                         nurbs_curve['weta'], nurbs_curve['wyaw'], gravity=self.gravity)
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
        pass 


    def __gen_groundspeed(self, airspeed_x_std=None, airspeed_y_std=None, nsamps=1000):
        
        # For now, works only in 2d (x, y)
        if self.velocity==[]:
            eta_unix = np.asarray([self.route.eta[ii].timestamp() for ii in range(len(self.route.eta))])
            xdot, ydot = np.diff(self.route.x)  / np.diff(eta_unix), np.diff(self.route.x)  / np.diff(eta_unix)
        else:
            xdot, ydot = self.velocity[:, 0], self.velocity[:, 1]
        
        norm_rv = utils.LHS(dist='normal')
        if self.wind:
            um, us = self.wind['u']['mean'], self.wind['u']['std']
            vm, vs = self.wind['v']['mean'], self.wind['v']['std']
            usamps = norm_rv(ndims=1, nsamps=nsamps, loc=um, scale=us)
            vsamps = norm_rv(ndims=1, nsamps=nsamps, loc=vm, scale=vs)
        else:
            print('WARNING: Trajectory ' + self.name + ' does not contain wind information. Samples will not include groundspeed.')
            usamps = 0.
            vsamps = 0.

        if (airspeed_x_std is not None and airspeed_y_std is not None) and (airspeed_x_std != 0  and airspeed_y_std != 0):
            if hasattr(airspeed_x_std, '__len__'):  assert len(airspeed_x_std)==len(xdot), "Average airspeed xdot and standard deviation airspeed_x_std must have same length."
            else:                                   airspeed_x_std *= np.ones((len(xdot),))
            if hasattr(airspeed_y_std, '__len__'):  assert len(airspeed_y_std)==len(ydot), "Average airspeed ydot and standard deviation airspeed_y_std must have same length."
            else:                                   airspeed_y_std *= np.ones((len(ydot),))
            xdot_samps = norm_rv(ndims=1, nsamps=nsamps, loc=xdot, scale=airspeed_x_std)
            ydot_samps = norm_rv(ndims=1, nsamps=nsamps, loc=ydot, scale=airspeed_y_std)
        else:
            xdot_samps = np.repeat(xdot.reshape((1, -1)), repeats=nsamps, axis=0)
            ydot_samps = np.repeat(ydot.reshape((1, -1)), repeats=nsamps, axis=0)

        gs_x_samps  = usamps + xdot_samps
        gs_y_samps  = vsamps + ydot_samps
        gs_x_m   = np.mean(gs_x_samps, axis=0)
        gs_y_m   = np.mean(gs_y_samps, axis=0)
        gs_hor_m = np.mean(np.sqrt(gs_x_samps**2.0 + gs_y_samps**2.0), axis=0)

        gs_x_s   = np.std(gs_x_samps, axis=0)
        gs_y_s   = np.std(gs_y_samps, axis=0)
        gs_hor_s = np.std(np.sqrt(gs_x_samps**2.0 + gs_y_samps**2.0), axis=0)
        return {'x': {'m': gs_x_m, 's': gs_x_s}, 'y': {'m': gs_y_m, 's': gs_y_s}, 'cruise': {'m': gs_hor_m, 's': gs_hor_s}}

    def __average_speed_between_wps(self, nsamps=1000):

        ground_speed = self.__gen_groundspeed(nsamps=nsamps, airspeed_x_std=self.airspeed_std/np.sqrt(2),
                                              airspeed_y_std=self.airspeed_std/np.sqrt(2))
        etas  = self.route.eta
        wp_ts = np.asarray([etas[item].timestamp() for item in range(len(etas))])
        vxm, vxs = ground_speed['x']['m'], ground_speed['x']['s']
        vym, vys = ground_speed['y']['m'], ground_speed['y']['s']
        if self.airspeed_std != 0:
            vmag_m, _, vmag_s, _ = utils.xy2magdir(xm=vxm, ym=vym, xs=vxs, ys=vys, nsamps=nsamps)
        
        if len(vxm) == len(wp_ts)-1:
            return np.sqrt(vxm**2.0 + vym**2.0), 0.

        traj_ts       = np.asarray([self.time[item] + self.route.departure_time.timestamp() for item in range(len(self.time))])
        idx_waypoints = np.asarray([np.argmin(abs(wp_ts[item] - traj_ts)) for item in range(len(wp_ts))])    
        # Remove duplicates from index vector; this is likely caused by a trajectory built with a dt too large.
        for ii in range(len(idx_waypoints)):
            diff_idx = np.diff(idx_waypoints[ii-1:ii+1])
            if diff_idx <= 0:       idx_waypoints[ii] += 1 - diff_idx
            
        n             = len(idx_waypoints)
        wp_speed_avg  = np.zeros((n-1,))
        wp_speed_std  = np.zeros((n-1,))
        for ii in range(n-1):
            if idx_waypoints[ii] >= len(traj_ts):
                print("WARNING: Trajectory " + str(self.name) + " degenerates at end point. This is likely driven by a dt that was too big. For best results please reduce dt.")
                wp_speed_avg[ii] = vmag_m[-1]
                wp_speed_std[ii] = vmag_s[-1]
            else:
                wp_speed_avg[ii] = np.mean(vmag_m[idx_waypoints[ii] : idx_waypoints[ii+1]])
                wp_speed_std[ii] = np.mean(vmag_s[idx_waypoints[ii] : idx_waypoints[ii+1]])
        return wp_speed_avg, wp_speed_std
        
    def __gen_etas_with_groundspeed(self, distance_method='greatcircle', speed_sampling_size=1000, return_average=True):
        new_cruise_speed_avg, new_cruise_speed_std = self.__average_speed_between_wps(nsamps=speed_sampling_size)
        if not hasattr(new_cruise_speed_std, '__len__') and np.isclose(new_cruise_speed_std, 0.):
            return_average = True
        if return_average:    # use lognormal transform to respect 0-limit.
            etas = self.route.compute_etas_from_speed(self.route.hovering, self.route.takeoff_time, self.route.landing_time, 
                                                      distance_method=distance_method, cruise_speed=new_cruise_speed_avg, assign_eta=False)  
            etas = np.asarray(etas)
        else:
            mu     = np.log( new_cruise_speed_avg**2.0 / np.sqrt(new_cruise_speed_avg**2.0 + new_cruise_speed_std**2.0) )
            sigma2 = np.log(1.0 + new_cruise_speed_std**2.0/new_cruise_speed_avg**2.0 )
            etas   = np.zeros((len(new_cruise_speed_avg)+1, speed_sampling_size))   # ETA samples: [number of waypoints , number of sample]
            for samp in range(speed_sampling_size):
                speed_sample = np.exp(np.random.normal(loc=mu, scale=np.sqrt(sigma2))) # generate speed samples for each segment in-between way-points
                etas[:, samp] = self.route.compute_etas_from_speed(self.route.hovering, self.route.takeoff_time, self.route.landing_time, 
                                                                   distance_method=distance_method, cruise_speed=speed_sample, assign_eta=False)    
        return etas

    def gen_samples(self, **kwargs):
        
        params = dict(n_samples=100, distance_method='greatcircle', generator_type='linear', 
                      return_average=False, dt=self.dt, t_i = None, nurbs_basis_length=1000)
        params.update(kwargs)
        
        if any([c not in list(self.route.__dict__.keys()) for c in ['x', 'y', 'z']]):
            x, y, z, eta_unix = self.gen_cartesian_coordinates()
            self.route.x, self.route.y, self.route.z = x, y, z  # assign cartesian coordinates to route
        else:
            x, y, z = self.route.x, self.route.y, self.route.z

        departure_timestamp = self.route.departure_time.timestamp()
        compass = geom.gen_compass_angles(np.vstack((x, y, z)).T)
        etas    = self.__gen_etas_with_groundspeed(distance_method=params['distance_method'], 
                                                   speed_sampling_size=params['n_samples'], return_average=params['return_average'])
        if params['return_average'] or len(etas.shape)==1:  
            n_samples = 1
            etas      = etas.reshape((-1,1))
        else:               
            n_samples = etas.shape[1]
        
        if params['generator_type']=='linear':
            if params['t_i'] is not None:   
                # if a vector of time instants is given, then generate trajectories at all those fixed time instants 
                print(' *** Generating samples (linear interpolation) at fixed time instants *** ')
                trajectory_samples = sample_generator_4d_fixed_t(etas, departure_timestamp, x, y, z, n_samples, params['t_i'], 
                                                                 self.route.lat[0], self.route.lon[0], self.route.alt[0], self.name)
            else:
                print(' *** Generating samples (linear interpolation) at variable time instants *** ')
                trajectory_samples = sample_generator_4d(etas, departure_timestamp, x, y, z, n_samples, params['dt'], 
                                                         self.route.lat[0], self.route.lon[0], self.route.alt[0], self.name)

        elif params['generator_type'] == 'nurbs':
            print(' *** Generating samples (NURBS-based) at variable time instants *** ')
            trajectory_samples = []
            wb = utils.ProgressBar(n_samples, prefix='Sampling ' + str(self.name) + ' with speed uncertainty (NURBS generator) ', suffix=' complete.', print_length=70)
            for samp in range(n_samples):
                temp = generate_3dnurbs(x, y, z, compass, etas[:, samp] - etas[0, samp], 
                                        params['dt'], self.nurbs_order, weightVector=self.weight_vector, basis_length=params['nurbs_basis_length'])
                temp.pop("wyaw")
                temp.pop("weta")
                temp['geodetic_pos'] = geom.transform_from_cart_to_geo(np.hstack((temp['px'].reshape((-1,1)), temp['py'].reshape((-1,1)), temp['pz'].reshape((-1,1)))), 
                                                                       self.route.lat[0], self.route.lon[0], self.route.alt[0])
                temp['name'] = self.name
                trajectory_samples.append(temp)
                wb(samp)
            wb(n_samples)

        return trajectory_samples


    def display(self, what='position', coord='geodetic', **kwargs):
        
        params = dict(figsize=(9,8), label_fontsize=12, xlabel='time stamps', ylabel=['x (East), m', 'y (North), m', 'z (Up), m'],
                      labels=['lat', 'lon', 'alt'], linewidth=2, alpha=0.75)
        params.update(kwargs)

        if what=='position':
            if coord=='geodetic':
                x = self.geodetic_pos[:, 0] * 180.0/np.pi
                y = self.geodetic_pos[:, 1] * 180.0/np.pi
                z = self.geodetic_pos[:, 2]
                params['ylabel'] = ['lat, deg', 'lon, deg', 'alt, m']
                xwp, ywp, zwp = np.asarray(self.route.lat) * 180.0/np.pi, np.asarray(self.route.lon) * 180.0/np.pi, np.asarray(self.route.alt)
                
            elif coord=='cartesian':
                x = self.cartesian_pos[:, 0]
                y = self.cartesian_pos[:, 1]
                z = self.cartesian_pos[:, 2]
                params['labels'] = ['x', 'y', 'z']
                coord = geom.Coord(self.route.lat[0], self.route.lon[0], self.route.alt[0])
                xwp, ywp, zwp = coord.geodetic2enu(self.route.lat, self.route.lon, self.route.alt)
        elif what == 'velocity':
            x = self.velocity[:, 0]
            y = self.velocity[:, 1]
            z = self.velocity[:, 2]
            params['ylabel'] = ['x, m/s', 'y, m/s', 'z, m/s']
            params['labels'] = ['x', 'y', 'z']
        elif what == 'attitude':
            x = self.attitude[:, 0]
            y = self.attitude[:, 1]
            z = self.attitude[:, 2]
            params['ylabel'] = [r'$\phi$, rad', r'$\theta$, rad', r'$\psi$, rad']
            params['labels'] = [r'$\phi$, rad', r'$\theta$, rad', r'$\psi$, rad']
        elif what == 'angular_velocity':
            x = self.angular_velocity[:, 0]
            y = self.angular_velocity[:, 1]
            z = self.angular_velocity[:, 2]
            params['ylabel'] = ['p, deg/s', 'q, deg/s', 'r, deg/s']
            params['labels'] = ['p', 'q', 'r']

        figs = []
        if what=='position':
            fig_3d = plt.figure(figsize=params['figsize'])
            ax1 = fig_3d.add_subplot(111, projection='3d')
            ax1.plot(xwp, ywp, zwp, 'o', alpha=1.0, linewidth=params['linewidth'], label='way-points')
            ax1.plot(  x,   y,   z,  '-', alpha=params['alpha'], linewidth=params['linewidth'], label='traj')
            ax1.set_xlabel(params['ylabel'][0], fontsize=params['label_fontsize'])
            ax1.set_ylabel(params['ylabel'][1], fontsize=params['label_fontsize'])
            ax1.set_zlabel(params['ylabel'][2], fontsize=params['label_fontsize'])
            figs.append(fig_3d)

        traj_tvec = [self.route.departure_time + dt.timedelta(0, self.time[item]) for item in range(len(self.time))]
        
        fig_unrolled = plt.figure(figsize=params['figsize'])
        ax2 = fig_unrolled.add_subplot(311)
        ax3 = fig_unrolled.add_subplot(312)
        ax4 = fig_unrolled.add_subplot(313)

        if what=='position':
            ax2.plot_date(self.route.eta, xwp, fmt='-o', alpha=params['alpha'], linewidth=params['linewidth'], label='way-points')
            ax3.plot_date(self.route.eta, ywp, fmt='-o', alpha=params['alpha'], linewidth=params['linewidth'], label='way-points')
            ax4.plot_date(self.route.eta, zwp, fmt='-o', alpha=params['alpha'], linewidth=params['linewidth'], label='way-points')

        ax2.plot_date(     traj_tvec,   x,  fmt='-', alpha=params['alpha'], linewidth=params['linewidth'], label='traj')
        ax3.plot_date(     traj_tvec,   y,  fmt='-', alpha=params['alpha'], linewidth=params['linewidth'], label='traj')
        ax4.plot_date(     traj_tvec,   z,  fmt='-', alpha=params['alpha'], linewidth=params['linewidth'], label='traj')
        ax4.set_xlabel(params['xlabel'], fontsize=params['label_fontsize'])
        ax2.set_ylabel(params['ylabel'][0], fontsize=params['label_fontsize'])
        ax3.set_ylabel(params['ylabel'][1], fontsize=params['label_fontsize'])
        ax4.set_ylabel(params['ylabel'][2], fontsize=params['label_fontsize'])
        figs.append(fig_unrolled)
        
        return figs



def compute_eta_bounds(eta, eta_timestamp, avg_speed, var_vel, minVnorm=1.0, dv_ratio=0.1, alpha_bound=1.0):
    """
    Compute the interval associated to each estimated time of arrival (ETA) based
    on the average speed in between way-points and the expected variance of the velocity.
    Assuming Gaussian variables for ETA and velocities, the error interval defined by
    \delta can be interpreted as standard deviation.

    The error on the ETA is calculated as:
                                E[t_i] \delta V_j
            \delta t_{i, j} = --------------------------
                                    ||V||_2

    Where:
    E[t_i]              ETA at way-point i in direction j (j=x, y, or z)
    \delta V_j          error (or standard deviation) of velocity along direction j
    ||V||_2             norm-2 of the mean velocity vector between way-points i and i+1

    \delta t has the same measurement unit of E[t].

    Two conditions are needed effective error-interval propagation:
    1) the standard deviation (or variance) of the speed, \delta V_j, in each direction j should be "sufficiently small"
    when compared to the velocity absolute value (in this case, use ||V||_2).
    2) the norm of the velocity vector should be "sufficiently" larger than 0 to introduce non-negligible uncertainty.

    The two properties are tied to each other, since a very small ||V||_2 needs to be accompained by a (much) smaller
    \delta V.

    The input parameters minVnorm (=1.0 m/s by default) and dv_ratio (=0.05 by default) help forcing the two properties.

    if ||V||_2 < minVnorm, then uncertainty is not propagated
    if \sigma_{v_j} / ||V||_2 > maxVvar, then maxVvar is propagated

    :param eta_wp:              (n,) array, ETA at each waypoint
    :param velocity_var:        (3,) array, variance of the velocity vector in x, y and z directions
    :param velocity_avg:        (n,) array, average velocity in-between way-points
    :param minVnorm:            scalar, m/s, minimum value of Vnorm to consider uncertainty (default is 0.5 m/s).
    :param dv_ratio:            scalar, maximum value of \sigma_v / v to consider the error propagation a valid equation
    :param alpha_bound:         multiplier to define inferior and superior bounds of ETA (default = 1.0, corresponding to 1 standard deviation)
    :return ETA_inf:            (n,) array, ETA inferior bound
    :return ETA_sup:            (n,) array, ETA superior bound
    """

    # Initialize differential time of arrival Et, and variance of time of arrival Vt
    Et = np.diff(eta)  # Et is calculated as ETA @ way-point i minus ETA @ way-point i-1. It's the same for each dimension x, y and z.
    Vt = np.zeros((len(eta),))  # variance of ETA at each waypoint in x-y-z (because we have different speed in each direciton x-y-z)
    for ii in range(len(eta) - 1):
        # Et[ii + 1] = eta_wp[ii + 1] - eta_wp[ii]  # Compute Et
        Vnorm2 = np.linalg.norm(avg_speed[ii, :])**2.0
        # if ||V||_2 > minVnorm m/s, then compute uncertainty. Otherwise neglect contribution
        # if ||V||_2 is too small, this waypoint does not contribute to uncertainty
        if np.sqrt(Vnorm2) > minVnorm:      Vt[ii + 1] = time_variance_norm(Et[ii], Vnorm2, np.linalg.norm(var_vel), dv_ratio)
        else:                               Vt[ii + 1] = 0.0
    
    # Compute ETA inferior and superior bounds
    # ------------------------------------------
    # First check if Vt is a matrix (that means that second dimension is 1)
    if len(Vt.shape) == 2:      Vt = np.max(Vt, axis=1)

    cum_time_error = np.cumsum(alpha_bound * np.sqrt(Vt))
    eta_inf   = eta - cum_time_error   # ETA inferior bound
    eta_sup   = eta + cum_time_error   # ETA superior  bound
    eta_inf_timestamp = []
    eta_sup_timestamp = []
    for idx, deltatime in enumerate(cum_time_error):
        eta_inf_timestamp.append(eta_timestamp[idx] - dt.timedelta(seconds=deltatime))
        eta_sup_timestamp.append(eta_timestamp[idx] + dt.timedelta(seconds=deltatime))
    return eta_inf, eta_sup, eta_inf_timestamp, eta_sup_timestamp


def compute_average_speed(t, eta, vx, vy, vz):
    """
    Compute average speed in-between way-points given the ETAs and the planned velocity as a function of time.

    :param t:           (n,) array, trajectory time vector
    :param eta:         (m,) array, ETA at each way-point along the trajectory
    :param vx:          (n,) array, x-axis velocity along the trajectory
    :param vy:          (n,) array, y-axis velocity along the trajectory
    :param vz:          (n,) array, z-axis velocity along the trajectory
    :return:            (m, 3) array, average velocity in-between way-points over x, y and z axes.
    """
    v0 = np.zeros((len(eta), 3))
    for index in range(len(eta) - 1):
        idx0 = np.argmin(abs(t - eta[index]))
        idx1 = np.argmin(abs(t - eta[index + 1]))

        v0[index, 0] = np.mean(vx[idx0:idx1])
        v0[index, 1] = np.mean(vy[idx0:idx1])
        v0[index, 2] = np.mean(vz[idx0:idx1])
    return v0





def time_variance_norm(t, norm_v_squared, var_v, max_dv_ratio):
    """
    Compute the variance of the time of arrival using the norm of the variances.
    The function returns:

            \sigma_t^2 = t^2 ||\sigma_v^2|| / ||v||^2

    where:
        t           is the average travel time from previous to current way-point
        \sigma_v^2  is the variance of the velocity vector, which is itself a 3x1 vector for variance in (x, y, z)
        v           is the velocity vector along (x, y, z)

    :param t:                   scalar, average travel time, s
    :param norm_v_squared:      scalar, squared norm of the velocity vector, (m/s)^2
    :param var_v:               vector of velocity variances or norm of such vector (m/s)^2
    :param max_dv_ratio:        maximum value of \sigma_v / v ratio.
    :return:                    variance of travel time, [s^2].
    """
    var_v = np.linalg.norm(var_v)
    # if ratio of \sigma_v/v is not too large, use error interval propagation to compute time variance
    # if ratio of \sigma_v/v is too large, then propagate max_dv_ratio only.
    if np.sqrt(var_v / norm_v_squared) < max_dv_ratio:      return compute_time_variance(t, var_v, norm_v_squared)
    else:                                                   return compute_time_variance(t, max_dv_ratio, norm_v_squared)

def compute_time_variance(t, dv2, v2):
    """
    Compute the travel time variance \sigma_t^2 given the travel time t, the variance of the velocity dv2, and the
    square of the velocity v2.
    :param t:               scalar, [s], travel time
    :param dv2:             scalar, [(m/s)^2], variance of velocity
    :param v2:              scalar, [(m/s)^2], squared of velocity
    :return:                scalar, variance of travel time, s^2
    """
    return t**2.0 * dv2/v2



# GROUND SPEED FUNCTIONS
# ======================
def compute_ground_speed_xy(traj_vel_xy, windspeed_uv, windspeed_std_uv, alpha=3):
    vx, vy = traj_vel_xy[:, 0], traj_vel_xy[:, 1]
    wx_avg, wy_avg = windspeed_uv[:, 0], windspeed_uv[:, 1]
    wx_low = windspeed_uv[:, 0] - alpha * windspeed_std_uv[:, 0]
    wy_low = windspeed_uv[:, 1] - alpha * windspeed_std_uv[:, 1]
    wx_up  = windspeed_uv[:, 0] + alpha * windspeed_std_uv[:, 0]
    wy_up  = windspeed_uv[:, 1] + alpha * windspeed_std_uv[:, 1]

    g_vx_avg = vx + wx_avg
    g_vy_avg = vy + wy_avg

    g_vx_low = vx + wx_low
    g_vy_low = vy + wy_low

    g_vx_up = vx + wx_up
    g_vy_up = vy + wy_up
    return g_vx_avg, g_vy_avg, g_vx_low, g_vy_low, g_vx_up, g_vy_up


def compute_new_cruise_speed(cruisespeed, groundspeed):
    """ Compute new cruise speed based on input ground speed """
    mean_cs, max_cs = np.mean(cruisespeed), np.max(cruisespeed)
    ratio           = (np.mean(groundspeed) - mean_cs)/ mean_cs
    return max_cs + ratio * max_cs

def from_air_wind_to_groundspeed(u, v, w, wx, wy, wz=0.0):
    n   = len(u['m'])
    g_m = np.zeros((n, 3))
    g_s = np.zeros((n, 3))
    # Compute average ground speed in cartesian coordinates
    g_m[:, 0] = u['m'] + wx['m']
    g_m[:, 1] = v['m'] + wy['m']
    g_m[:, 2] = w['m'] + wz['m']
    # Compute standard deviation of ground speed in cartesian coordinates
    g_s[:, 0] = np.sqrt(u['std']**2.0 + wx['std']**2.0)
    g_s[:, 1] = np.sqrt(v['std']**2.0 + wy['std']**2.0)
    g_s[:, 2] = np.sqrt(w['std']**2.0 + wz['std']**2.0)
    return {'m': g_m, 'std': g_s}


def get_ground_speed_for_routes(routes, wind_uv, windstd_uv):
    routes             = make_list(routes)
    new_ref_speed_mean = []
    new_ref_speed_std  = []
    for route in routes:
        # Extract trajectory velocity values
        vx, vy, vz = route.traj['velocity'][:, 0], route.traj['velocity'][:, 1], route.traj['velocity'][:, 2]
        # Get wind speed estimates in right format
        groundspeed = from_air_wind_to_groundspeed(u={'m': vx, 'std': np.zeros_like(vx)}, 
                                                   v={'m': vy, 'std': np.zeros_like(vy)}, 
                                                   w={'m': vz, 'std': np.zeros_like(vx)}, 
                                                   wx={'m': wind_uv[0][:, 0],                'std': windstd_uv[0][:, 0]}, 
                                                   wy={'m': wind_uv[0][:, 1],                'std': windstd_uv[0][:, 1]}, 
                                                   wz={'m': np.zeros_like(wind_uv[0][:, 0]), 'std': np.zeros_like(windstd_uv[0][:, 0])})
        new_ref_speed_mean.append(groundspeed['m'])
        new_ref_speed_std.append(groundspeed['std'])

    return new_ref_speed_mean, new_ref_speed_std


def compute_groundspeed_profile(vx, vy, wind_uv, windstd_uv, alpha_bound=1):
    gs_x, gs_y, \
        gs_x_low, gs_y_low, \
            gs_x_up, gs_y_up = compute_ground_speed_xy(np.column_stack((vx, vy)),
                                                        wind_uv, windstd_uv, alpha=alpha_bound)

    ground_vel_       = np.zeros((len(gs_x), 3))
    ground_vel_[:, 0] = np.sqrt(gs_x_low**2.0 + gs_y_low**2.0)
    ground_vel_[:, 1] = np.sqrt(gs_x**2.0 + gs_y**2.0)
    ground_vel_[:, 2] = np.sqrt(gs_x_up**2.0 + gs_y_up**2.0)
    # Get min, max, mean
    ground_vel_max = np.max(ground_vel_, axis=1)
    ground_vel_min = np.min(ground_vel_, axis=1)
    ground_vel_mean = np.mean(ground_vel_, axis=1)
    return ground_vel_mean, ground_vel_max, ground_vel_min


def compute_new_cruise_speed_with_wind(routes, wind_uv, windstd_uv, alpha_bound=1):
    routes            = make_list(routes)
    new_ref_speed_max = np.zeros((len(routes),))
    new_ref_speed_min = np.zeros((len(routes),))
    new_ref_speed_avg = np.zeros((len(routes),))
    for idx, route in enumerate(routes):
        vx, vy = route.traj['velocity'][:, 0], route.traj['velocity'][:, 1]
        gspeed, gspeed_max, gspeed_min = compute_groundspeed_profile(vx, vy, wind_uv[idx], windstd_uv[idx], alpha_bound=alpha_bound)
        ref_speed              = np.sqrt(vx**2.0 + vy**2.0)
        new_ref_speed_avg[idx] = compute_new_cruise_speed(ref_speed, gspeed)
        new_ref_speed_max[idx] = compute_new_cruise_speed(ref_speed, gspeed_max)
        new_ref_speed_min[idx] = compute_new_cruise_speed(ref_speed, gspeed_min)
    return new_ref_speed_avg, new_ref_speed_max, new_ref_speed_min



def from_speed_to_time(lat, lon, alt, horizontal_speed, ascent_speed, descent_speed, landing_speed=None, land_alt=10, geodetic_dist_method='greatcircle'):
    # land_alt is in meters

    assert hasattr(horizontal_speed, "__len__")==False or len(horizontal_speed) == len(lat), "Horizontal speed value horizontal_speed must be a scalar or an array with same length as lat."
    assert hasattr(ascent_speed,     "__len__")==False or len(ascent_speed)     == len(lat), "Vertical speed value ascent_speed must be a scalar or an array with same length as lat."
    assert hasattr(descent_speed,    "__len__")==False or len(descent_speed)    == len(lat), "Vertical speed value descent_speed must be a scalar or an array with same length as lat."
    
    # Replicate waypoints for each landing region (when altitude is < land_alt)
    lat, lon, alt = introduce_landing_waypoints(lat, lon, alt, land_alt=land_alt)

    n = len(lat)-1
    if landing_speed is None:                       landing_speed = descent_speed * 0.5
    if hasattr(landing_speed, "__len__") is False:  landing_speed = [landing_speed, ] * n
    if hasattr(ascent_speed, "__len__") is False:   ascent_speed  = [ascent_speed,  ] * n
    if hasattr(descent_speed, "__len__") is False:  descent_speed = [descent_speed, ] * n

    surface_dist = np.zeros((n,))
    vert_dist    = np.zeros_like(surface_dist)
    vert_speed   = np.zeros_like(surface_dist)
    for coord_i in range(1, n+1):
        surface_dist[coord_i-1], vert_dist[coord_i-1] = geom.geodetic_distance(lats=[lat[coord_i-1], lat[coord_i]], lons=[lon[coord_i-1], lon[coord_i]],
                                                                               alts=[alt[coord_i-1], alt[coord_i]], return_surf_vert=True, method=geodetic_dist_method)
        sign_vert_speed = np.sign(vert_dist[coord_i-1])
        if sign_vert_speed == 1:                    vert_speed[coord_i-1] =  ascent_speed[coord_i-1]
        elif sign_vert_speed == -1:
            if vert_dist[coord_i-1] <= -land_alt:   vert_speed[coord_i-1] = -1.0 * descent_speed[coord_i-1]
            else:                                   vert_speed[coord_i-1] = -1.0 * landing_speed[coord_i-1]

    eta = np.max(np.column_stack((np.nan_to_num(surface_dist / horizontal_speed), np.nan_to_num(vert_dist/vert_speed))), axis=1) # eta as max between horizontal and vertical ETAs
    return eta


def introduce_landing_waypoints(lat, lon, alt, land_alt=10):
    """ 
        lat, lon, alt = introduce_landing_waypoints(lat, lon, alt, land_alt=10)

    introduce fictitious waypoints to reduce speed in proximity of the ground 

    """
    # land altitude is in meters
    
    # Convert to arrays in case they're list
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    alt = np.asarray(alt)
    # alt[10] = 0.0   # artificial, to test the function
    
    idx_land = np.asarray(alt < land_alt) * np.insert(np.sign(np.diff(alt)) < 0, 0, False)
    idx_pos  = np.where(idx_land)[0]
    lat      = np.insert(lat, idx_pos, lat[idx_pos])
    lon      = np.insert(lon, idx_pos, lon[idx_pos])
    alt      = np.insert(alt, idx_pos, [land_alt*1.0, ]*len(idx_pos))
    return lat, lon, alt



def traj_from_flightplan(delta_t=0.01, gravity=9.81, nurbs_order=3, weight_vector=None, cruise_speed=6, ascent_speed=0.5, descent_speed=0.5, landing_speed=0.5, hovering=0., takeofftime=40, landingtime=40, alpha_bound=3):
    
    lat, lon, alt, _, tstamps = load.get_flightplan()
    route = Route(name='traj1', 
                  departure_time=tstamps[0], 
                  cruise_speed=cruise_speed, 
                  ascent_speed=ascent_speed, 
                  descent_speed=descent_speed, 
                  landing_speed=landing_speed)
    route.set_waypoints(lat, lon, alt) 
    route.set_eta(hovering=10, add_takeoff_time=takeofftime, add_landing_time=landingtime)
    
    T = Trajectory(name='traj1', route=route)
    T.generate(dt=delta_t, nurbs_order=nurbs_order, weight_vector=weight_vector, gravity=gravity)
    new_cruise_speed_avg, new_cruise_speed_std = horizontal_cruise_speed_from_cartesian_speed(T.velocity[:, 0],      
                                                                                              T.velocity[:, 1], 
                                                                                              T.velocity[:, 2],
                                                                                              np.abs(T.velocity[:, 0])*0.05, 
                                                                                              np.abs(T.velocity[:, 1])*0.05, 
                                                                                              np.abs(T.velocity[:, 2])*0.05)
    traj_timestamps = np.asarray([T.route.scheduled_departure + dt.timedelta(seconds=T.time[item]) for item in range(len(T.time))])
    new_wp_speed_avg, new_wp_speed_std = average_speed_between_wps(np.asarray(T.route.eta), traj_timestamps, new_cruise_speed_avg, new_cruise_speed_std)
    
    eta_avg = route.compute_etas_from_speed(hovering=0., takeoff_time=takeofftime, landing_time=landingtime, 
                                            cruise_speed=np.max(new_wp_speed_avg))
    eta_up  = route.compute_etas_from_speed(hovering=0., takeoff_time=takeofftime, landing_time=landingtime, 
                                            cruise_speed=np.max(new_wp_speed_avg) + alpha_bound * np.max(new_wp_speed_std))
    eta_low = route.compute_etas_from_speed(hovering=0., takeoff_time=takeofftime, landing_time=landingtime, 
                                            cruise_speed=np.max(new_wp_speed_avg) - alpha_bound * np.max(new_wp_speed_std))
    timestamps_avg = np.asarray([T.route.scheduled_departure + dt.timedelta(seconds=eta_avg[item]) for item in range(len(eta_avg))])
    timestamps_up  = np.asarray([T.route.scheduled_departure + dt.timedelta(seconds=eta_up[item]) for item in range(len(eta_up))])
    timestamps_low = np.asarray([T.route.scheduled_departure + dt.timedelta(seconds=eta_low[item]) for item in range(len(eta_low))])
    route.eta           = {'upper': eta_up,        'lower': eta_low,        'avg': eta_avg}
    route.eta_timestamp = {'upper': timestamps_up, 'lower': timestamps_low, 'avg': timestamps_avg}
    return route


def get_route(dt=1.0, gravity=9.81, desired_cruisespeed=100, takeofftime=60, landingtime=60, nurbs_order=5, weight_vector=None, alpha_bound=3):
    route_filename = 'data/SF_SJ_OAK_routes.txt'
    routes = read_routes(route_filename, ft2m=True)
    route  = routes[0]

    route.compute_etas_from_speed(cruise_speed=desired_cruisespeed, takeoff_time=takeofftime, landing_time=landingtime)
    traj = route.compute_trajectory(gravity=gravity, delta_t=dt, n_order=nurbs_order, weight_vector=weight_vector)
    new_cruise_speed_avg, new_cruise_speed_std = horizontal_cruise_speed_from_cartesian_speed(traj['velocity'][:, 0],      
                                                                                              traj['velocity'][:, 1], 
                                                                                              traj['velocity'][:, 2],
                                                                                              np.abs(traj['velocity'][:, 0])*0.05, 
                                                                                              np.abs(traj['velocity'][:, 1])*0.05, 
                                                                                              np.abs(traj['velocity'][:, 2])*0.05)

    new_wp_speed_avg, new_wp_speed_std = average_speed_between_wps(route.eta_timestamp, traj['timestamps'], new_cruise_speed_avg, new_cruise_speed_std)
    
    eta_avg, timestamps_avg = route.compute_etas_from_speed(cruise_speed=np.max(new_wp_speed_avg), takeoff_time=takeofftime, landing_time=landingtime)
    eta_up,   timestamps_up = route.compute_etas_from_speed(cruise_speed=np.max(new_wp_speed_avg) + alpha_bound * np.max(new_wp_speed_std), takeoff_time=takeofftime, landing_time=landingtime)
    eta_low, timestamps_low = route.compute_etas_from_speed(cruise_speed=np.max(new_wp_speed_avg) - alpha_bound * np.max(new_wp_speed_std), takeoff_time=takeofftime, landing_time=landingtime)
    route.eta           = {'upper': eta_up,        'lower': eta_low,        'avg': eta_avg}
    route.eta_timestamp = {'upper': timestamps_up, 'lower': timestamps_low, 'avg': timestamps_avg}
    return route


if __name__ == '__main__':
    

    print("trajectory generation code")
    
    # =======
    # END