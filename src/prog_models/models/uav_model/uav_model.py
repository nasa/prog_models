# Copyright © 2022 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from prog_models.prognostics_model import PrognosticsModel
import prog_models.models.uav_model.trajectory.route as route 
import prog_models.models.uav_model.trajectory.trajectory as trajectory
from prog_models.models.uav_model.vehicles import AircraftModels
from prog_models.models.uav_model.utilities import geometry

import numpy as np
import datetime
from copy import deepcopy
import prog_models.models.uav_model.utilities.geometry as geom
from warnings import warn
from prog_models.exceptions import ProgModelInputException

class UAVGen(PrognosticsModel):
    """
    Vectorized prognostics :term:`model` to generate a predicted trajectory for a UAV. 
    
    Model generates cartesian positions and velocities, pitch, roll, and yaw, and angular velocities throughout time to satisfy some user-define waypoints. 

    :term:`Events<event>`: (1)
        TrajectoryComplete: percentage of waypoints remaining for trajectory to pass through
    
    :term:`Inputs/Loading<input>`: ()
        | User-defined inputs: waypoints plus ETAs or speeds
        | Model-defined inputs: 
            | T: thrust
            | mx: moment in x 
            | my: moment in y
            | mz: moment in z

    :term:`States<state>`: (13)
        | x: cartesian position
        | y: cartesian position
        | z: cartesian position
        | phi: pitch
        | theta: roll
        | psi: yaw
        | vx: velocity 
        | vy: velocity 
        | vz: velocity 
        | p: angular velocity 
        | q: angular velocity 
        | r: angular velocity 
        | t: time 

    :term:`Outputs<output>`: (12)
        | x: cartesian position
        | y: cartesian position
        | z: cartesian position
        | phi: pitch
        | theta: roll
        | psi: yaw
        | vx: velocity 
        | vy: velocity 
        | vz: velocity 
        | p: angular velocity 
        | q: angular velocity 
        | r: angular velocity 
        | t: time 

    Keyword Args
    ------------
        process_noise : Optional, float or dict[str, float]
          :term:`Process noise<process noise>` (applied at dx/next_state). 
          Can be number (e.g., .2) applied to every state, a dictionary of values for each 
          state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        process_noise_dist : Optional, str
          distribution for :term:`process noise` (e.g., normal, uniform, triangular)
        measurement_noise : Optional, float or dict[str, float]
          :term:`Measurement noise<measurement noise>` (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of values for each
          output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        measurement_noise_dist : Optional, str
          distribution for :term:`measurement noise` (e.g., normal, uniform, triangular)
        flight_file : Optional, str
          Text file to specify waypoint information. Necessary columns must be in the following
          order with units specified: latitude (labeled 'lat_deg' or 'lat_rad'), longitude 
          (labeled 'lon_deg' or 'lon_rad'), and altitude (labeled 'alt_ft' or 'alt_m'). An 
          additional column for ETAs may be included (labeled 'time_unix). Note that while
          'flight_file' is optional, either 'flight_file' or 'flight_plan' must be specified.
        flight_plan : Optional, dict[str, numpy array]
          Dictionary to specify waypoint information. Necessary keys must include the following
          with units specified: latitude ('lat_deg' or 'lat_rad'), longitude 
          (labeled 'lon_deg' or 'lon_rad'), and altitude (labeled 'alt_ft' or 'alt_m'). An 
          additional key for ETAs may be included (labeled 'time_unix). Each key must correspond
          to a numpy array of values. Note that while 'flight_plan' is optional, either 
          'flight_file' or 'flight_plan' must be specified.
        flight_name : Optional, str
          Optional string to identify flight plan.
        aircraft_name : Optional, str
          Optional string to identify aircraft. 
        dt : float
          Time step in seconds for trajectory generation
        gravity : float
          m/s^2, gravity magnitude
        cruise_speed : float
          m/s, avg speed in-between way-points
        ascent_speed : float
          m/s, vertical speed (up)
        descent_speed : float
          m/s, vertical speed (down)
        landing_speed : float
          m/s, landing speed when altitude < 10m
        hovering_time : float
          s, time to hover between waypoints
        takeoff_time : float
          s, additional takeoff time 
        landing_time: float
          s, additional landing time 
        waypoint_weights: 
          weights of the waypoints in nurbs calculation 
        adjust_eta: dict 
          Dictionary with keys ['hours', 'seconds'], to adjust route time
        nurbs_basis_length: float
          Length of the basis function in the nurbs algorithm
        nurbs_order: int
          Order of the nurbs curve
        final_time_buffer_sec: float
          s, defines an acceptable time range to reach the final waypoint
        final_space_buffer_m: float
          m, defines an acceptable distance range to reach final waypoint 
        vehicle_model: str
          String to specify vehicle type. 'tarot18' and 'djis1000' are supported
        vehicle_payload: float
          kg, payload mass

    References 
    ----------

    """
    events = ['TrajectoryComplete']
    inputs = ['T','mx','my','mz']
    states = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r','t']
    outputs = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r']
    is_vectorized = True

    default_parameters = {  # Set to defaults
        # Flight information
        'flight_file': None, 
        'flight_name': 'flight-1', 
        'aircraft_name': 'aircraft-1', 
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
        'waypoint_weights': 20.0, 
        'adjust_eta': None, 
        'nurbs_basis_length': 2000, 
        'nurbs_order': 4, 
        'final_time_buffer_sec': 30, 
        'final_space_buffer_m': 2, 

        # Vehicle parameters:
        'vehicle_model': 'tarot18', 
        'vehicle_payload': 0.0,
    }

    def initialize(self, u=None, z=None): 
        
        if self.parameters['flight_plan'] and self.parameters['flight_file'] == None:
            flightplan = trajectory.load.convert_dict_inputs(self.parameters['flight_plan'])
            lat = flightplan['lat_rad']
            lon = flightplan['lon_rad']
            alt = flightplan['alt_m']
            tstamps = flightplan['timestamp']
        elif self.parameters['flight_file'] and self.parameters['flight_plan'] == None:
            flightplan = trajectory.load.get_flightplan(fname=self.parameters['flight_file'])
            lat, lon, alt, tstamps = flightplan['lat'], flightplan['lon'], flightplan['alt'], flightplan['timestamp']
        elif self.parameters['flight_file'] and self.parameters['flight_plan']:
            raise ProgModelInputException("Too many flight plan inputs - please input either flight_plan dictionary or flight_file.")
        else:
            raise ProgModelInputException("No flight plan information supplied.")

        aircraft1 = AircraftModels.build_model(name=self.parameters['aircraft_name'],
                                               model=self.parameters['vehicle_model'],
                                               payload=self.parameters['vehicle_payload'])
        self.vehicle_model = aircraft1 

        # Generate route
        if len(tstamps) > 1:
            # ETAs specified: 
            # Check if speeds have been defined and warn user if so:
            if self.parameters['cruise_speed'] is not None or self.parameters['ascent_speed'] is not None or self.parameters['descent_speed'] is not None:
                warn("Speed values are ignored since ETAs were specified. To define speeds (cruise, ascent, descent) instead, do not specify ETAs.")
            route_ = route.build(name=self.parameters['flight_name'], lat=lat, lon=lon, alt=alt, departure_time=tstamps[0],
                                 etas=tstamps,  # ETAs override any cruise/ascent/descent speed requirements. Do not feed etas if want to use desired speeds values.
                                 vehicle_max_speed = self.vehicle_model.dynamics['max_speed'],
                                 parameters = self.parameters)
        else: 
            # ETAs not specified:  
            # Check that speeds have been provided:
            if self.parameters['cruise_speed'] is None or self.parameters['ascent_speed'] is None or self.parameters['descent_speed'] is None:
                raise ProgModelInputException("ETA or speeds must be provided. If ETAs are not defined, desired speed (cruise, ascent, descent) must be provided.")  
            route_ = route.build(name=self.parameters['flight_name'], lat=lat, lon=lon, alt=alt, departure_time=tstamps[0],
                                 parameters = self.parameters)

        # Save final waypoint information for threshold_met and event_state 
        self.parameters['final_time'] = datetime.datetime.timestamp(route_.eta[-1]) - datetime.datetime.timestamp(route_.eta[0])
        coord_end = geometry.Coord(route_.lat[0], route_.lon[0], route_.alt[0])
        self.coord_transform = deepcopy(coord_end)
        self.parameters['final_x'], self.parameters['final_y'], self.parameters['final_z'] = coord_end.geodetic2enu(route_.lat[-1], route_.lon[-1], route_.alt[-1])
        wypt_time_unix = [datetime.datetime.timestamp(route_.eta[iter]) - datetime.datetime.timestamp(route_.eta[0]) for iter in range(len(route_.eta))]
        wypt_x = []
        wypt_y = []
        wypt_z = []
        for iter1 in range(len(route_.lat)):
            x_temp, y_temp, z_temp = coord_end.geodetic2enu(route_.lat[iter1], route_.lon[iter1], route_.alt[iter1])
            wypt_x.append(x_temp)
            wypt_y.append(y_temp)
            wypt_z.append(z_temp)
        self.parameters['waypoints'] = {'waypoints_time': wypt_time_unix, 'waypoints_x': wypt_x, 'waypoints_y': wypt_y, 'waypoints_z': wypt_z, 'next_waypoint': 0}
        
        # Generate trajectory
        ref_traj = trajectory.Trajectory(name=self.parameters['flight_name'], route=route_)
        ref_traj.generate(dt=self.parameters['dt'], 
                        nurbs_order=self.parameters['nurbs_order'], 
                        gravity=self.parameters['gravity'], 
                        weight_vector=np.array([self.parameters['waypoint_weights'],]*len(route_.lat)),   # weight of waypoints
                        nurbs_basis_length=self.parameters['nurbs_basis_length'],
                        max_phi=aircraft1.dynamics['max_roll'],                    # rad, allowable roll for the aircraft
                        max_theta=aircraft1.dynamics['max_pitch'])                 # rad, allowable pitch for the aircraft

        self.ref_traj = ref_traj
        self.current_time = 0

        # Initialize vehicle 
        aircraft1.set_state(state=np.concatenate((ref_traj.cartesian_pos[0, :], ref_traj.attitude[0, :], ref_traj.velocity[0, :], ref_traj.angular_velocity[0, :]), axis=0))
        aircraft1.set_dt(dt=self.parameters['dt'])

        # self.parameters['x0'] = self.StateContainer({'x': ref_traj.cartesian_pos[0,0], 'y': ref_traj.cartesian_pos[0,1], 'z': ref_traj.cartesian_pos[0,2],
        #                                         'phi': ref_traj.attitude[0,0], 'theta': ref_traj.attitude[0,1], 'psi': ref_traj.attitude[0,2],
        #                                         'vx': ref_traj.velocity[0,0], 'vy': ref_traj.velocity[0,1], 'vz': ref_traj.velocity[0,2],
        #                                         'p': ref_traj.angular_velocity[0,0], 'q': ref_traj.angular_velocity[0,1], 'r': ref_traj.angular_velocity[0,2],
        #                                         't': 0})

        return self.StateContainer({
            'x': ref_traj.cartesian_pos[0,0],
            'y': ref_traj.cartesian_pos[0,1],
            'z': ref_traj.cartesian_pos[0,2],
            'phi': ref_traj.attitude[0,0],
            'theta': ref_traj.attitude[0,1],
            'psi': ref_traj.attitude[0,2],
            'vx': ref_traj.velocity[0,0],
            'vy': ref_traj.velocity[0,1],
            'vz': ref_traj.velocity[0,2],
            'p': ref_traj.angular_velocity[0,0],
            'q': ref_traj.angular_velocity[0,1],
            'r': ref_traj.angular_velocity[0,2],
            't': 0
            })
    
    def dx(self, x : dict, u : dict):

        # Extract values from vectors
        # --------------------------------
        m = self.vehicle_model.mass['total']  # vehicle mass
        T = u['T'] 
        tp = u['mx']
        tq = u['my']
        tr = u['mz']
        Ixx, Iyy, Izz = self.vehicle_model.mass['Ixx'], self.vehicle_model.mass['Iyy'], self.vehicle_model.mass['Izz']    # vehicle inertia

        # Extract state variables from current state vector
        # -------------------------------------------------
        phi = x['phi'] 
        theta = x['theta'] 
        psi = x['psi']
        vx_a = x['vx']
        vy_a = x['vy']
        vz_a = x['vz']
        p = x['p']
        q = x['q']
        r = x['r']

        # Pre-compute Trigonometric values
        # --------------------------------
        sin_phi   = np.sin(phi)
        cos_phi   = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        tan_theta = np.tan(theta)
        sin_psi   = np.sin(psi)
        cos_psi   = np.cos(psi)
        
        # Compute drag forces
        # -------------------
        v_earth = np.array([vx_a, vy_a, vz_a]).reshape((-1,))
        v_body = np.dot(geom.rot_eart2body_fast(sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi), v_earth)
        fb_drag = self.vehicle_model.aero['drag'](v_body) 
        fe_drag = np.dot(geom.rot_body2earth_fast(sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi), fb_drag)
        fe_drag[-1] = np.sign(v_earth[-1]) * np.abs(fe_drag[-1])

        # Update state vector
        # -------------------
        dxdt     = np.zeros((len(x),))
        
        dxdt[0] = vx_a
        dxdt[1] = vy_a
        dxdt[2] = vz_a
        
        dxdt[3]  = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta
        dxdt[4]  = q * cos_phi - r * sin_phi
        dxdt[5]  = q * sin_phi / cos_theta + r * cos_phi / cos_theta
        
        dxdt[6]  = (sin_theta * cos_psi * cos_phi + sin_phi * sin_psi) * T / m - 1.0/m * fe_drag[0]
        dxdt[7]  = (sin_theta * sin_psi * cos_phi - sin_phi * cos_psi) * T / m - 1.0/m * fe_drag[1]
        dxdt[8]  = - self.parameters['gravity'] + cos_phi * cos_theta  * T / m - 1.0/m * fe_drag[2]

        dxdt[9]  = (Iyy - Izz) / Ixx * q * r + tp * self.vehicle_model.geom['arm_length'] / Ixx
        dxdt[10] = (Izz - Ixx) / Iyy * p * r + tq * self.vehicle_model.geom['arm_length'] / Iyy
        dxdt[11] = (Ixx - Iyy) / Izz * p * q + tr *        1                / Izz
        dxdt[12] = 1 

        # Set vehicle state:
        state_temp = np.array([x[iter] for iter in x.keys()])
        self.vehicle_model.set_state(state=state_temp + dxdt*self.parameters['dt'])

        return self.StateContainer(np.array([
            np.atleast_1d(dxdt[0]),
            np.atleast_1d(dxdt[1]),
            np.atleast_1d(dxdt[2]),
            np.atleast_1d(dxdt[3]),
            np.atleast_1d(dxdt[4]),
            np.atleast_1d(dxdt[5]),
            np.atleast_1d(dxdt[6]),
            np.atleast_1d(dxdt[7]),
            np.atleast_1d(dxdt[8]),
            np.atleast_1d(dxdt[9]),
            np.atleast_1d(dxdt[10]),
            np.atleast_1d(dxdt[11]),
            np.atleast_1d(dxdt[12])
        ]))
    
    def event_state(self, x : dict) -> dict:
        # Extract next waypoint information 
        num_wypts = len(self.parameters['waypoints']['waypoints_time']) - 1 # Don't include initial waypoint
        index_next = self.parameters['waypoints']['next_waypoint']

        # Check if at intial waypoint. If so, event_state is 1
        if index_next == 0:
            self.parameters['waypoints']['next_waypoint'] += 1
            return {
                'TrajectoryComplete': 1
            }
        # Check if passed final waypoint. If so, event_state is 0
        if index_next > num_wypts:
            return {
                'TrajectoryComplete': 0
            }
        
        t_next = self.parameters['waypoints']['waypoints_time'][index_next]
        x_next = self.parameters['waypoints']['waypoints_x'][index_next]
        y_next = self.parameters['waypoints']['waypoints_y'][index_next]
        z_next = self.parameters['waypoints']['waypoints_z'][index_next]

        # Define time interval for acceptable arrival at waypoint
        time_buffer_left = (self.parameters['waypoints']['waypoints_time'][index_next] - self.parameters['waypoints']['waypoints_time'][index_next - 1])/2
        if index_next == num_wypts:
            # Final waypoint, add final buffer time 
            time_buffer_right = t_next + self.parameters['final_time_buffer_sec']
        else: 
            time_buffer_right = (self.parameters['waypoints']['waypoints_time'][index_next+1] - self.parameters['waypoints']['waypoints_time'][index_next])/2

        # Check if next waypoint is satisfied:
        if x['t'] < t_next - time_buffer_left:
            # Not yet within time of next waypoint
            return {
                    'TrajectoryComplete': (num_wypts - (index_next - 1))/num_wypts
                }
        elif t_next - time_buffer_left <= x['t'] <= t_next + time_buffer_right:
            # Current time within ETA interval. Check if distance also within acceptable range
            dist_now = np.sqrt((x['x']-x_next)**2 + (x['y']-y_next)**2 + (x['z']-z_next)**2)
            if dist_now <= self.parameters['final_space_buffer_m']:
                # Waypoint achieved
                self.parameters['waypoints']['next_waypoint'] += 1
                return {
                    'TrajectoryComplete': (num_wypts - index_next)/num_wypts
                }
            else:
                # Waypoint not yet achieved
                return {
                    'TrajectoryComplete': (num_wypts - (index_next - 1))/num_wypts
                }
        else:
            # ETA passed before waypoint reached 
            warn("Trajectory did not reach waypoint associated with ETA of {})".format(t_next))
            self.parameters['waypoints']['next_waypoint'] += 1
            return {
                    'TrajectoryComplete': (num_wypts - index_next)/num_wypts
                }
 
    def output(self, x : dict):
        # Output is the same as the state vector
        return self.OutputContainer({
            'x': x['x'],
            'y': x['y'],
            'z': x['z'],
            'phi': x['phi'],
            'theta': x['theta'],
            'psi': x['psi'],
            'vx': x['vx'],
            'vy': x['vy'],
            'vz': x['vz'],
            'p': x['p'],
            'q': x['q'],
            'r': x['r'],
            't': x['t']
            })

    def threshold_met(self, x : dict) -> dict:
        if x['t'] < self.parameters['final_time'] - self.parameters['final_time_buffer_sec']: 
            # Trajectory hasn't reached final ETA
            return {
                'TrajectoryComplete': False
            }
        elif self.parameters['final_time'] - self.parameters['final_time_buffer_sec'] <= x['t'] <= self.parameters['final_time'] + self.parameters['final_time_buffer_sec']:
            # Trajectory is within bounds of final ETA
            dist_now = np.sqrt((x['x']-self.parameters['final_x'])**2 + (x['y']-self.parameters['final_y'])**2 + (x['z']-self.parameters['final_z'])**2)
            if dist_now <= self.parameters['final_space_buffer_m']:
                return {
                    'TrajectoryComplete': True
                }
            else: 
                return {
                    'TrajectoryComplete': False
                }
        else: 
            # Trajectory has passed acceptable bounds of final ETA - simulation terminated
            warn("Trajectory simulation extends beyond the final ETA. Either the final waypoint was not reached in the alotted time (and the simulation was terminated), or simulation was run for longer than the trajectory length.")
            return {
                'TrajectoryComplete': True
            }

    def simulate_to_threshold(self, future_loading_eqn, first_output = None, threshold_keys = None, **kwargs):
        
        if 'dt' in kwargs and kwargs['dt'] != self.parameters['dt']:
            if isinstance(kwargs['dt'], int): 
                warn("dt value for simulation must be equal to the constant dt value defined in model generation. To change the simulate_to time step, change the model parameter 'dt' value")
            else: 
                warn("dt value for simulation must be equal to the constant dt value defined in model generation. Other time step functionalities (e.g. 'auto' or function definition) are not supported with trajectory generation.")
        kwargs['dt'] = self.parameters['dt']
        if 'save_pts' in kwargs:
            warn("'save_pts' functionality is not available with trajectory generation. A trajectory can only be generated for the same dt value as specified in the model generation.")
            kwargs['save_pts'] = []
        if 'save_freq' not in kwargs:
            kwargs['save_freq'] = self.parameters['dt'] # if save_freq not specified, set at dt for best output 

        # Define initial x to avoid re-initializing 
        # kwargs['x'] = deepcopy(self.parameters['x0'])

        def future_loading_new(t, x=None): 
            if t == 0:
                ref_now = np.concatenate((self.ref_traj.cartesian_pos[0,:], self.ref_traj.attitude[0,:], 
                                    self.ref_traj.velocity[0,:], self.ref_traj.angular_velocity[0,:]), axis=0)
                
                u = self.vehicle_model.control_scheduled(self.vehicle_model.state - ref_now)
                u[0]      += self.vehicle_model.steadystate_input
                u[0]       = min(max([0., u[0]]), self.vehicle_model.dynamics['max_thrust'])
     
                return self.InputContainer({'T': u[0], 'mx': u[1], 'my': u[2], 'mz': u[3]})
            else:
                t_temp = np.round(t - self.parameters['dt']/2,1) # THIS NEEDS HELP
                time_ind = np.argmin(np.abs(t_temp - self.ref_traj.time.tolist()))
                ref_now = np.concatenate((self.ref_traj.cartesian_pos[time_ind,:], self.ref_traj.attitude[time_ind,:], 
                                        self.ref_traj.velocity[time_ind,:], self.ref_traj.angular_velocity[time_ind,:]), axis=0)

                # Define controller
                x_temp = np.array([x.matrix[ii][0] for ii in range(len(x.matrix)-1)])
                u = self.vehicle_model.control_scheduled(x_temp - ref_now) 
                u[0]      += self.vehicle_model.steadystate_input
                u[0]       = min(max([0., u[0]]), self.vehicle_model.dynamics['max_thrust'])
                return self.InputContainer({'T': u[0], 'mx': u[1], 'my': u[2], 'mz': u[3]})

        # Simulate to threshold 
        results = super().simulate_to_threshold(future_loading_new,first_output, threshold_keys, **kwargs)
        return results 

    def visualize_traj(self, pred):

        import matplotlib.pyplot as plt

        # Conversions
        # -----------
        deg2rad = np.pi/180.0
        rad2deg = 180.0/np.pi
        
        # Extract reference trajectory information
        # ----------------------------------------
        waypoints   = self.ref_traj.route
        eta         = self.ref_traj.route.eta
        depart_time = self.ref_traj.route.departure_time
        time        = self.ref_traj.time
        ref_pos     = self.ref_traj.geodetic_pos

        # Extract predicted trajectory information
        # ----------------------------------------
        pred_time = pred.times
        pred_x = [pred.outputs[iter]['x'] for iter in range(len(pred_time))]
        pred_y = [pred.outputs[iter]['y'] for iter in range(len(pred_time))]
        pred_z = [pred.outputs[iter]['z'] for iter in range(len(pred_time))]

        pred_lat = []
        pred_lon = []
        pred_alt = []

        for iter1 in range(len(pred_time)):
            x_temp, y_temp, z_temp = self.coord_transform.enu2geodetic(pred_x[iter1], pred_y[iter1], pred_z[iter1])
            pred_lat.append(x_temp)
            pred_lon.append(y_temp)
            pred_alt.append(z_temp)
        
        pred_lat_deg = [pred_lat[iter] * rad2deg for iter in range(len(pred_lat))]
        pred_lon_deg = [pred_lon[iter] * rad2deg for iter in range(len(pred_lon))]

        # Initialize Figure
        # ----------------
        params = dict(figsize=(13, 9), fontsize=14, linewidth=2.0, alpha_preds=0.6)
        fig, (ax1, ax2) = plt.subplots(2)

        # Plot trajectory predictions
        # -------------------------
        # First plot waypoints (dots) and reference trajectory (commanded, line)
        ax1.plot(waypoints.lon * rad2deg, waypoints.lat  * rad2deg, 'o', color='tab:orange', alpha=0.5, markersize=10, label='__nolegend__')
        ax1.plot(ref_pos[:, 1] * rad2deg, ref_pos[:, 0] * rad2deg, '--', linewidth=params['linewidth'], color='tab:orange', alpha=0.5, label='traj')
        ax1.plot(pred_lon_deg, pred_lat_deg,'-', color='tab:blue', alpha=params['alpha_preds'], linewidth=params['linewidth'], label='prediction')

        ax1.set_xlabel('longitude, deg', fontsize=params['fontsize'])
        ax1.set_ylabel('latitude, deg', fontsize=params['fontsize'])
        ax1.legend(fontsize=params['fontsize'])

        # Add altitude plot
        # -------------------------------------------------------
        time_vector = [depart_time + datetime.timedelta(seconds=pred_time[ii]) for ii in range(len(pred_time))]
        ax2.plot_date(eta, waypoints.alt, '--o', color='tab:orange', alpha=0.5, linewidth=params['linewidth'], label='__nolegend__')
        ax2.plot(time_vector, pred_alt,'-', color='tab:blue',
                      alpha=params['alpha_preds'], linewidth=params['linewidth'], label='__nolegend__')
        
        ax2.set_xlabel('time stamp, -', fontsize=params['fontsize'])
        ax2.set_ylabel('altitude, m', fontsize=params['fontsize'])

        return fig
