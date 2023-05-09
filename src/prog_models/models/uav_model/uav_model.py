# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
import datetime
from warnings import warn
from typing import Callable

from prog_models.prognostics_model import PrognosticsModel
from .vehicles import AircraftModels
from prog_models.aux_fcns.traj_gen_utils import geometry as geom
from prog_models.exceptions import ProgModelInputException

class UAVGen(PrognosticsModel):
    """
    Vectorized prognostics :term:`model` to generate a predicted trajectory for a UAV using a n=6 degrees-of-freedom dynamic model
    with feedback control loop. The model follows the form:
    
    u     = h(x, x_{ref})
    dx/dt = f(x, \theta, u)
    
    where:
      x is a 2n state vector containing position, attitude and corresponding derivatives
      \theta is a vector of model parameters including UAV mass, inertia moment, aerodynamic coefficients, etc
      u is the input vector: thrust along the body vertical axis, and three moments along the UAV body axis to follow the desired trajectory.
      x_{ref} is the desired state vector at that specific time step, with dimension 2n
      f(.) is growth rate function of all vechile state
      h(.) is the feedback-loop control function that returns the necessary thrust and moments (u vector) to cover the error between desired state x_{ref} and current state x
      dx/dt is the state-increment per unit time.

    
    Model generates cartesian positions and velocities, pitch, roll, and yaw, and angular velocities throughout time to satisfy some user-define waypoints. 

    See [0]_ for modeling details. 

    :term:`Events<event>`: (1)
        TrajectoryComplete: All waypoints are completed 
    
    :term:`Inputs/Loading<input>`: (0)
        | User-defined inputs: waypoints plus ETAs or speeds
        | Model-defined inputs: 
            | T: thrust
            | mx: moment in x 
            | my: moment in y
            | mz: moment in z

    :term:`States<state>`: (13)
        | x: first position in cartesian reference frame East-North-Up (ENU), i.e., East in fixed inertia frame, center is at first waypoint
        | y: second position in cartesian reference frame East-North-Up (ENU), i.e., North in fixed inertia frame, center is at first waypoint
        | z: third position in cartesian reference frame East-North-Up (ENU), i.e., Up in fixed inertia frame, center is at first waypoint
        | phi: Euler's first attitude angle
        | theta: Euler's second attitude angle
        | psi: Euler's third attitude angle
        | vx: velocity along x-axis, i.e., velocity along East in fixed inertia frame
        | vy: velocity along y-axis, i.e., velocity along North in fixed inertia frame
        | vz: velocity along z-axis, i.e., velocity Up in fixed inertia frame
        | p: angular velocity around UAV body x-axis
        | q: angular velocity around UAV body y-axis 
        | r: angular velocity around UAV body z-axis 
        | t: time 

    :term:`Outputs<output>`: (12)
        | x: first position in cartesian reference frame East-North-Up (ENU), i.e., East in fixed inertia frame, center is at first waypoint
        | y: second position in cartesian reference frame East-North-Up (ENU), i.e., North in fixed inertia frame, center is at first waypoint 
        | z: third position in cartesian reference frame East-North-Up (ENU), i.e., Up in fixed inertia frame, center is at first waypoint
        | phi: Euler's first attitude angle
        | theta: Euler's second attitude angle
        | psi: Euler's third attitude angle
        | vx: velocity along x-axis, i.e., velocity along East in fixed inertia frame
        | vy: velocity along y-axis, i.e., velocity along North in fixed inertia frame
        | vz: velocity along z-axis, i.e., velocity Up in fixed inertia frame
        | p: angular velocity around UAV body x-axis
        | q: angular velocity around UAV body y-axis 
        | r: angular velocity around UAV body z-axis 

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
        dt : Optional, float
          Time step in seconds for trajectory generation
        gravity : Optional, float
          m/s^2, gravity magnitude
        final_time_buffer_sec: Optional, float
          s, defines an acceptable time range to reach the final waypoint
        final_space_buffer_m: Optional, float
          m, defines an acceptable distance range to reach final waypoint 
        vehicle_model: Optional, str
          String to specify vehicle type. 'tarot18' and 'djis1000' are supported
        vehicle_payload: Optional, float
          kg, payload mass

    References 
    ----------
        References 
    -------------
     .. [0] M. Corbetta et al., "Real-time UAV trajectory prediction for safely monitoring in low-altitude airspace," AIAA Aviation 2019 Forum,  2019. https://arc.aiaa.org/doi/pdf/10.2514/6.2019-3514
    """
    events = ['TrajectoryComplete']
    inputs = ['T','mx','my','mz']
    states = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r','t']
    outputs = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r']
    is_vectorized = True

    default_parameters = {  # Set to defaults

        # Simulation parameters:
        'dt': 0.1, 
        'gravity': 9.81,
        'final_time_buffer_sec': 30, 
        'final_space_buffer_m': 2, 

        # Vehicle parameters:
        'vehicle_model': 'tarot18', 
        'vehicle_payload': 0.0,
        'vehicle_max_speed': 15.0, #### These are needed for traj_gen, are these okay defaults?
        'vehicle_max_roll': 0.7853981633974483, ##### fix this 
        'vehicle_max_pitch': 0.7853981633974483, #### fix this 
        'ref_traj': None
    }

    def __init__(self, **kwargs):

      super().__init__(**kwargs)

      # Build aircraft model
      # ====================
      # Check for supported vehicle type
      if self.parameters['vehicle_model'] != 'tarot18' and self.parameters['vehicle_model'] != 'djis1000':
        raise ProgModelInputException("Specified vehicle type is not supported. Only 'tarot18' and 'djis1000' are currently supported.")
      
      # build aicraft, which means create rotorcraft from type (model), initialize state vector, steady-state input (i.e., hover thrust for rotorcraft), controller type 
      # and corresponding setup (scheduled, real-time) and initialization.
      aircraft1 = AircraftModels.build_model(model=self.parameters['vehicle_model'],
                                              payload=self.parameters['vehicle_payload'])
      self.vehicle_model = aircraft1 

      self.current_time = 0

      # Initialize vehicle: set initial state and dt for integration.
      # ---------------------------------------------------------------
      # aircraft1.set_state(state=np.concatenate((ref_traj.cartesian_pos[0, :], ref_traj.attitude[0, :], ref_traj.velocity[0, :], ref_traj.angular_velocity[0, :]), axis=0))  # set initial state
      aircraft1.set_state(state=np.array([0,0,0,0,0,0,0,0,0,0,0]))  # TODO: needs to be more general (see line above), but no longer have ref_traj here 
      aircraft1.set_dt(dt=self.parameters['dt'])  # set dt for simulation

    def initialize(self, u=None, z=None): 
      # Extract initial state from reference trajectory    
      # TODO: needs to be more general, but don't have ref_traj here 
      return self.StateContainer({
          'x': 0, # self.ref_traj.cartesian_pos[0, 0],
          'y': 0, #self.ref_traj.cartesian_pos[0, 1],
          'z': 0, #self.ref_traj.cartesian_pos[0, 2],
          'phi': 0, #self.ref_traj.attitude[0, 0],
          'theta': 0, #self.ref_traj.attitude[0, 1],
          'psi': 0, # self.ref_traj.attitude[0, 2],
          'vx': 0, #self.ref_traj.velocity[0, 0],
          'vy': 0, #self.ref_traj.velocity[0, 1],
          'vz': 0, #self.ref_traj.velocity[0, 2],
          'p': 0, #self.ref_traj.angular_velocity[0, 0],
          'q': 0, #self.ref_traj.angular_velocity[0, 1],
          'r': 0, #self.ref_traj.angular_velocity[0, 2],
          't': 0
          })
    
    def dx(self, x : dict, u : dict):

        # Extract useful values
        # ---------------------
        m = self.vehicle_model.mass['total']  # vehicle mass
        Ixx, Iyy, Izz = self.vehicle_model.mass['Ixx'], self.vehicle_model.mass['Iyy'], self.vehicle_model.mass['Izz']    # vehicle inertia
        
        # Input vector
        T  = u['T']   # Thrust (along body z)
        tp = u['mx']  # Moment along body x
        tq = u['my']  # Moment along body y
        tr = u['mz']  # Moment along body z

        # Extract state variables from current state vector
        # -------------------------------------------------
        phi   = x['phi'] 
        theta = x['theta'] 
        psi   = x['psi']
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
        v_earth = np.array([vx_a, vy_a, vz_a]).reshape((-1,)) # velocity in Earth-fixed frame
        v_body  = np.dot(geom.rot_eart2body_fast(sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi), v_earth)  # Velocity in body-axis
        fb_drag = self.vehicle_model.aero['drag'](v_body)   # drag force in body axis
        fe_drag = np.dot(geom.rot_body2earth_fast(sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi), fb_drag) # drag forces in Earth-fixed frame
        fe_drag[-1] = np.sign(v_earth[-1]) * np.abs(fe_drag[-1])  # adjust vertical (z=Up) force according to velocity in fixed frame

        # Update state vector
        # -------------------
        dxdt = np.zeros((len(x),))
        
        dxdt[0] = vx_a    # x-position increment (airspeed along x-direction)
        dxdt[1] = vy_a    # y-position increment (airspeed along y-direction)
        dxdt[2] = vz_a    # z-position increment (airspeed along z-direction)
        
        dxdt[3]  = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta        # Euler's angle phi increment
        dxdt[4]  = q * cos_phi - r * sin_phi                                    # Euler's angle theta increment
        dxdt[5]  = q * sin_phi / cos_theta + r * cos_phi / cos_theta            # Euler's angle psi increment
        
        dxdt[6]  = ((sin_theta * cos_psi * cos_phi + sin_phi * sin_psi) * T - fe_drag[0]) / m   # Acceleration along x-axis
        dxdt[7]  = ((sin_theta * sin_psi * cos_phi - sin_phi * cos_psi) * T - fe_drag[1]) / m   # Acceleration along y-axis
        dxdt[8]  = - self.parameters['gravity'] + (cos_phi * cos_theta  * T - fe_drag[2]) / m   # Acceleration along z-axis

        dxdt[9]  = ((Iyy - Izz) * q * r + tp * self.vehicle_model.geom['arm_length']) / Ixx     # Angular acceleration along body x-axis: roll rate
        dxdt[10] = ((Izz - Ixx) * p * r + tq * self.vehicle_model.geom['arm_length']) / Iyy     # Angular acceleration along body y-axis: pitch rate
        dxdt[11] = ((Ixx - Iyy) * p * q + tr *        1               ) / Izz                   # Angular acceleration along body z-axis: yaw rate
        dxdt[12] = 1                                                                            # Auxiliary time variable

        # Set vehicle state:
        state_temp = np.array([x[iter] for iter in x.keys()])
        self.vehicle_model.set_state(state=state_temp + dxdt*self.parameters['dt'])
        
        # I'd suggest a more compact way of generating the StateContainer
        return self.StateContainer(np.array([np.atleast_1d(item) for item in dxdt]))
    
    def event_state(self, x : dict) -> dict:

        # Based on time 
        return {
                'TrajectoryComplete': x['t']/self.parameters['ref_traj']['t'][-1]
        }

        # Based on reference trajectory
        # # Extract next waypoint information 
        # num_wypts = len(self.parameters['waypoints']['waypoints_time']) - 1 # Don't include initial waypoint
        # index_next = self.parameters['waypoints']['next_waypoint']

        # # Check if at intial waypoint. If so, event_state is 1
        # if index_next == 0:
        #     self.parameters['waypoints']['next_waypoint'] = 1
        #     return {
        #         'TrajectoryComplete': 1
        #     }
        # # Check if passed final waypoint. If so, event_state is 0
        # if index_next > num_wypts:
        #     return {
        #         'TrajectoryComplete': 0
        #     }
        
        # t_next = self.parameters['waypoints']['waypoints_time'][index_next]
        # x_next = self.parameters['waypoints']['waypoints_x'][index_next]
        # y_next = self.parameters['waypoints']['waypoints_y'][index_next]
        # z_next = self.parameters['waypoints']['waypoints_z'][index_next]

        # # Define time interval for acceptable arrival at waypoint
        # time_buffer_left = (self.parameters['waypoints']['waypoints_time'][index_next] - self.parameters['waypoints']['waypoints_time'][index_next - 1])/2
        # if index_next == num_wypts:
        #     # Final waypoint, add final buffer time 
        #     time_buffer_right = t_next + self.parameters['final_time_buffer_sec']
        # else: 
        #     time_buffer_right = (self.parameters['waypoints']['waypoints_time'][index_next+1] - self.parameters['waypoints']['waypoints_time'][index_next])/2

        # # Check if next waypoint is satisfied:
        # if x['t'] < t_next - time_buffer_left:
        #     # Not yet within time of next waypoint
        #     return {
        #             'TrajectoryComplete': (num_wypts - (index_next - 1))/num_wypts
        #         }
        # elif t_next - time_buffer_left <= x['t'] <= t_next + time_buffer_right:
        #     # Current time within ETA interval. Check if distance also within acceptable range
        #     dist_now = np.sqrt((x['x']-x_next)**2 + (x['y']-y_next)**2 + (x['z']-z_next)**2)
        #     if dist_now <= self.parameters['final_space_buffer_m']:
        #         # Waypoint achieved
        #         self.parameters['waypoints']['next_waypoint'] += 1
        #         return {
        #             'TrajectoryComplete': (num_wypts - index_next)/num_wypts
        #         }
        #     else:
        #         # Waypoint not yet achieved
        #         return {
        #             'TrajectoryComplete': (num_wypts - (index_next - 1))/num_wypts
        #         }
        # else:
        #     # ETA passed before waypoint reached 
        #     warn("Trajectory may not have reached waypoint associated with ETA of {})".format(t_next))
        #     self.parameters['waypoints']['next_waypoint'] += 1
        #     return {
        #             'TrajectoryComplete': (num_wypts - index_next)/num_wypts
        #         }
 
    def output(self, x : dict):
        # Output is the same as the state vector, without time 
        return self.OutputContainer(x.matrix[0:-1])


    def threshold_met(self, x : dict) -> dict:
        # threshold_met is defined based on success of completing the reference trajectory
        # For threshold_met to evaluate as True, the vehicle must be within a defined sphere around the final point in the reference trajectory, within some acceptable time interval
        t_lower_bound = self.parameters['ref_traj']['t'][-1] - (self.parameters['ref_traj']['t'][-1] - self.parameters['ref_traj']['t'][-2])/2
        t_upper_bound = self.parameters['ref_traj']['t'][-1] + self.parameters['final_time_buffer_sec']
        if x['t'] < t_lower_bound:
            # Trajectory hasn't reached final ETA
            return {
                'TrajectoryComplete': False
            }
        elif t_lower_bound <= x['t'] <= t_upper_bound:
            # Trajectory is within bounds of final ETA
            dist_now = np.sqrt((x['x']-self.parameters['ref_traj']['x'][-1])**2 + (x['y']-self.parameters['ref_traj']['y'][-1])**2 + (x['z']-self.parameters['ref_traj']['z'][-1])**2)
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

        # Check for appropriately defined dt - must be same as vehicle model 
        if 'dt' in kwargs and kwargs['dt'] != self.parameters['dt']:
          kwargs['dt'] = self.parameters['dt']
          warn("Simulation dt must be equal to dt defined for the vehicle model. dt = {} is used.".format(self.parameters['dt'])) 
        elif 'dt' not in kwargs:
          kwargs['dt'] = self.parameters['dt']

        # Simulate to threshold at DMD time step
        sim_res = super().simulate_to_threshold(future_loading_eqn,first_output, threshold_keys, **kwargs)

        return sim_res

    def linear_model(self, phi, theta, psi, p, q, r, T):
        """ The linear model ignores gyroscopic effect and wind rate of change"""
        m         = self.vehicle_model.mass['total']
        Ixx       = self.vehicle_model.mass['Ixx']
        Iyy       = self.vehicle_model.mass['Iyy']
        Izz       = self.vehicle_model.mass['Izz']
        l         = self.vehicle_model.geom['arm_length'] # Is this correct? 
        sin_phi   = np.sin(phi)
        cos_phi   = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        tan_theta = np.tan(theta)
        sin_psi   = np.sin(psi)
        cos_psi   = np.cos(psi)

        A = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
                      [0, 0, 0, q*cos_phi*tan_theta - r*sin_phi*tan_theta, q*(tan_theta**2 + 1)*sin_phi + r*(tan_theta**2 + 1)*cos_phi, 0, 0, 0, 0, 1, sin_phi*tan_theta, cos_phi*tan_theta], 
                      [0, 0, 0, -q*sin_phi - r*cos_phi, 0, 0, 0, 0, 0, 0, cos_phi, -sin_phi], 
                      [0, 0, 0, q*cos_phi/cos_theta - r*sin_phi/cos_theta, q*sin_phi*sin_theta/cos_theta**2 + r*sin_theta*cos_phi/cos_theta**2, 0, 0, 0, 0, 0, sin_phi/cos_theta, cos_phi/cos_theta], 
                      [0, 0, 0, T*(-sin_phi*sin_theta*cos_psi + sin_psi*cos_phi)/m, T*cos_phi*cos_psi*cos_theta/m, T*(sin_phi*cos_psi - sin_psi*sin_theta*cos_phi)/m, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 0, T*(-sin_phi*sin_psi*sin_theta - cos_phi*cos_psi)/m, T*sin_psi*cos_phi*cos_theta/m, T*(sin_phi*sin_psi + sin_theta*cos_phi*cos_psi)/m, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 0, -T*sin_phi*cos_theta/m, -T*sin_theta*cos_phi/m, 0, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r*(Iyy - Izz)/Ixx, q*(Iyy - Izz)/Ixx], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, r*(-Ixx + Izz)/Iyy, 0, p*(-Ixx + Izz)/Iyy], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, q*(Ixx - Iyy)/Izz, p*(Ixx - Iyy)/Izz, 0]])

        B = np.array([ [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [(sin_phi*sin_psi + sin_theta*cos_phi*cos_psi)/m, 0, 0, 0], 
                       [(-sin_phi*cos_psi + sin_psi*sin_theta*cos_phi)/m, 0, 0, 0], 
                       [cos_phi*cos_theta/m, 0, 0, 0], 
                       [0, l/Ixx, 0, 0], 
                       [0, 0, l/Iyy, 0], 
                       [0, 0, 0, 1.0/Izz]])

        return A, B

    def visualize_traj(self, pred):
        """
        This method provides functionality to visualize a predicted trajectory generated, plotted with the reference trajectory and coarse waypoints. 

        Calling this returns a figure with two subplots: 1) x vs y, and 2) z vs time.

        Parameters
        ----------
        pred : UAVGen model simulation  
               SimulationResults from simulate_to or simulate_to_threshold for a defined UAVGen class

        Returns 
        -------
        fig : Visualization of trajectory generation results 
        """

        import matplotlib.pyplot as plt

        # Conversions
        # -----------
        deg2rad = np.pi/180.0
        rad2deg = 180.0/np.pi
        
        # Extract reference trajectory information
        # ----------------------------------------
        depart_time = self.parameters['ref_traj']['t'][0] # self.ref_traj.route.departure_time
        time        = self.parameters['ref_traj']['t']
        ref_x       = self.parameters['ref_traj']['x'].tolist()
        ref_y       = self.parameters['ref_traj']['y'].tolist()
        ref_z       = self.parameters['ref_traj']['z'].tolist() 

        # Extract predicted trajectory information
        # ----------------------------------------
        pred_time = pred.times
        pred_x = [pred.outputs[iter]['x'] for iter in range(len(pred_time))]
        pred_y = [pred.outputs[iter]['y'] for iter in range(len(pred_time))]
        pred_z = [pred.outputs[iter]['z'] for iter in range(len(pred_time))]

        # Initialize Figure
        # ----------------
        params = dict(figsize=(13, 9), fontsize=14, linewidth=2.0, alpha_preds=0.6)
        fig, (ax1, ax2) = plt.subplots(2)

        # Plot trajectory predictions
        # -------------------------
        # First plot waypoints (dots) and reference trajectory (commanded, line)
        # ax1.plot(waypoints.lon * rad2deg, waypoints.lat  * rad2deg, 'o', color='tab:orange', alpha=0.5, markersize=10, label='__nolegend__')
        ax1.plot(ref_x, ref_y, '--', linewidth=params['linewidth'], color='tab:orange', alpha=0.5, label='reference trajectory')
        ax1.plot(pred_x, pred_y,'-', color='tab:blue', alpha=params['alpha_preds'], linewidth=params['linewidth'], label='prediction')

        ax1.set_xlabel('x', fontsize=params['fontsize'])
        ax1.set_ylabel('y', fontsize=params['fontsize'])
        ax1.legend(fontsize=params['fontsize'])

        # Add altitude plot
        # -------------------------------------------------------
        # time_vector = [depart_time + datetime.timedelta(seconds=pred_time[ii]) for ii in range(len(pred_time))]
        # ax2.plot_date(eta, waypoints.alt, '--o', color='tab:orange', alpha=0.5, linewidth=params['linewidth'], label='__nolegend__')
        ax2.plot(time, ref_z, '-', color='tab:orange', alpha=params['alpha_preds'], linewidth=params['linewidth'], label='reference trajectory')
        ax2.plot(pred_time, pred_z,'-', color='tab:blue',alpha=params['alpha_preds'], linewidth=params['linewidth'], label='prediction')
        
        ax2.set_xlabel('time stamp, -', fontsize=params['fontsize'])
        ax2.set_ylabel('z', fontsize=params['fontsize'])

        return fig
