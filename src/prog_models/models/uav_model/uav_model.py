# Copyright © 2022 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

# from .. import PrognosticsModel
from prog_models.prognostics_model import PrognosticsModel
import prog_models.models.uav_model.trajectory.route as route 
import prog_models.models.uav_model.trajectory.trajectory as trajectory
from prog_models.models.uav_model.vehicles import AircraftModels
from prog_models.models.uav_model.vehicles.control import allocation_functions

import numpy as np
import prog_models.models.uav_model.utilities.geometry as geom
from warnings import warn

class UAVGen(PrognosticsModel):
    """

    :term:`Events<event>`: (1)
    
    :term:`Inputs/Loading<input>`: (1)

    :term:`States<state>`: (4)

    :term:`Outputs<output>`: (2)

    Keyword Args
    ------------

    """
    events = [] # fill in ['EOD']
    inputs = ['T','mx','my','mz']
    states = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r']
    outputs = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r']
    is_vectorized = True

    default_parameters = {  # Set to defaults
        # Flight information
        'flight_file': 'src/prog_models/models/uav_model/data/20181207_011200_Flight.txt', 
        'flight_name': 'LaRC_20181207', 
        'aircraft_name': 'aircraft-1', 

        # Simulation parameters:
        'dt': 0.2, 
        'gravity': 9.81,
        'cruise_speed': 6.0,
        'ascent_speed': 3.0,
        'descent_speed': 3.0, 
        'landing_speed': 1.5,
        'hovering_time': 10.0,
        'takeoff_time': 60.0, 
        'landing_time': 60.0, 
        'nurbs_basis_length': 2000, 
        'nurbs_order': 5, 

        # Vehicle params:
        'vehicle_model': 'djis1000',
        'vehicle_payload': 0.0,
        'vehicle_integrator_fn': 'RK4'
    }

    def initialize(self, u=None, z=None): # initialize(self, u : dict, z = None):
        
        flightplan = trajectory.load.get_flightplan(fname=self.parameters['flight_file'])
        lat, lon, alt, tstamps = flightplan['lat'], flightplan['lon'], flightplan['alt'], flightplan['timestamp']

        # Generate route
        route_ = route.build(name=self.parameters['flight_name'], lat=lat, lon=lon, alt=alt, departure_time=tstamps[0],
                            cruise_speed=self.parameters['cruise_speed'], 
                            ascent_speed=self.parameters['ascent_speed'], 
                            descent_speed=self.parameters['descent_speed'], 
                            landing_speed=self.parameters['landing_speed'],
                            hovering_time=self.parameters['hovering_time'], 
                            add_takeoff_time=self.parameters['takeoff_time'], 
                            add_landing_time=self.parameters['landing_time'])
        # Generate trajectory
        ref_traj = trajectory.Trajectory(name=self.parameters['flight_name'], route=route_)
        ref_traj.generate(dt=self.parameters['dt'], 
                        nurbs_order=self.parameters['nurbs_order'], 
                        gravity=self.parameters['gravity'], 
                        nurbs_basis_length=self.parameters['nurbs_basis_length'])

        self.ref_traj = ref_traj

        # Initialize vehicle 
        init_pos = np.concatenate((ref_traj.cartesian_pos[0,:], ref_traj.attitude[0,:], 
                                    ref_traj.velocity[0,:], ref_traj.angular_velocity[0,:]), axis=0)
        aircraft1 = AircraftModels.build_model(init_pos, 
                                               ref_traj.dt, 
                                               name=self.parameters['aircraft_name'], 
                                               model=self.parameters['vehicle_model'], 
                                               integrator_fn=self.parameters['vehicle_integrator_fn'],
                                               payload=self.parameters['vehicle_payload'])
        self.vehicle_model = aircraft1 

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
            'r': ref_traj.angular_velocity[0,2]
            })

    def dx(self, x : dict, u : dict):
        # Extract params
        # -------------
        # wind = self.parameters['wind']
        wx = 0 #wind['u']
        wy = 0 #wind['v']

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
        v_earth = np.dot(geom.rot_earth2body(phi, theta, psi),
                        np.array([vx_a - wx, vy_a - wy, vz_a]).reshape((-1,)))
        v_body = np.dot(geom.rot_earth2body(phi, theta, psi), v_earth)
        fb_drag = self.vehicle_model.aero['drag'](v_body)
        fe_drag = np.dot(geom.rot_body2earth(phi, theta, psi), fb_drag)
        
        # Update state vector
        # -------------------
        dxdt     = np.zeros((len(x),))
        
        dxdt[0] = vx_a + wx   # add wind u-component to generate ground speed
        dxdt[1] = vy_a + wy   # add wind v-component to generate ground speed
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
        ]))
    
    def event_state(self, x : dict) -> dict:
        pass
        # return {
        #     'event_name': event_val
        # }

    def output(self, x : dict):
        # Currently, output is the same as the state vector
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
            'r': x['r']
            })

    def threshold_met(self, x : dict) -> dict:
        pass
        # return {
        #      'EOD': V < parameters['VEOD']
        # }

    def simulate_to_threshold(self, future_loading_eqn, first_output = None, threshold_keys = None, **kwargs):
        
        if kwargs['dt'] != self.parameters['dt']:
            warn("dt must be equal to simulation dt value. To change the simulate_to time step, change the simulation parameter 'dt' value")
        kwargs['dt'] = self.parameters['dt']
        kwargs['save_freq'] = self.parameters['dt']

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
                x_temp = np.array([x.matrix[ii][0] for ii in range(len(x.matrix))])
                u = self.vehicle_model.control_scheduled(x_temp - ref_now) 
                u[0]      += self.vehicle_model.steadystate_input
                u[0]       = min(max([0., u[0]]), self.vehicle_model.dynamics['max_thrust'])
                return self.InputContainer({'T': u[0], 'mx': u[1], 'my': u[2], 'mz': u[3]})

        # Simulate to threshold 
        results = super().simulate_to_threshold(future_loading_new,first_output, threshold_keys, **kwargs)
        return results 
