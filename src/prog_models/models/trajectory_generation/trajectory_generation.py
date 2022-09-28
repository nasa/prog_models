# Copyright © 2022 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

# from .. import PrognosticsModel
from prog_models.prognostics_model import PrognosticsModel
import prog_models.models.trajectory_generation.trajectory.route as route 
import prog_models.models.trajectory_generation.trajectory.trajectory as trajectory
from prog_models.models.trajectory_generation.vehicles import AircraftModels
from prog_models.models.trajectory_generation.vehicles.control import allocation_functions

import numpy as np


class TrajGen(PrognosticsModel):
    """

    :term:`Events<event>`: (1)
    
    :term:`Inputs/Loading<input>`: (1)

    :term:`States<state>`: (4)

    :term:`Outputs<output>`: (2)

    Keyword Args
    ------------

    """
    events = [] # fill in ['EOD']
    inputs = ['u_x', 'u_y', 'u_z', 'u_phi', 'u_theta', 'u_psi', 'u_vx', 'u_vy', 'u_vz', 'u_p', 'u_q', 'u_r']
    states = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r']
    outputs = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r']
    is_vectorized = True

    default_parameters = {  # Set to defaults
        # Flight information
        'flight_file': 'src/prog_models/models/trajectory_generation/data/20181207_011200_Flight.txt', 
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
        # self.parameters['ref_traj'] = ref_traj

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
        pass 
        # return self.StateContainer(np.array([
        #     np.atleast_1d(state1),  # fill in, one for each dx state 
        # ]))
    
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
