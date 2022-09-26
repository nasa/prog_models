

import prog_models.models.trajectory_generation.trajectory.route as route 
import prog_models.models.trajectory_generation.trajectory.trajectory as trajectory
from prog_models.models.trajectory_generation.vehicles import AircraftModels
from prog_models.models.trajectory_generation.vehicles.control import allocation_functions

import numpy as np

def run_example(): 

    # Flight info
    # ----------------------------------------------
    flight_file    = 'src/prog_models/models/trajectory_generation/data/20181207_011200_Flight.txt'  # flight plan file txt
    flight_name    = 'LaRC_20181207'                    # give it a name for saving
    aircraft_name  = 'aircraft-1'                       # give it a name for saving

    # Simulation params
    # --------------
    sim_params = dict(dt=0.2, # s, simulation dt
                      gravity=9.81)       # m/s^2, gravity magnitude

    traj_speed_params = dict(cruise=6.0,  # m/s, avg speed in-between way-points
                             ascent=3.0,  # m/s, vertical speed (up)
                             descent=3.0, # m/s, vertical speed (down)
                             landing=1.5)  # m/s, landing speed when altitude < 10m
    traj_add_time_params = dict(hovering=10.0,   # s, only if necessary
                                takeoff= 60.0,  # s, only if necessary
                                landing=60.0)    # s, only if necessary
    traj_ref_params = dict(nurbs_basis_length=2000, nurbs_order=5)  # Nurbs algorithm (do not touch unless traj generation fails)

    # SELECT FLIGHT PLAN AND CREATE TRAJECTORY
    # =======================================
    flightplan = trajectory.load.get_flightplan(fname=flight_file)
    lat, lon, alt, tstamps = flightplan['lat'], flightplan['lon'], flightplan['alt'], flightplan['timestamp']

    # Generate route
    route_ = route.build(name=flight_name, lat=lat, lon=lon, alt=alt, departure_time=tstamps[0],
                         cruise_speed=traj_speed_params['cruise'], 
                         ascent_speed=traj_speed_params['ascent'], 
                         descent_speed=traj_speed_params['descent'], 
                         landing_speed=traj_speed_params['landing'],
                         hovering_time=traj_add_time_params['hovering'], 
                         add_takeoff_time=traj_add_time_params['takeoff'], 
                         add_landing_time=traj_add_time_params['landing'])
    # Generate trajectory
    ref_traj = trajectory.Trajectory(name=flight_name, route=route_)
    ref_traj.generate(dt=sim_params['dt'], 
                      nurbs_order=traj_ref_params['nurbs_order'], 
                      gravity=sim_params['gravity'], 
                      nurbs_basis_length=traj_ref_params['nurbs_basis_length'])

    init_pos = np.concatenate((ref_traj.cartesian_pos[0,:], ref_traj.attitude[0,:], 
                                ref_traj.velocity[0,:], ref_traj.angular_velocity[0,:]), axis=0)

    # Aircraft Model:
     # Variable vehicle_params contains only a subset of all the vehicle parameters, since some are already defined by default 
    # by selecting model = 'djis1000'
    vehicle_params = dict(model = 'djis1000',       # select aircraft model (only DJIS1000 for now)
                          payload=0.0,              # kg, payload mass
                          integrator_fn='RK4')      # -, integrator

    aircraft1 = AircraftModels.build_model(init_pos, 
                                           ref_traj.dt, 
                                           name=aircraft_name, 
                                           model=vehicle_params['model'], 
                                           integrator_fn=vehicle_params['integrator_fn'],
                                           payload=vehicle_params['payload'])


    debug = 1







# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()