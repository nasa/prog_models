

# import prog_models.models.trajectory_generation.trajectory.route as route 
# import prog_models.models.trajectory_generation.trajectory.trajectory as trajectory
# from prog_models.models.trajectory_generation.vehicles import AircraftModels
# from prog_models.models.trajectory_generation.vehicles.control import allocation_functions

from prog_models.models.trajectory_generation.trajectory_generation import TrajGen 

import numpy as np

def run_example(): 

    traj_gen = TrajGen()

    x0_test = traj_gen.initialize()


    debug = 1







# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()