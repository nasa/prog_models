

# import prog_models.models.trajectory_generation.trajectory.route as route 
# import prog_models.models.trajectory_generation.trajectory.trajectory as trajectory
# from prog_models.models.trajectory_generation.vehicles import AircraftModels
# from prog_models.models.trajectory_generation.vehicles.control import allocation_functions

from prog_models.models.trajectory_generation.trajectory_generation import TrajGen 

import numpy as np

def run_example(): 

    traj_gen = TrajGen()
    traj_gen.parameters['process_noise'] = 0

    x0_test = traj_gen.initialize()

    def future_loading(t, x=None):
        return traj_gen.InputContainer({})  # Loading defined internally 

    options = {
        'dt': 0.2,
        # 'save_freq': 0.2
    }

    # simulated_results = traj_gen.simulate_to_threshold(future_loading, **options)
    simulated_results = traj_gen.simulate_to(425,future_loading, **options)



    debug = 1







# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()