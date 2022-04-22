# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example demonstrating and benchmarking the DMD surrogate model 

Compares simulations: 1) Full model, 2) DMD approximation in next_state, 3) DMD + Linear Model (including class definition)
"""

from prog_models.models.battery_electrochem import BatteryElectroChemEOD, DMDModel_dx
from prog_models import LinearModel
from statistics import mean
import numpy as np
from numpy.random import normal
from timeit import timeit
import matplotlib.pyplot as plt
import time

def run_example_Full(): 
    batt = BatteryElectroChemEOD()
    batt.parameters['process_noise'] = 0
    batt.parameters['process_noise']['qpB'] = 0.25
    batt.parameters['process_noise']['qpS'] = 0.25

    ## Define Loading Profile
    def future_loading(t, x=None):
        if (t < 200):
            i = 1
        elif (t < 800):
            i = 3
        elif (t < 1000):
            i = 5
        else:
            i = 4
        return {'i': i}

    # Simulation Options
    options = {
        'save_freq': 1,  # Frequency at which results are saved
        'dt': 0.1  # Timestep
    }
    
    ## Simulate to Threshold 
    simulated_results = batt.simulate_to_threshold(future_loading, **options)

    ## Simulate to specific time 
    simulated_results = batt.simulate_to(600, future_loading, **options)

    ## Benchmarking
    print('Benchmarking Full Simulation:')
    def sim_Full():  
        simulated_results = batt.simulate_to_threshold(future_loading, **options)
    timeFull = timeit(sim_Full, number=100)

    # Print results
    print('Simulation Time from Full Simulation: {} ms/sim'.format(timeFull*2))


def run_example_DMD_dx(): 
    batt = DMDModel_dx()
    batt.parameters['process_noise'] = 0

    ## Define Loading Profile 
    def future_loading(t, x=None):
        if (t < 500):
            i = 3
        elif (t < 1000):
            i = 1
        elif (t < 2000):
            i = 2
        else:
            i = 4
        return {'i': i}

    # Simulation Options
    options = {
        'save_freq': 1,  # Frequency at which results are saved
        'dt': 1  # Timestep
    }

    ## Simulate to Threshold 
    simulated_results = batt.simulate_to_threshold(future_loading, **options)

    ## Simulate to specific time 
    # simulated_results = batt.simulate_to(600,future_loading,**options)

    ## Benchmarking
    print('Benchmarking DMD (in next_state):')
    def sim_DMD():  
        simulated_results = batt.simulate_to_threshold(future_loading, **options)
    timeDMD = timeit(sim_DMD, number=100)

    # Print results
    print('Simulation Time from DMD (in next_state): {} ms/sim'.format(timeDMD*2))


def run_example_DMD_linear(): 

    class DMDModel_Linear(LinearModel, BatteryElectroChemEOD):
        """
        This class is a subclass of LinearModel and uses Dynamic Mode Decomposition to simulate a battery throughout time.
        Given an initial state and the current imposed, it returns the states of an electrochemistry battery model, 
        voltage, and SOC throughout time until threshold is met. 

        See BatteryElectroChemEOD and LinearModel for more details. 
        
        This class 1) defines the DMD matrix learned from data (and specific to the electrochemistry battery model), 
                   2) sets up the matrices necessary to solve the LinearModel, 
                   3) defines next_state to override the function dx in LinearModel, since DMD calculates the next state at time t+1 given the state at time t
        """

        # Matrix for state order: V0, Vsn, Vsp, tb, qpS, qpB, qnS, qnB, voltage, SOC 
        mat_DMD = np.array([[0.846432997947377, -0.000667098306969, -0.003499055627984, -0.000005962670043, 0.000002538926066, -0.000000079499872, -0.000002494565138, 0.000000479248614, -0.000135296929176, -0.000000000265174, 0.017926553930082],
                [0.000013805997785, 0.999127480001299, 0.011748463029297, -0.000004303858395, 0.000000051919248, 0.000000095473959, -0.000000030536919, 0.000000099949084, 0.000038090840986, 0.000000000009139, 0.000020598758753],
                [0.000186666241108, -0.000104306447939, 0.998182329965222, -0.000002492017443, -0.000001048132155, 0.000000176230319, 0.000001059990774, -0.000000061631353, 0.000016326810677, 0.000000000131199, 0.000012977790646],
                [0.194410212897001, -0.055385368247585, -3.005033376088249, 0.993595612820197, -0.000492808908563, 0.000212734643755, 0.000527230492758, 0.000097035884052, 0.017357750313124, 0.000000082140805, 0.008225955453474],
                [-0.049104731005372, -0.035644705365939, -0.225643302430399, -0.000301861153368, 0.471771055314256, 0.058702377415622, -0.459573684501072, 0.051073957980525, -0.006821704390177, -0.000053749963979, 0.970318358923638],
                [-0.001634912321492, -0.000874072105944, -0.024119154782966, 0.000007075448195, 0.089347859313690, 0.990072258802371, 0.020408191425950, -0.002267802336599, 0.000100118815681, 0.000002386893305, 0.031570749150319],
                [0.049104730954516, 0.035644705123843, 0.225643301950186, 0.000301861153828, -0.459575932436164, 0.051053720042373, 0.471768805267909, 0.058682137366119, 0.006821704338023, 0.000069796176625, -0.970318358923819],
                [0.001634912474003, 0.000874072838087, 0.024119155714288, -0.000007075447954, 0.020408246589364, -0.002267381680534, 0.089347895475308, 0.990072660456291, -0.000100118700630, 0.000142029020513, -0.031570749155353],
                [0.153518502044249, 0.009233498282786, -0.065880970046692, 0.000013386722553, 0.000033855152439, -0.000004230502556, -0.000033952228789, 0.000003357002074, 1.000196952146898, -0.000000004025677, -0.018889419478453],
                [0.006676268871134e-03, 0.004805102383654e-03, 0.032863481145284e-03, 0.000038787592921e-03, -0.057785221821960e-03, 0.006419255047610e-03, 0.073831144834634e-03, 0.137994052345052e-03, 0.000884419152636e-03, 0.000027871736466e-03,  -0.131827514220952e-03],
                ])

        # Define matrices for linear approximation (see LinearModel)
        A = mat_DMD[:,:-1]
        B = np.vstack(mat_DMD[:,-1])
        C = np.zeros((1,10))
        C[0,-2] = 1
        F = np.zeros((1,10))
        F[0,-1] = 1

        states = BatteryElectroChemEOD.states + ['v','SOC']
        outputs = ['v']
        inputs = ['i']

        def initialize(self, u=None, z=None):
            # x = BatteryElectroChemEOD.initialize(self)
            # x['v'] = BatteryElectroChemEOD.output(self, x)['v']
            # x['SOC'] = 1 # Arbitrary value necessary for consistency in matrix size 
            # x['SOC'] = BatteryElectroChemEOD.event_state(self,x)['EOD']
            # return x
            return self.StateContainer({
                'tb': self.parameters['x0']['tb'],
                'Vo': self.parameters['x0']['Vo'],
                'Vsn': self.parameters['x0']['Vsn'],
                'Vsp': self.parameters['x0']['Vsp'],
                'qnB': self.parameters['x0']['qnB'],
                'qnS': self.parameters['x0']['qnS'],
                'qpB': self.parameters['x0']['qpB'],
                'qpS': self.parameters['x0']['qpS'],
                'v': 3.2,
                'SOC': 1
            })

        def next_state(self, x, u, dt):   
            # x_array = np.array([list(x.values())]).T
            # u_array = np.array([list(u.values())]).T

            # next_array = np.matmul(self.A, x_array) + np.matmul(self.B, u_array) + self.E
            # return {key: value[0] for key, value in zip(self.states, next_array)}
            x.matrix = np.matmul(self.A, x.matrix) + np.matmul(self.B, u.matrix) + self.E
            return x

    ### Simulate battery using DMD + LinearModel 
    # Generate class object 
    batt = DMDModel_Linear()
    batt.parameters['process_noise'] = 0

    # Simulation Options
    options = {
        'save_freq': 1,  # Frequency at which results are saved
        'dt': 1  # Timestep - for now, this is determined by the DMD Matrix and how it is defined
    }

    # Define Loading Profile 
    def future_loading(t, x=None):
        # Adjust time to previous time step for DMD consistency
            # simulate_to_threshold in PrognosticsModel calculates load at the next time point and uses this as input to next_state
            # DMD, however, takes the state and load at a particular (same) time, and uses this to calculate the state at the next time 
            # Thus, when calling future_loading with DMD + LinearModel, we need the load input for next_state to be at the previous time point to be consistent with the previous state, so we subtract dt from the input time 
        # Note: this should be made more rigorous in the future 
        if t == 0:
            t = 0
        else:
            t = t - options['dt'] 

        if (t < 60):
            i = 3
        elif (t < 120):
            i = 5
        elif (t < 180):
            i = 4
        elif (t < 250):
            i = 2
        elif (t < 300):
            i = 1
        else:
            i = 6
        return batt.InputContainer({'i': i})

    # Simulate to threshold
    debug = 2
    simulated_results = batt.simulate_to_threshold(future_loading,**options)

    # Simulate to a specific time 
    # simulated_results = batt.simulate_to(350,future_loading,**options)

    # Benchmarking
    # print('Benchmarking DMD + LinearModel:')
    # def sim_DMD_linear():  
    #     simulated_results = batt.simulate_to_threshold(future_loading, **options)
    # timeDMD_linear = timeit(sim_DMD_linear, number=100)

    # Print results
    # print('Simulation Time from DMD + LinearModel: {} ms/sim'.format(timeDMD_linear*2))


# This allows the module to be executed directly 
if __name__ == '__main__':
    # Run Full Simulation:
    run_example_Full()
    
    # Run DMD approximation in next_state
    # run_example_DMD_dx()

    # Run DMD approximation using LinearModel
    # run_example_DMD_linear()