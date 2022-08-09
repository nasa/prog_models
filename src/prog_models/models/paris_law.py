# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from prog_models import PrognosticsModel
from numpy import inf

# Paris Law model that simulates 
# Fatigue crack growth testing measures rate of advance of a fatigue crack in terms of the applied driving force for growth using 
# Linear Elastic Fracture Mechanics (LEFM) principles. 
# The range of stress intensity factor in the loading cycle (ΔK), is used as the driving force parameter.
class CrackGrowth(PrognosticsModel): 
    """
    Events: (1)
        CGF :   Crack Growth Fracture 

    Inputs/Loading: (2)
      |k_max :  Maximum crack growth 
      |k_min :  Minimum crack growth 

    States: (1)
      |c_li :    crack length # wrong Yes, Yes One state
  
    Outputs: (1) #output  user needs to be able to measure
       |c_l :   crack length

    Model Configuration Parameters:
       |a :     Length of crack
       |c :     Constant
       |m :     Constant
       |n :     cycles per loading
    """ 
    # Event: Crack Growth Fracture
    events = ['CGF']
    # Inputs are ['k_min', 'k_max']
    inputs = ['k_max','k_min']
    # State: Crack Length
    states = ['c_li']
    # Output: Crack Length
    outputs = ['c_li']

    # The default parameters
    default_parameters = {
        'config_length': 1e-4,
        'c': 3.24,
        'm': 0.1527,
        'dndt': 10, 
        'x0' : {
            'c_li': 0.00001,
        }  
    }
    
    state_limits = {
        'c_li': (0, inf),
    }
    
    def initialize(self, u=None, z=None):
        return self.StateContainer(self.parameters['x0'])

    # The model equations
    def dx(self, x : dict, u : dict):
        parameters = self.parameters
        r = (parameters['c']*(u['k_max'] - u['k_min'])**parameters['m'])*parameters['dndt'] # Paris Law Equation with respect to time
        dxdt = {
             'c_li': r,
         }
        return self.StateContainer(dxdt)

    def output(self, x):
        return self.OutputContainer(x) 

    def event_state(self, x : dict) -> dict: 
       return {
            'CGF' : 1- x['c_li'] / self.parameters['config_length']
        }

    def threshold_met(self, x):
        t_met = {
           'CGF': x['c_li'] > self.parameters['config_length']
        }
        return t_met
         