# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from prog_models import PrognosticsModel
from numpy import inf

class ParisLawCrackGrowth(PrognosticsModel): 
    """
    A simple Paris Law Model Implementation

    Events: (1)
        CGF :   Crack Growth Fracture 

    Inputs/Loading: (2)
      |k_max :  Maximum crack growth 
      |k_min :  Minimum crack growth 

    States: (1)
      |c_l :    crack length
  
    Outputs: (1)
       |c_l :   crack length

    Model Configuration Parameters:
       |crack_limit : crack length limit after which the crack growth fracture event is triggered
       |c :     Material Constant
       |m :     Material Constant
       |dndt :  cycles per second
    """ 
    # Event: Crack Growth Fracture
    events = ['CGF']
    # Inputs are ['k_min', 'k_max']
    inputs = ['k_max','k_min']
    # State: Crack Length
    states = ['c_l']
    # Output: Crack Length
    outputs = ['c_l']

    # The default parameters
    default_parameters = {
        'crack_limit': 1e-4,
        'c': 3.24,
        'm': 0.1527,
        'dndt': 10, 
        'x0' : {
            'c_l': 0.00001,
        }  
    }
    
    state_limits = {
        'c_l': (0, inf),
    }
    
    def initialize(self, u=None, z=None):
        return self.StateContainer(self.parameters['x0'])

    # The model equations
    def dx(self, x : dict, u : dict):
        parameters = self.parameters
        r = (parameters['c']*(u['k_max'] - u['k_min'])**parameters['m'])*parameters['dndt'] # Paris Law Equation with respect to time
        dxdt = {
             'c_l': r,
         }
        return self.StateContainer(dxdt)

    def output(self, x):
        return self.OutputContainer(x) 

    def event_state(self, x : dict) -> dict: 
       return {
            'CGF' : 1- x['c_l'] / self.parameters['crack_limit']
        }

    def threshold_met(self, x):
        t_met = {
           'CGF': x['c_l'] > self.parameters['crack_limit']
        }
        return t_met
         