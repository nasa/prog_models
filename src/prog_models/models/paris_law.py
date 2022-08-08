# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from prog_models import PrognosticsModel
from numpy import inf

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

    events = ['CGF']
    
    # REPLACE THE FOLLOWING LIST WITH INPUTS (LOADING)
    inputs = ['k_max', 'k_min'] # change

    # REPLACE THE FOLLOWING LIST WITH STATES
    states = ['c_li']

    # REPLACE THE FOLLOWING LIST WITH OUTPUTS (MEASURED VALUES)
    outputs = ['c_li']

    # REPLACE THE FOLLOWING LIST WITH CONFIGURED PARAMETERS
    default_parameters = {# Set default parameters
        'config_length': 1e-4,
        'c': 3.24,
        'm': 0.1527,
        'dndt': 10, 
        'x0' : {
            'c_li': 0.00001,
        }  
    }

    # REPLACE THE FOLLOWING WITH STATE BOUNDS IF NEEDED
    state_limits = {
        # 'state': (lower_limit, upper_limit)
        # only specify for states with limits
        #crack length cannot be 0 else it is not a crack and cannot be infinite length
        'c_li': (0, inf),
    }

    # Identify callbacks used by this model
    # See examples.derived_params
    # Format: "trigger": [callbacks]
    # Where trigger is the parameter that the derived parameters are derived from.
    # And callbacks are one or more callback functions that define parameters that are 
    # derived from that parameter
    # REPLACE THIS WITH ACTUAL DERIVED PARAMETER CALLBACKS
    #param_callbacks = {
    #    "Example Parameter 2": [example_callback]
    #}


    #def __init__(self, **kwargs):
        
        # ADD OPTIONS CHECKS HERE

        # e.g., Checking for required parameters
        # if not 'required_param' in kwargs: 
        #   throw Exception;

        # e.g. 2, Modify parameters
        # kwargs['some_param'] = some_function(kwargs['some_param'])

    #super().__init__(**kwargs) # Run Parent constructor

    # Sometimes initial input (u) and initial output (z) are needed to initialize the model
    # In that case remove the '= None' for the appropriate argument
    # Note: If they are needed, that requirement propogated through to the simulate_to* functions
    def initialize(self, u=None, z=None):
        return self.StateContainer(self.parameters['x0'])

    #- UNCOMMENT THIS FUNCTION FOR CONTINUOUS MODELS
    # update crack length based of derivative of  da/dt
    # cycle rate wont change
    #get the derivative of the crack length
    #multiply by the cycle rate
    def dx(self, x : dict, u : dict):
        parameters = self.parameters
        r = (parameters['c']*(u['k_max'] - u['k_min'])**parameters['m'])*parameters['dndt']#r = C*(k_max / k_min)^m (Stress ratio)
        dxdt = {
             'c_li': r,
         }
        return self.StateContainer(dxdt)

    def output(self, x):
        return self.OutputContainer(x) #disctionary or an array/coloumn vector of outputs

        # when crack length is maximum it should be 0
        # when crack length is 0 it should be 1
        # when the crack length is half the maximum it should be 0.5


    def event_state(self, x : dict) -> dict: #true if it greater than the threashold
       return {
            'CGF' : 1- x['c_li'] / self.parameters['config_length']
        }

       
    # Note: Thresholds met equation below is not strictly necessary. By default threshold_met will check if event_state is ≤ 0 for each event
    # crack length ends when we reach a configurable size
    def threshold_met(self, x):
        t_met = {
           'CGF': x['c_li'] > self.parameters['config_length']
        }
        return t_met 