# Copyright © 2022 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from .. import PrognosticsModel

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
    inputs = []
    states = []
    outputs = []
    is_vectorized = True

    default_parameters = {  # Set to defaults
    }

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
        pass
        # return self.OutputContainer(np.array([
        #     np.atleast_1d(x['tb']))]))   # v

    def threshold_met(self, x : dict) -> dict:
        pass
        # return {
        #      'EOD': V < parameters['VEOD']
        # }
