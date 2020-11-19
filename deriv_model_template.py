# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from prog_models import deriv_prog_model

class ProgModelTemplate(deriv_prog_model.DerivProgModel):
    """
    Template for Deriv Prognostics Model
    """

    # REPLACE THE FOLLOWING LIST WITH EVENTS BEING PREDICTED
    events = [
        'Example Event' 
    ]
    
    # REPLACE THE FOLLOWING LIST WITH INPUTS (LOADING)
    inputs = [
        'Example Input 1',
        'Example Input 2'
    ]

    # REPLACE THE FOLLOWING LIST WITH STATES
    states = [
        'Examples State 1',
        'Examples State 2',
        'Examples State 3',
        'Examples State 4'
    ]

    # REPLACE THE FOLLOWING LIST WITH OUTPUTS (MEASURED VALUES)
    outputs = [
        'Example Output 1', 
        'Example Output 2'  
    ]

    # REPLACE THE FOLLOWING LIST WITH CONFIGURED PARAMETERS
    default_parameters = { # Set default parameters
        'Example Parameter 1': 0,
        'process_noise': 0.1, # Process noise
    }

    def __init__(self, options = {}):
        """
        Constructor for model
        """

        # ADD OPTIONS CHECKS HERE

        super().__init__(options) # Run Parent constructor

    def initialize(self, u, z):
        """
        Calculate initial state given inputs and outputs

        Parameters
        ----------
        u : dict
            Inputs, with keys defined by model.inputs.
            e.g., u = {'i':3.2} given inputs = ['i']
        z : dict
            Outputs, with keys defined by model.outputs.
            e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']

        Returns
        -------
        x : dict
            First state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        """

        # REPLACE BELOW WITH LOGIC TO CALCULATE INITIAL STATE
        # NOTE: KEYS FOR x0 MATCH 'states' LIST ABOVE

        # YOU CAN ACCESS ANY PARAMETERS USING self.parameters[key]
        x0 = {
            'Examples State 1': 99.2,
            'Examples State 2': False,
            'Examples State 3': 44,
            'Examples State 4': [1, 2, 3]
        }
        return x0

    def dx(self, t, x, u, dt):
        """
        Returns the first derivative of state `x` at a specific time `t`, given state and input

        Parameters
        ----------
        t : number
            Current timestamp in seconds (≥ 0)
            e.g., t = 3.4
        x : dict
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        u : dict
            Inputs, with keys defined by model.inputs.
            e.g., u = {'i':3.2} given inputs = ['i']

        Returns
        -------
        dx : dict
            First derivitive of state, with keys defined by model.states
            e.g., dx = {'abc': 3.1, 'def': -2.003} given states = ['abc', 'def']
        """

        dx = {key: 0 for key in self.states}
        # ADD LOGIC TO CALCULATE dx (first derivative of x)

        # Apply Process Noise and return
        next_x = self._Model__apply_process_noise(next_x) 
        return next_x

    def output(self, t, x):
        """
        Calculate next statem, forward one timestep

        Parameters
        ----------
        t : number
            Current timestamp in seconds (≥ 0.0)
            e.g., t = 3.4
        x : dict
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        z : dict
            Outputs, with keys defined by model.outputs.
            e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']
        """

        # REPLACE BELOW WITH LOGIC TO CALCULATE OUTPUTS
        # NOTE: KEYS FOR z MATCH 'outputs' LIST ABOVE
        z = {
            'Example Output 1': 0.0,
            'Example Output 2': 0.0  
        }

        return z

    def event_state(self, t, x):
        """
        Calculate event states (i.e., measures of progress towards event (0-1, where 0 means event has occured))

        Parameters
        ----------
        t : number
            Current timestamp in seconds (≥ 0.0)
            e.g., t = 3.4
        x : dict
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        event_state : dict
            Event States, with keys defined by prognostics_model.events.
            e.g., event_state = {'EOL':0.32} given events = ['EOL']
        """

        # REPLACE BELOW WITH LOGIC TO CALCULATE EVENT STATES
        # NOTE: KEYS FOR event_x MATCH 'events' LIST ABOVE
        event_x = {
            'Example Event': 0.95
        }

        return event_x
        
    def threshold_met(self, t, x):
        """
        For each event threshold, calculate if it has been met

        Parameters
        ----------
        t : number
            Current timestamp in seconds (≥ 0.0)
            e.g., t = 3.4
        x : dict
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        thresholds_met : dict
            If each threshold has been met (bool), with deys defined by prognostics_model.events
            e.g., thresholds_met = {'EOL': False} given events = ['EOL']
        """

        # REPLACE BELOW WITH LOGIC TO CALCULATE IF THRESHOLDS ARE MET
        # NOTE: KEYS FOR t_met MATCH 'events' LIST ABOVE
        t_met = {
            'Example Event': False
        }

        return t_met