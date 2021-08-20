import prog_models
from math import inf

class IndividualConstLoad(prog_models.PrognosticsModel):
    inputs = ["is_on"]
    outputs = ["current"]
    states = ["load", "is_on"]

    default_parameters = {
        "xrange": {}
    }
    
    def __init__(self, load, **kwargs):
        kwargs['x0'] = {
            'load': load,
            'is_on': False
        }

        # Run constructor for PrognosticsModel
        super().__init__(**kwargs)

        # Link state_limits to the parameters 
        # Note: Must be after PrognosticsModel constructor
        self.state_limits = self.parameters["xrange"]

    def __str__(self):
        return "Constant Load: {}".format(self.parameters['x0']['load'])

    def initialize(self, u={}, z={}):
        return self.parameters["x0"]
    
    def next_state(self, x, u, dt):
        return {
            'load': x['load'],
            'is_on': u['is_on']
        }

    def output(self, x):
        return {'current': x['load'] * x['is_on']}
