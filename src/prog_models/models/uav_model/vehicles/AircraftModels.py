# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Aircraft Models - originally developed by Matteo Corbetta (matteo.corbetta@nasa.gov) for SWS project
"""
import numpy as np

# from .control import controllers
from .aero import aerodynamics as aero
from . import utils
from ..vehicles import vehicles
from prog_models.exceptions import ProgModelException


# Initialize rotorcraft
# =====================
def build_model(**kwargs):

    params = dict(model='djis1000',
                  init_state_vector=None, dt=None,
                  payload=0.0, # Q=None, R=None, qi=None, i_lag=None,
                  steadystate_input=None)
    params.update(kwargs)
    
    # Generate UAV model
    # ------------------
    uav = Rotorcraft(model=params['model'], payload=params['payload'])
    
    # Build vehicle properties
    # ------------------------
    uav.build(initial_state     = params['init_state_vector'],
              steadystate_input = params['steadystate_input'],     # None assigns deault value (hover condition)
              dt                = params['dt']) # should be small enough to converge (RK4 allows larger dt)

    return uav


# CLASSES
# =========
class Rotorcraft():
    """ Lumped-mass Rotorcraft Model """
    def __init__(self, 
                 model   = 'djis1000',
                 payload = 0.0,
                 gravity = 9.81,
                 air_density=1.225,
                 **kwargs):
        self.model = model
        self.gravity = gravity
        self.state = None
        self.state_names = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'xdot', 'ydot', 'zdot', 'p', 'q', 'r']
        self.input = None
        self.input_names = ['T', 'Mx', 'My', 'Mz']
        self.dt = None
        self.propulsion = None
        self.aero = None
        self.air_density = air_density
        
        # select model
        if self.model.lower() == 'djis1000':    self.mass, self.geom, self.dynamics = vehicles.DJIS1000(payload, gravity)
        elif self.model.lower() == 'tarot18':   self.mass, self.geom, self.dynamics = vehicles.TAROT18(payload, gravity)
        else:                                   self.mass, self.geom, self.dynamics = dict(), dict(), dict()
        
        # update model based on input parameters
        self.mass.update(kwargs)
        self.mass['payload'] = payload  # Update payload separately

        self.geom.update(kwargs)
        self.dynamics.update(kwargs)

    def set_state(self, state):
        self.state = state.copy()

    def set_dt(self, dt):
        self.dt = dt
    
    def build(self, initial_state=None, steadystate_input=None, dt=0.01):

        # Initialize state and input
        if initial_state is None:       initial_state = np.zeros((self.dynamics['num_states'],))
        if steadystate_input is None:   steadystate_input = self.mass['total'] * self.gravity

        # Assign 
        self.state             = initial_state
        self.steadystate_input = steadystate_input  # typically hover condition: [weight, 0, 0, 0]
        self.input             = np.zeros((self.dynamics['num_inputs']))  # initialize input with hover condition
        self.input[0]          = steadystate_input
        
        # Introduction of Aerodynamic effects:
        self.aero = dict(drag=aero.DragModel(bodyarea=self.dynamics['aero']['ad'],
                                             Cd=self.dynamics['aero']['cd'],
                                             air_density=self.air_density),
                         lift=None)

        # Integration properties
        self.dt = dt
