# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Aircraft Models - originally developed by Matteo Corbetta (matteo.corbetta@nasa.gov) for SWS project
"""
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../utilities/'))

import numpy as np

from prog_models.models.uav_model.vehicles.control import controllers
from prog_models.models.uav_model.vehicles.aero import aerodynamics as aero
import prog_models.models.uav_model.utilities.utils as utils
import prog_models.models.uav_model.vehicles.vehicles as vehicles
from prog_models.exceptions import ProgModelException


# Initialize rotorcraft
# =====================
def build_model(**kwargs):

    params = dict(name='rotorcraft-1', model='djis1000',
                  init_state_vector=None, dt=None,
                  payload=0.0, Q=None, R=None, qi=None, i_lag=None,
                  steadystate_input=None)
    params.update(kwargs)
    
    # Default control parameters (if not provided)
    if params['Q'] is None:
        params['Q'] = np.diag([1000, 1000, 25000,  # x, y, z
                               100.0, 100.0, 100.0,  # phi, theta, psi,
                               1000, 1000, 5000,  # vx, vy, vz,
                               1000, 1000, 1000])  # p, q, r
    if params['R'] is None:         params['R']     = np.diag([500, 4000, 4000, 4000])   # T, Mu, Mv, Mw
    if params['qi'] is None:        params['qi']    = np.array([100, 100, 1000])  # Integral error weights in position x, y, z
    if params['i_lag'] is None:     params['i_lag'] = 100                        # integral lag: how far back to use (in data points, so 1 point = 1 dt) for integral error


    # Generate UAV model
    # ------------------
    uav = Rotorcraft(name=params['name'], model=params['model'], payload=params['payload'])
    
    # Build vehicle properties
    # ------------------------
    uav.build(initial_state     = params['init_state_vector'],
              steadystate_input = params['steadystate_input'],     # None assigns deault value (hover condition)
              dt                = params['dt']) # should be small enough to converge (RK4 allows larger dt)


    # Define controller and control strategy
    # ---------------------------------------
    # States for scheduled controller
    # use -2pi / 2pi because the rotation of the UAV may happen in either direction
    # to ensure that the minimum rotation necessary is achieved (weird thing). using 720 +1 points to ensure that yaw = 0 has a matrix.
    scheduled_states       = np.zeros((uav.dynamics['num_states'], 360*2+1))
    scheduled_states[5, :] = np.linspace(start=-2.0*np.pi, stop=2.0*np.pi, num=360*2+1)
    
    # Set controller
    uav.set_controller(type_            = 'LQR_I',          # LQR (normal lqr control) or LQR_I (lqr with integral action)
                       strategy         = 'scheduled',      # Compute control gain in real time or pre-compute and schedule it
                       scheduled_states = scheduled_states, # Scheduled states to pre-compute control gains; valid only if control_strategy=='scheduled'
                       scheduled_var    = 'psi',            # Variable that changes in scheduled-control; valid only if control_strategy=='scheduled'
                       Q                = params['Q'],                # Control weights and integral lag
                       R                = params['R'],             
                       qi               = params['qi'],
                       int_lag          = params['i_lag'])    
    return uav


# CLASSES
# =========
class Rotorcraft():
    """ Lumped-mass Rotorcraft Model """
    def __init__(self, 
                 name    = 'rotorcraft-1', 
                 model   = 'djis1000',
                 payload = 0.0,
                 gravity = 9.81,
                 air_density=1.225,
                 **kwargs):
        self.name = name
        self.model = model
        self.controller = None
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
        
        pass

    def set_state(self, state):
        self.state = state.copy()
        pass

    def set_dt(self, dt):
        self.dt = dt
        self.controller.dt = dt
        pass
    
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
        pass

    def set_controller(self, type_='LQR', strategy='realtime', scheduled_states=None, scheduled_var='psi', Q=None, R=None, qi=None, int_lag=np.inf):

        type_ = type_.lower().replace("_","").replace("-","").replace(" ","")

        # Set default controller parameters
        # -------------------------------
        if Q is None:   Q  = np.ones((self.dynamics['num_states'], self.dynamics['num_states']))
        if R is None:   R  = np.ones((self.dynamics['num_inputs'], self.dynamics['num_inputs']))
        
        # Set Controller
        # --------------
        if type_ == 'lqri':  # LQR with integral action
            if qi is None:  
                qi = np.ones((self.dynamics['num_outputs'],))  # default integral gain
            self.controller = controllers.LQR_I(n_states=self.dynamics['num_states'],
                                            n_inputs=self.dynamics['num_inputs'],
                                            n_outputs=self.dynamics['num_outputs'], 
                                            dt=self.dt,
                                            int_lag=int_lag,
                                            C=self.dynamics['C'],
                                            Q=Q, R=R, qi=qi)
        elif type_ == 'lqr':    # traditional LQR                
            self.controller = controllers.LQR(n_states=self.dynamics['num_states'],
                                          n_inputs=self.dynamics['num_inputs'],
                                          Q=Q, R=R)
        else:   # everything else 
            raise Exception(" Controller type " + type_ + " is not implemented (yet).")

        # Set control strategy
        # ---------------------
        if strategy.lower() == 'scheduled':    
            self.control_gain_states = scheduled_states
            self.scheduled_var_idx   = self.dynamics['state_vars'].index(scheduled_var)
            self.control_gains       = None   # initialize control gain variable
            self.build_scheduled_control(scheduled_states)   
            self.control_fn = self.control_scheduled
        elif strategy.lower() == 'realtime':    
            self.control_fn = self.control_realtime
        else:
            raise Exception("The only two options for control_strategy are: scheduled or realtime")

        print('Controller Set correctly for ' + self.name + '.\n')

    def build_scheduled_control(self, states):
        # states = 12 x m
        # inputs = 1 (only thrust at hover is needed)
        # where: m is the number of points per state variation. Assuming only one state varies (psi, for yaw)
        # Check input
        n, m = states.shape
        if self.controller is None: 
            raise ProgModelException("A controller for this vehicle has not been assigned yet.")
        if self.input is None: 
            raise ProgModelException("A steady-state input should be assigned before generating the controller schedule.")
        if n != self.dynamics['num_states']:
            raise ProgModelException("The input states dimension does not match the dimension of the vehicle state vector.")
        
        self.scheduled_states = states  # store states linked to control scheduling
        control_gain_size     = [self.dynamics['num_inputs'], self.dynamics['num_states'], m]   # dimensions of the control gain scheduler
        # if controller is LQR with integral part, modify dimensions accordingly
        if self.controller.type == 'LQR_I':     control_gain_size[1] = self.dynamics['num_states'] + self.dynamics['num_outputs']
        
        # Define control schedule
        self.control_gains = np.zeros(control_gain_size)
        wb = utils.ProgressBar(n=m, prefix='Building controller schedule', suffix=' complete', print_length=70)
        for jj in range(m):
            phi, theta, psi = states[3:6, jj]
            p,       q,   r = states[-3:, jj]
            A,            B = self.linear_model(phi, theta, psi, p, q, r, self.input[0])
            K,           _  = self.controller.compute_gain(A, B)
            self.control_gains[:, :, jj] = K
            wb(jj)
        wb(m)
        pass

    def control_realtime(self, error):
        phi, theta, psi = self.state[3:6]
        p,       q,   r = self.state[-3:]
        A,            B = self.linear_model(phi, theta, psi, p, q, r, self.input[0])
        K,            _ = self.controller.compute_gain(A, B)
        return self.controller.compute_input(K, error)

    def control_scheduled(self, error):
        psi   = self.state[self.scheduled_var_idx]
        k_idx = np.argmin(np.abs(self.scheduled_states[self.scheduled_var_idx, :] - psi))
        K     = self.control_gains[:, :, k_idx]
        return self.controller.compute_input(K, error)

    def linear_model(self, phi, theta, psi, p, q, r, T):
        m         = self.mass['total']
        Ixx       = self.mass['Ixx']
        Iyy       = self.mass['Iyy']
        Izz       = self.mass['Izz']
        l         = self.geom['arm_length']
        sin_phi   = np.sin(phi)
        cos_phi   = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        tan_theta = np.tan(theta)
        sin_psi   = np.sin(psi)
        cos_psi   = np.cos(psi)

        A = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
                      [0, 0, 0, q*cos_phi*tan_theta - r*sin_phi*tan_theta, q*(tan_theta**2 + 1)*sin_phi + r*(tan_theta**2 + 1)*cos_phi, 0, 0, 0, 0, 1, sin_phi*tan_theta, cos_phi*tan_theta], 
                      [0, 0, 0, -q*sin_phi - r*cos_phi, 0, 0, 0, 0, 0, 0, cos_phi, -sin_phi], 
                      [0, 0, 0, q*cos_phi/cos_theta - r*sin_phi/cos_theta, q*sin_phi*sin_theta/cos_theta**2 + r*sin_theta*cos_phi/cos_theta**2, 0, 0, 0, 0, 0, sin_phi/cos_theta, cos_phi/cos_theta], 
                      [0, 0, 0, T*(-sin_phi*sin_theta*cos_psi + sin_psi*cos_phi)/m, T*cos_phi*cos_psi*cos_theta/m, T*(sin_phi*cos_psi - sin_psi*sin_theta*cos_phi)/m, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 0, T*(-sin_phi*sin_psi*sin_theta - cos_phi*cos_psi)/m, T*sin_psi*cos_phi*cos_theta/m, T*(sin_phi*sin_psi + sin_theta*cos_phi*cos_psi)/m, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 0, -T*sin_phi*cos_theta/m, -T*sin_theta*cos_phi/m, 0, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r*(Iyy - Izz)/Ixx, q*(Iyy - Izz)/Ixx], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, r*(-Ixx + Izz)/Iyy, 0, p*(-Ixx + Izz)/Iyy], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, q*(Ixx - Iyy)/Izz, p*(Ixx - Iyy)/Izz, 0]])

        B = np.array([ [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [(sin_phi*sin_psi + sin_theta*cos_phi*cos_psi)/m, 0, 0, 0], 
                       [(-sin_phi*cos_psi + sin_psi*sin_theta*cos_phi)/m, 0, 0, 0], 
                       [cos_phi*cos_theta/m, 0, 0, 0], 
                       [0, l/Ixx, 0, 0], 
                       [0, 0, l/Iyy, 0], 
                       [0, 0, 0, 1.0/Izz]])

        return A, B
