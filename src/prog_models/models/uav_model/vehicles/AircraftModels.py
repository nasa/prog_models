"""
Aircraft Models
SWS Project

Matteo Corbetta
matteo.corbetta@nasa.gov
"""
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../utilities/'))

from sklearn.linear_model import GammaRegressor
import numpy as np
import sympy as sym

from prog_models.models.uav_model.vehicles.control import controllers
# from prog_models.models.uav_model.vehicles.control import dn_allocation_functions as caf
from prog_models.models.uav_model.vehicles.aero import aerodynamics as aero
# from control import controllers
# import control.allocation_functions as caf
# from aero import aerodynamics as aero

import prog_models.models.uav_model.utilities.utils as utils
import prog_models.models.uav_model.vehicles.vehicles as vehicles
# import vehicles

import prog_models.models.uav_model.utilities.geometry as geom


# Functions
# ===========
def build_rotorcraft_inertia(m, g):
    
    n_rotors   = g['num_rotors']
    m['body']  = m['body_empty'] + n_rotors * m['arm']
    m['total'] = m['body'] + m['payload']

    # Define rotor positions on the 360 degree circle
    # ------------------------------------------------
    arm_angle      = 2.0 * np.pi / n_rotors
    angular_vector = np.zeros((int(n_rotors/2),))
    for ri in range(int(n_rotors/2)):
        angular_vector[ri] = arm_angle/2.0 * (2*ri + 1)
    
    motor_distance_from_xaxis = g['arm_length'] * np.sin(angular_vector)
    if g['body_type'].lower()=='sphere': I0 = 2.0 * m['body'] * g['body_radius']**2.0 / 5.0
    else:                                raise Exception("Body geometries other than sphere have not been implemented yet.")
    m['Ixx'] = I0 + 2.0 * sum(m['arm'] * motor_distance_from_xaxis**2.0)  # [kg m^2], inertia along x
    m['Iyy'] = m['Ixx']                                                 # [kg m^2], inertia along y (symmetric uav)
    m['Izz'] = I0 + g['num_rotors'] * (g['arm_length']**2.0 * m['arm']) # [kg m^2], inertia along z
    
    return m, g

    

# def rotorcraft_earthframe_ang_moments(phidot, thetadot, psidot, p, q, r, Ixx, Iyy, Izz, b):
    """
    Return moments along x, y, z on inertial reference frame (Earth's reference frame) given
    rate of change of Euler's angles, body angular velocities, and inertia terms
    
    :param phidot:      first Euler's angle (phi) rate of change
    :param thetadot:    second Euler's angle (theta) rate of change
    :param psidot:      third Euler's angle (psi) rate of change
    :param p:           roll velocity
    :param q:           pitch velocity
    :param r:           yaw velocity
    :param Ixx:         Inertia moment around body x-axis
    :param Iyy:         Inertia moment around body y-axis
    :param Izz:         Inertia moment around body z-axis
    :param b:           Body arm's length (from center of mass to rotor's center)
    :return:            Three moments wrt inertial reference frame Mx, My, Mz
    """
#     Mx = phidot   - ((Iyy - Izz) * q * r) * 1.0/b
#     My = thetadot - ((Izz - Ixx) * p * r) * 1.0/b
#     Mz = psidot   - ((Ixx - Iyy) * p * q)
#     return Mx, My, Mz


# Symbolic functions
# =================
    """
def rotorcraft_symbolic_f(phi, theta, psi, vx, vy, vz, p, q, r, T, tp, tq, tr, m, l, Ixx, Iyy, Izz, g):

    # Calculate inertia parameters
    a1 = (Iyy - Izz) / Ixx
    a2 = (Izz - Ixx) / Iyy
    a3 = (Ixx - Iyy) / Izz

    b1 = l / Ixx
    b2 = l / Iyy
    b3 = 1 / Izz

    return sym.Matrix([vx,
                       vy,
                       vz,
                       p + q * sym.sin(phi) * sym.tan(theta) + r * sym.cos(phi) * sym.tan(theta),
                       q * sym.cos(phi) - r * sym.sin(phi),
                       q * sym.sin(phi) / sym.cos(theta) + r * sym.cos(phi) / sym.cos(theta),
                       (sym.sin(theta) * sym.cos(psi) * sym.cos(phi) + sym.sin(phi) * sym.sin(psi)) * T / m,
                       (sym.sin(theta) * sym.sin(psi) * sym.cos(phi) - sym.sin(phi) * sym.cos(psi)) * T / m,
                       - g + sym.cos(phi) * sym.cos(theta) * T / m,
                       a1 * q * r + b1 * tp,
                       a2 * p * r + b2 * tq,
                       a3 * p * q + b3 * tr,])
"""    
"""
def rotorcraft_symbolicStateMatrices():

    x, y, z, vx, vy, vz, \
        phi, theta, psi, p, q, r, \
            T, tp, tq, tr, \
                l, m, Ixx, Iyy, Izz, g = sym.symbols('x y z vx vy vz phi theta psi p q r T tp tq tr l m Ixx Iyy Izz g')

    stateEquation = rotorcraft_symbolic_f(phi, theta, psi, vx, vy, vz, p, q, r, T, tp, tq, tr, m, l, Ixx, Iyy, Izz, g)
    stateVector   = sym.Matrix([x, y, z, phi, theta, psi, vx, vy, vz, p, q, r])
    inputVector   = sym.Matrix([T, tp, tq, tr])

    jacobStateMatrix = stateEquation.jacobian(stateVector)
    jacobInputMatrix = stateEquation.jacobian(inputVector)

    return  (jacobStateMatrix, jacobInputMatrix)
"""

# Initialize rotorcraft
# =====================
def build_model(init_state_vector, dt, **kwargs):

    params = dict(name='rotorcraft-1', model='djis1000', 
                  integrator_fn='RK4', payload=2.0, Q=None, R=None, qi=None, i_lag=None,
                  aero_effects=True, steadystate_input=None)
    params.update(kwargs)
    params['i_lag'] = 100
    params['qi'] = np.array([100, 100, 300])
    params['R'] = np.diag([10, 2000, 2000, 2000])
    params['Q'] = np.diag([1000, 1000, 5000,  # x, y, z
                           100.0, 100.0, 100.0,  # phi, theta, psi,
                           1000, 1000, 5000,  # vx, vy, vz,
                           1000, 1000, 1000])  # p, q, r
    # Default control parameters (if not provided)
    if params['Q'] is None:
        params['Q'] = np.diag([1000,  1000,  5000, # x, y, z
                               100, 100, 100,      # phi, theta, psi, 
                               10000, 10000, 10000,     # vx, vy, vz,
                               10000, 10000, 10000])   # p, q, r
    if params['R'] is None:         params['R']     = np.diag([50, 7e3, 7e3, 7e3])   # T, Mu, Mv, Mw
    if params['qi'] is None:        params['qi']    = np.array([5000, 5000, 1000])  # Integral error weights in position x, y, z
    if params['i_lag'] is None:     params['i_lag'] = 500                        # integral lag: how far back to use (in data points, so 1 point = 1 dt) for integral error


    # Generate UAV model
    # ------------------
    uav = Rotorcraft(name=params['name'], model=params['model'], payload=params['payload'], aero_effects=params['aero_effects'])
    
    # Build vehicle properties
    # ------------------------
    uav.build(initial_state     = init_state_vector,
              steadystate_input = params['steadystate_input'],     # None assigns deault value (hover condition)
              integrator_fn     = params['integrator_fn'],    # Available types: Euler or RK4
              dt                = dt) # should be small enough to converge (RK4 allows larger dt)

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
                 aero_effects=False,
                 **kwargs):
        self.name = name
        self.model = model
        self.controller = None
        self.gravity = gravity
        self.state = None
        self.state_names = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'xdot', 'ydot', 'zdot', 'p', 'q', 'r']
        self.input = None
        self.input_names = ['T', 'Mx', 'My', 'Mz']
        self.integrator = None
        self.dt = None
        self.propulsion = None
        self.aero_effects=aero_effects
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
        
        # Build rotorcraft inertia properties
        self.mass, self.geom = build_rotorcraft_inertia(self.mass, self.geom)

    """    
    def reset_state(self, state0=None):
        if state0 is None:  self.state = np.zeros((self.dynamics['num_states'],))
        else:               self.state = state0.copy()
        print('Aircraft state reset complete.')
        return

    def reset_input(self, input0=None):
        if input0 is None: 
            self.input = np.zeros((self.dynamics['num_inputs']))
            self.input[0] = self.steadystate_input
        else:
            self.input = input0.copy()
        print('Aircraft input reset complete')
        return

    def reset_controller(self, state0=None, input0=None):
        
        self.reset_state(state0=state0)
        self.reset_input(input0=input0)

        # Reset K
        # ======
        phi, theta, psi = self.state[3:6]
        p,       q,   r = self.state[-3:]
        A,            B = self.linear_model(phi, theta, psi, p, q, r, self.input[0])
        self.controller.K, self.controller.E = self.controller.compute_gain(A, B)
        
        # Reset Error history (from integral action, if exist)
        # ===================================================
        if hasattr(self.controller, 'err_hist'):
            self.controller.err_hist = []    
        print("Aircraft controller reset complete")
        return
    """
    
    def build(self, initial_state=None, steadystate_input=None, integrator_fn='Euler', dt=0.01):

        # Initialize state and input
        if initial_state is None:       initial_state = np.zeros((self.dynamics['num_states'],))
        if steadystate_input is None:   steadystate_input = self.mass['total'] * self.gravity

        # Assign 
        self.state             = initial_state
        self.steadystate_input = steadystate_input  # typically hover condition: [weight, 0, 0, 0]
        self.input             = np.zeros((self.dynamics['num_inputs']))  # initialize input with hover condition
        self.input[0]          = steadystate_input
        
        # Introduction of Aerodynamic effects:
        if self.aero_effects:
            self.aero = dict(drag=aero.DragModel(bodyarea=self.dynamics['aero']['ad'],
                                                 Cd=self.dynamics['aero']['cd'],
                                                 air_density=self.air_density),
                             lift=None)

        # Integration properties
        self.dt = dt
        if   integrator_fn.lower() == 'euler':      self.int_fn = utils.euler
        elif integrator_fn.lower() == 'rk4':        self.int_fn = utils.rk4
        else:   raise Exception("Integrator function not recognized. Available options (so far) are: Euler (default) or RK4")

    """
    # Introducing control allocation matrices for rotor speed-based control
    def set_control_allocation_matrix(self, constrained_cam=False):
        Gamma, Gamma_inv, _ = caf.rotorcraft_cam(n=self.geom['num_rotors'],
                                                 l=self.geom['arm_length'],
                                                 b=self.dynamics['kt'], d=self.dynamics['kq'], constrained=constrained_cam)
        self.dynamics['Gamma'] = Gamma
        self.dynamics['Gamma_inv'] = Gamma_inv
        return

    def set_propulsion_system(self, prop_system):
        self.propulsion = prop_system
        return
    """

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
        # elif strategy.lower() == 'powertrain_based':
        else:
            raise Exception("The only two options for control_strategy are: scheduled or realtime")

        print('Controller Set correctly for ' + self.name + '.\n')

    def build_scheduled_control(self, states):
        # states = 12 x m
        # inputs = 1 (only thrust at hover is needed)
        # where: m is the number of points per state variation. Assuming only one state varies (psi, for yaw)
        # Check input
        n, m = states.shape
        assert self.controller is not None, "A controller for this vehicle has not been assigned yet."
        assert self.input is not None, "A steady-state input should be assigned before generating the controller schedule"
        assert n == self.dynamics['num_states'], "The input states dimension does not match the dimension of the vehicle state vector."
        
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

    """
    def compute_state(self, ref, params=None):
        u          = self.control_fn(self.state - ref)
        
        u[0]      += self.steadystate_input
        u[0]       = min(max([0., u[0]]), self.dynamics['max_thrust'])

        dstatedt   = self.int_fn(self.f, self.dt, self.state, u, params=params)
        self.state = self.state + dstatedt * self.dt     
        self.input = u
        return self.state
    """
    """
    def f(self, x, u, params=None):
        
        # Extract params
        # -------------
        wind = params['wind']
        wx = wind['u']
        wy = wind['v']

        # Extract values from vectors
        # --------------------------------
        m = self.mass['total']  # vehicle mass
        T, tp, tq, tr = u       # extract control input
        Ixx, Iyy, Izz = self.mass['Ixx'], self.mass['Iyy'], self.mass['Izz']    # vehicle inertia

        # Extract state variables from current state vector
        # -------------------------------------------------
        phi, theta, psi  = x[3:6]
        p, q, r          = x[-3:]
        vx_a, vy_a, vz_a = x[6:9]

        # Pre-compute Trigonometric values
        # --------------------------------
        sin_phi   = np.sin(phi)
        cos_phi   = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        tan_theta = np.tan(theta)
        sin_psi   = np.sin(psi)
        cos_psi   = np.cos(psi)
        
        # Compute drag forces
        # -------------------
        v_earth = np.dot(geom.rot_earth2body(phi, theta, psi),
                        np.array([vx_a - wx, vy_a - wy, vz_a]).reshape((-1,)))
        v_body = np.dot(geom.rot_earth2body(phi, theta, psi), v_earth)
        fb_drag = self.aero['drag'](v_body)
        fe_drag = np.dot(geom.rot_body2earth(phi, theta, psi), fb_drag)
        
        # Update state vector
        # -------------------
        dxdt     = np.zeros((len(x),))
        
        dxdt[0] = vx_a + wx   # add wind u-component to generate ground speed
        dxdt[1] = vy_a + wy   # add wind v-component to generate ground speed
        dxdt[2] = vz_a
        
        dxdt[3]  = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta
        dxdt[4]  = q * cos_phi - r * sin_phi
        dxdt[5]  = q * sin_phi / cos_theta + r * cos_phi / cos_theta
        
        dxdt[6]  = (sin_theta * cos_psi * cos_phi + sin_phi * sin_psi) * T / m - 1.0/m * fe_drag[0]
        dxdt[7]  = (sin_theta * sin_psi * cos_phi - sin_phi * cos_psi) * T / m - 1.0/m * fe_drag[1]
        dxdt[8]  = - self.gravity + cos_phi * cos_theta  * T / m - 1.0/m * fe_drag[2]

        dxdt[9]  = (Iyy - Izz) / Ixx * q * r + tp * self.geom['arm_length'] / Ixx
        dxdt[10] = (Izz - Ixx) / Iyy * p * r + tq * self.geom['arm_length'] / Iyy
        dxdt[11] = (Ixx - Iyy) / Izz * p * q + tr *        1                / Izz
        return dxdt
    """
    """
    def linear_f(self, x, u):
        # Extract state variables from state-vector
        phi, theta, psi = x[3:6]
        p,       q,   r = x[-3:]
        # Extract thrust from input vector
        thrust = u[0]
        # Compute linearized model
        A, B = self.linear_model(phi, theta, psi, p, q, r, thrust)
        return np.dot(A, x) + np.dot(B, u)
    """

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

    """
    # POWERTRAIN STUFF
    def get_omega(self, prop_num, thr, ti, dt_highfreq, input_voltage):
        x = self.propulsion[prop_num].get_next_state(ti, thr, input_voltage, dt_highfreq)
        return x[3]

    def get_all_omegas(self, thr, tf, dt_highfreq, input_voltage):
        omegas        = np.zeros((self.geom['num_rotors'],))
        t_vec         = np.arange(0.0, tf, dt_highfreq)
        for ii in range(self.geom['num_rotors']):
            for ti in t_vec:
                omega = self.get_omega(prop_num=ii, thr=thr[ii], ti=ti, dt_highfreq=dt_highfreq, input_voltage=input_voltage)
            omegas[ii] = omega    # Store values
        return omegas

    def get_U_from_propulsion(self, thr, tf, dt_highfreq=1e-6, input_voltage=23.1, omegas=None):
        # dt_highfreq   = 1e-6
        # input_voltage = 23.1
        if omegas is None:
            omegas = self.get_all_omegas(thr, tf, dt_highfreq, input_voltage)
        U = self.dynamics['Gamma'] @ omegas**2.0
        return U
    
    def get_omega_from_desired_input(self, u_des):
        return np.sqrt(self.dynamics['Gamma_inv'] @ u_des.T)

    @staticmethod
    def get_throttle_from_omega_des(omega_des):
        return caf.omega2throttle(omega_des)
    """
