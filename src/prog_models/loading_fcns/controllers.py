"""
Controllers
"""

# IMPORTS
# ========
# from utilities.imports_ import np
import numpy as np


# LQR general function
# ====================
def lqr_fn(A, B, Q, R):
    """
    Calculate control matrix K using LQR

    The function estimates the gain matrix associated with a linear quadratic regulator(LQR) via Hamiltonian matrix.
    The function assumes a linearized system:

    dx / dt = Ax + Bu
    y = Cx + Du

    The closed loop uses - K * x as input dx / dt = (A - B * K) * x

    :param A:       Nx x Nx state matrix, linearized w.r.t. current attitude and thrust
    :param B:       Nu x Nx input matrix, linearized w.r.t. current attitude
    :param Q:       Nx x Nx state cost matrix, higher the values of the (i,i)-th element is the more aggresive the controller is w.r.t. that variable
    :param R:       Nx x Nu input cost matrix, higher values imply higher cost of producing that input (used to limit the amount of power)
    :return K:      Nx x Nu, control gain matrix
    :return E:      Nx x 1, eigenvalues of the matrix A - B * K
    """
    n     = A.shape[0]
    R_inv = np.linalg.inv(R)

    # --------- Generate hamiltonian matrix ------------ #
    HM1 = A
    HM2 = - np.dot( np.dot( B, R_inv), B.T )
    HM3 = - Q
    HM4 = - A.T

    HM12 = np.concatenate((HM1, HM2), axis=1)
    HM34 = np.concatenate((HM3, HM4), axis=1)
    HM   = np.concatenate((HM12, HM34), axis=0)

    # --------- Extract eigevectors whose eigenvalues have real part < 0 ------------ #
    eig_val, eig_vec = np.linalg.eig(HM)
    e = len(eig_val)
    V_ = np.zeros((e, 1))
    for ii in range(e):
        if np.real(eig_val[ii]) < 0:
            V_ = np.concatenate((V_, eig_vec[:, ii].reshape((e,1))), axis=1)
    V_ = V_[:, 1:]

    # --------- Partition matrix of resulting eigenvectors ------------ #
    X = V_[:n, :n] # first half (row-wise)
    Y = V_[n:, :n] # second half (row-wise) 

    # ----------- Estimate control gain ----------------- #
    K = np.dot( np.dot( np.dot(R_inv, B.T), Y ), np.linalg.inv(X) )
    K = np.real(K) # some spuriorus imaginary parts (1e-13) sometimes remain in the matrix. we manually remove them

    # Calculate the eigenvalues of the matrix A - B*K
    E, _ = np.linalg.eig( A - np.dot(B, K) )
    return K, E

    
# # CONTROLLERS: FULL STATE VECTOR CONTROL  (i.e., no powertrain)
# # ==============================================================
# class LQR():
#     """ Linear Quadratic Regulator"""
#     def __init__(self, n_states, n_inputs, Q=None, R=None) -> None:
#         self.type = 'LQR'
#         if Q is None:       Q = np.eye(n_states)
#         if R is None:       R = np.eye(n_inputs)

#         self.Q = Q
#         self.R = R
        
    
#     def compute_gain(self, A, B):
#         """ Compute controller gain given state of the system described by linear model A, B"""
#         self.K, self.E = lqr_fn(A, B, self.Q, self.R)
#         return self.K, self.E

#     def compute_input(self, gain, state_error):
#         """ Compute system input given the controller gain 'gain' and the error w.r.t. the reference state 'error' """
#         return - np.dot(gain, state_error)
    
#     def build_scheduled_control(self, system_linear_model_fun, input_vector, state_vector_vals=None, index_scheduled_var=None):

#         if state_vector_vals is None:
#             # using psi (yaw angle) as scheduled variable as the LQR control cannot work with yaw=0 since it's in the inertial frame.
#             n_schedule_grid = 360*2 + 1
#             index_scheduled_var = 5
#             state_vector_vals = np.zeros((self.n_states, n_schedule_grid))
#             state_vector_vals[index_scheduled_var, :] = np.linspace(-2.0*np.pi, 2.0*np.pi, n_schedule_grid)

#         n, m = state_vector_vals.shape
#         assert n == self.n_states, "number of states set at initialization and size of state_vector_vals mismatch."
#         self.control_gains = np.zeros((self.n_inputs, self.n_states, m))
        
#         for j in range(m):
#             phi, theta, psi = state_vector_vals[3:6, j]
#             p,       q,   r = state_vector_vals[-3:, j]
#             Aj,          Bj = system_linear_model_fun(phi, theta, psi, p, q, r, input_vector[0])
#             self.control_gains[:, :, j], _ = self.compute_gain(Aj, Bj)

#         print('Control gain matrices complete.')

##### KJ TEST:
class LQR_I():
    """ Linear Quadratic Regulator with Integral Effect"""

    def __init__(self, x_ref, vehicle, **kwargs):

        self.type      = 'LQR_I'                # type of controller
        self.states    = vehicle.states         # state variables of the system to be controlled (x, y, z, phi, theta, psi)
        self.n_states  = len(self.states) - 1       # number of states
        self.outputs   = vehicle.outputs[:3]    # output variables of the system to be controlled (x, y, z only)
        self.n_outputs = 3                      # number of outputs
        self.inputs    = vehicle.inputs         # input variables of the system to be controlled ()
        self.n_inputs  = len(self.inputs)       # number of inputs
        self.err_hist  = []                     # error history (integral)
        self.ref_traj  = x_ref                  # reference state to follow during simulation (x_ref, y_ref, z_ref, phi_ref, theta_ref, psi_ref, ...)

        # Default control parameters
        # --------------------------------
        self.parameters = dict(int_lag=100,             # error integral length: how far back in time to compute integral error
                               Q=np.eye(self.n_states), # state error penalty matrix: how 'bad' is an error in the state vector w.r.t. the reference state vector
                               R=np.eye(self.n_inputs), # input penalty matrix: how 'hard' it is to produce the desired input (thrust and three moments along three axes)
                               qi=np.ones((self.n_outputs,)),   # integral error penalty: how much the error history is contributing to the overall error
                               scheduled_var='psi',     # variable used to create the scheduled controller gains (only psi allowed for now)
                               index_scheduled_var=5)   # index corresponding to the scheduled_var (psi) in the state vector x; i.e., x[5] = psi
        self.parameters.update(kwargs)                  # update control parameters according to user

        # Get scheduled variable index (only necessary if scheduled_var is changed, which is not happening at the moment)
        self.parameters['index_scheduled_var'] = self.states.index(self.parameters['scheduled_var'])

        # Initialize other controller-related variables
        # ---------------------------------------------
        self.dt        = vehicle.parameters['dt']
        self.C         = np.zeros((self.n_outputs, self.n_states))
        self.C[:self.n_outputs, :self.n_outputs] = np.eye(self.n_outputs)

        # Initialize Augmented state space matrices for integral action
        self.Ai                       = np.zeros((self.n_states+self.n_outputs, self.n_states+self.n_outputs))
        self.Bi                       = np.zeros((self.n_states+self.n_outputs, self.n_inputs))
        self.Ai[self.n_states:, :self.n_states] = self.C

        # Generate augmented Q and R for integral term
        self.parameters['Qi'] = np.diag(np.concatenate((np.diag(self.parameters['Q']), self.parameters['qi']), axis=0))
        self.parameters['Ri'] = self.parameters['R']

    
    def __call__(self, t, x=None):
        #  
        #   future_loading_fnc = LQR_I(x_ref)
        #   u = future_loading_fnc(t, x)
        #   vehicle.simulate_to(future_loading_fnc, ...)
        if x is None:
            x_k = np.zeros((self.n_states, 1))
        else:
            x_k = np.array([x.matrix[ii][0] for ii in range(len(x.matrix)-1)])
        
        # Identify reference state (desired state) at t
        t_k = np.round(t + self.dt/2.0, 1)  # current time step
        time_ind = np.argmin(np.abs(t_k - self.ref_traj['t'].tolist())) # get index of time value in ref_traj closest to t_k
        x_ref_k = []
        for state in self.states:
            x_ref_k.append(self.ref_traj[state][time_ind])
        x_ref_k = np.asarray(x_ref_k[:-1]) # get rid of time index in state vector
        x_k = x_k.reshape(x_k.shape[0],)

        error         = x_k - x_ref_k    # Error between current and reference state
        scheduled_var = x_k[self.parameters['index_scheduled_var']]    # get psi from current state vector (self.parameters = 'psi')
        k_idx         = np.argmin(np.abs(self.scheduled_states[self.parameters['index_scheduled_var'], :] - scheduled_var)) # find the psi value stored in the controller closest to the current psi --> extract index
        K             = self.control_gains[:, :, k_idx]     # extract gain corresponding to the current psi value
        u             = self.compute_input(K, error)                 # compute input u given the gain matrix K and the error between current and reference state
        return {'T': u[0], 'mx': u[1], 'my': u[2], 'mz': u[3]}

    def compute_gain(self, A, B):
        """ Compute controller gain given state of the system described by linear model A, B"""
        self.Ai[:self.n_states, :self.n_states] = A
        self.Bi[:self.n_states,              :] = B
        self.K, self.E = lqr_fn(self.Ai, self.Bi, self.parameters['Qi'], self.parameters['Ri'])
        return self.K, self.E

    def compute_input(self, gain, error):
        """ Compute system input given the controller gain 'gain' and the error w.r.t. the reference state 'error' """
        self.err_hist.append(np.dot(self.C, error))
        err_hist     = np.asarray(self.err_hist).T
        err_integral = np.sum(err_hist[:, max([0, len(self.err_hist) - self.parameters['int_lag']]):], axis=1) * self.dt
        return - np.dot(gain[:, :self.n_states], error) - np.dot(gain[:, self.n_states:], err_integral)

    def build_scheduled_control(self, system_linear_model_fun, input_vector, state_vector_vals=None, index_scheduled_var=None):
        
        # controller.build_scheduled_control(vehicle.linear_model, input_vector=thrust_at_hover=mass_total*gravity)

        if state_vector_vals is None:
            # using psi (yaw angle) as scheduled variable as the LQR control cannot work with yaw=0 since it's in the inertial frame.
            n_schedule_grid = 360*2 + 1
            index_scheduled_var = 5
            state_vector_vals = np.zeros((self.n_states, n_schedule_grid))
            state_vector_vals[index_scheduled_var, :] = np.linspace(-2.0*np.pi, 2.0*np.pi, n_schedule_grid)

        n, m = state_vector_vals.shape
        assert n == self.n_states, "number of states set at initialization and size of state_vector_vals mismatch."
        self.control_gains = np.zeros((self.n_inputs, self.n_states + self.n_outputs, m))
        self.scheduled_states = state_vector_vals # ASK MATTEO if this is right; originally 'states' was passed in here 

        for j in range(m):
            phi, theta, psi = state_vector_vals[3:6, j]
            p,       q,   r = state_vector_vals[-3:, j]
            Aj,          Bj = system_linear_model_fun(phi, theta, psi, p, q, r, input_vector[0])
            self.control_gains[:, :, j], _ = self.compute_gain(Aj, Bj)

        print('Control gain matrices complete.')




        

# # PD Controller (PID coming soon..) 
# # =================================
# class PDController():
#     def __init__(self, kp=1.0, kd=1.0) -> None:
#         self.kp = kp
#         self.kd = kd

#     def __call__(self, x_des, x, xdot_des, xdot):
#         """
#         Compute PD Control action
#         :param x_des:       desired value
#         :param x:           current value
#         :param xdot_des:    desired first order derivative value
#         :param xdot:        curretn first order derivative value
#         """
#         return self.kp * (x_des - x) + self.kd * (xdot_des - xdot)


