# controllers

import numpy as np

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

    
# CONTROLLERS: FULL STATE VECTOR CONTROL  (i.e., no powertrain)
# ==============================================================
class LQR():
    """ Linear Quadratic Regulator"""
    def __init__(self, n_states, n_inputs, Q=None, R=None) -> None:
        self.type = 'LQR'
        if Q is None:       Q = np.eye(n_states)
        if R is None:       R = np.eye(n_inputs)

        self.Q = Q
        self.R = R
        pass
    
    def compute_gain(self, A, B):
        """ Compute controller gain given state of the system described by linear model A, B"""
        self.K, self.E = lqr_fn(A, B, self.Q, self.R)
        return self.K, self.E

    def compute_input(self, gain, state_error):
        """ Compute system input given the controller gain 'gain' and the error w.r.t. the reference state 'error' """
        return - np.dot(gain, state_error)


class LQR_I():
    """ Linear Quadratic Regulator with Integral Effect"""
    def __init__(self, n_states, n_inputs, n_outputs, dt, int_lag=np.inf, C=None, Q=None, R=None, qi=None) -> None:
        
        self.type      = 'LQR_I'    # type of controller
        self.n_states  = n_states   # number of states
        self.n_outputs = n_outputs  # number of outputs
        self.n_inputs  = n_inputs   # number of inputs
        self.err_hist   = []         # error history (integral)
        self.int_lag   = int_lag
        self.dt        = dt
        
        # Initialize Augmented state space matrices for integral action
        self.Ai                       = np.zeros((n_states+n_outputs, n_states+n_outputs))
        self.Bi                       = np.zeros((n_states+n_outputs, n_inputs))
        self.C                        = C # store C since it's constant
        self.Ai[n_states:, :n_states] = self.C

        # Default Q, R, qi if not provided
        # --------------------------------
        if Q is None:   Q = np.eye(n_states)
        if R is None:   R = np.eye(n_inputs)
        if qi is None:  qi = np.ones((n_outputs,))
        self.Q  = Q
        self.R  = R
        self.qi = qi
        self.Qi = np.diag(np.concatenate((np.diag(self.Q), self.qi), axis=0))
        self.Ri = self.R
        pass

    def compute_gain(self, A, B):
        """ Compute controller gain given state of the system described by linear model A, B"""
        self.Ai[:self.n_states, :self.n_states] = A
        self.Bi[:self.n_states,              :] = B
        self.K, self.E = lqr_fn(self.Ai, self.Bi, self.Qi, self.Ri)
        return self.K, self.E

    def compute_input(self, gain, error):
        """ Compute system input given the controller gain 'gain' and the error w.r.t. the reference state 'error' """
        self.err_hist.append(np.dot(self.C, error))
        err_hist     = np.asarray(self.err_hist).T
        err_integral = np.sum(err_hist[:, max([0, len(self.err_hist) - self.int_lag]):], axis=1) * self.dt
        return - np.dot(gain[:, :self.n_states], error) - np.dot(gain[:, self.n_states:], err_integral)

# PD Controller (PID coming soon..) 
# =================================
class PDController():
    def __init__(self, kp=1.0, kd=1.0) -> None:
        self.kp = kp
        self.kd = kd
        pass

    def __call__(self, x_des, x, xdot_des, xdot):
        """
        Compute PD Control action
        :param x_des:       desired value
        :param x:           current value
        :param xdot_des:    desired first order derivative value
        :param xdot:        curretn first order derivative value
        """
        return self.kp * (x_des - x) + self.kd * (xdot_des - xdot)
