"""
NURBS curve functions
"""

import numpy as np
import scipy as sp
import scipy.interpolate as interp

# New NURBS algorithm
# ===================




## NURBS Functions
# ================

def normalize_t_1(t, a, b):
    return (t-a) / (b - a)

def normalize_t_2(t, a, b):
    return (b-t) / (b-a)

def basisfunction(order, u, t):
    """
    :param order:
    :param u (evalAtPoints):
    :param t (knotVector):
    :return N:
    """
    n_points = len(t)
    N        = np.zeros((n_points,))
    # N[(u > (t + 1.0e-6)) * np.insert((u<=(t[1:]+1e-6)), -1, True)] = 1
    N[(u > t) * np.insert((u <= t[1:]), -1, True)] = 1
    if N.any() != 0:
        for n in range(1, order):
            t_pn  = np.concatenate((t[n:], np.ones((n,))))
            d     = normalize_t_1(u, t, t_pn) * N
            t_pn1 = np.concatenate((t[n+1:], np.ones((n+1,))))
            t_p1  = np.concatenate((t[1:], np.ones((1,))))
            e     = normalize_t_2(u, t_p1, t_pn1) * np.insert(N[1:], -1, 0) 
            N     = np.nan_to_num(e) + np.nan_to_num(d)
            
    return N[:n_points - order]


def evaluate(order, t, weights, waypoint_vector, basis_length=1000):
    """
    :param order:
    :param t (knot vector):
    :param weights:
    :param waypointVector:
    :return:
    """
    # Check validity of waypoint vector
    n, m = waypoint_vector.shape
    if n != 2:             waypoint_vector = waypoint_vector.T
    if waypoint_vector.shape[0] != 2:     raise Exception('The waypoint vector provided is not valid. It should be a 2 x m array where m is the number of waypoints, \
                                             the first row is the ETA vector and the second row is the actual waypoint vector.')
    # U = np.linspace(t[order-1]+1.0e-6, t[-1]-1.0e-6, int(round(abs(t[-1]-t[order-1])*basis_length)))
    U = np.linspace(t[order-1], t[-1], int(round(abs(t[-1]-t[order-1])*basis_length)))
    # nU = len(U)
    q = len(U)
    # Calculate basis functions
    # -----------------------------------------
    N = np.zeros((q, waypoint_vector.shape[1]))
    # for ii in range(q):    N[ii, :] = basisfunction(order, U[ii], t)   # first piece of basis function
    for ii in range(q):
        N[ii, :] = basisfunction(order, U[ii], t)   # first piece of basis function
    if N[0, 0] == 0:    # Numerical issue: the first point of the first basis is not computed but remain 0 when it should be 1. This fixes it.
        N[0, 0] = 1.0

    S = np.dot(N, weights)        # Calculate denominator of rational basis function
    R = N * np.repeat(weights.reshape((1, -1)), repeats=N.shape[0], axis=0)       # Calculate rational basis function
    with np.errstate(divide="ignore"):  # ignore warning divide by zero
        R /= np.repeat(S.reshape((-1,1)), repeats=m, axis=1)
    R[np.isnan(R)] = 0.

    # Generate points of the planned path
    # ----------------------------------
    C = np.zeros((2, q))
    C[0, :] = np.dot(R, waypoint_vector[0, :].reshape((-1,)))
    C[1, :] = np.dot(R, waypoint_vector[1, :].reshape((-1,)))
    return C


def generate_intermediate_points(px, py, pz, yaw, eta, weight_vector=None):
    """
    The function takes the original set of curve points and generate a new set containing fictitious, intermediate points.
    The fictitious points help generating a NURBS trajectory that pass through the desired (true) points according
    to the weight vector.

    The fictitious points are particularly useful for rotary-wing UAV trajectories, since the latter have very small minimum-radius
    turns and can do sharp turns. Without such fictitious points, the trajectory will look more like a fixed-wing trajectory, where the way-points
    are approached, but they are not reached.

    :param px:             (n,) array, x-coordinate of points, [m]
    :param py:             (n,) array, y-coordinate of points, [m]
    :param pz:             (n,) array, z-coordinate of points, [m]
    :param yaw:            (n,) array, yaw angle in between waypoints (ini yaw=0), [rad]
    :param eta:            (n,) array, estimated time of arrival at each waypoint, [s]
    :param weight_vector:  (n,) array, weights assigned to each waypoint, [-] (default=None)
    :return:
    """

    # ------- Weight Vector -------- #
    # If no weight vector is provided, then a default uniform weight vector with value = 2 is assigned to the waypoints.
    # do not assign 1, since 1 is the default weight value of the intermediate (fictitious) waypoints.
    if weight_vector is None:    weight_vector = 2 * np.ones((len(px),))

    # ------- New array of waypoints ------- #
    ps = np.array([px, py, pz]).T                   # collect waypoints into a matrix
    fictitiousWPs = np.zeros((px.shape[0] - 1, 3))     # initialize array of fictitious waypoints
    for ii in range(1, ps.shape[0]):
        fictitiousWPs[ii - 1, :] = (ps[ii, :] - ps[ii - 1, :]) / 2.0 + ps[ii - 1, :] # generate a fictitious waypoint as average of true waypoints

    # Generate the new waypoint matrix
    ps_new = np.zeros((ps.shape[0] + fictitiousWPs.shape[0], ps.shape[1]))
    tmpcount1, tmpcount2 = 0, 0
    for ii in range(ps_new.shape[0]):
        if ii == 0 or ii % 2 == 0:
            ps_new[ii, :] = ps[tmpcount1, :]
            tmpcount1 += 1
        else:
            ps_new[ii, :] = fictitiousWPs[tmpcount2, :]
            tmpcount2 += 1

    # -------- New Weight Vector ------------- #
    weight_vector_new = np.zeros((ps_new.shape[0],))            # initialize weight vector
    counter_1 = 0  # add a temporary counter for the true weight vector
    for jj in range(len(weight_vector_new)):
        if jj % 2 == 0:
            weight_vector_new[jj] = weight_vector[counter_1]      # Assign pre-defined weight to the waypoint
            counter_1 += 1
        else:
            weight_vector_new[jj] = 1                            # Assign standard weight of 1 to the fititious waypoint

    # -------- Generate new ETA vector ------------- #
    fictitiousETA = np.zeros((weight_vector.shape[0] - 1,))      # initialize fictitious ETA vector
    for ii in range(1, len(fictitiousETA) + 1):
        fictitiousETA[ii - 1] = (eta[ii] - eta[ii - 1]) / 2.0 + eta[ii - 1] # generate fictitious ETA as average of real ETAs

    # Generate new ETA vector
    eta_new = np.zeros((weight_vector_new.shape[0],))
    counter_1, counter_2 = 0, 0
    for ii in range(len(eta_new)):
        if ii == 0 or ii % 2 == 0:
            eta_new[ii] = eta[counter_1]                # Assign existing ETA
            counter_1 += 1
        else:
            eta_new[ii] = fictitiousETA[counter_2]      # Assign fictitious ETA
            counter_2 += 1

    # -------- Generate new Yaw vector ---------- #
    # The fictitious yaw is NOT the average of the true yaw, but it is equal
    # to the yaw at the previous (true) waypoint.
    yaw_new = np.zeros((ps_new.shape[0],))            # Initialize new yaw vector
    tmpcount1 = 1
    for jj in range(len(yaw)):
        yaw_new[counter_1:counter_1 + 2] = yaw[jj]    # Assign new yaw value
        counter_1 += 2                                # this counter is

    # Extract waypoints along x, y, z from the new waypoint matrix
    px_new = ps_new[:, 0]
    py_new = ps_new[:, 1]
    pz_new = ps_new[:, 2]

    return px_new, py_new, pz_new, yaw_new, eta_new, weight_vector_new


# def generate_3dnurbs(wpx, wpy, wpz, wyaw, eta, delta_t, nOrder, weightVector=None, basis_length=1000):
def generate_3dnurbs(wpx, wpy, wpz, eta, delta_t, order, weight_vector, basis_length=1000):
    """

    :param wpx:
    :param wpy:
    :param wpz:
    :param wyaw:
    :param eta:
    :param delta_t:
    :param nOrder:
    :param weightVector:
    :return:
    """
    """
    print('Generating position profile (NURBS)', end=" ")
    # --------- Add fictitious waypoints to ensure the trajectory will pass through the true waypoints --------- #
    wpx, wpy, wpz, wyaw, eta, weightVector = generate_intermediate_points(wpx, wpy, wpz, wyaw, eta, weightVector)

    # --------- get some useful dimensions ----------- #
    n = len(wpx)-1  # define number of waypoints-1
    k = nOrder

    # -------- Define knot vector ----------- #
    t            = np.zeros((n+k+1,))
    idx_vec      = np.arange(1, n+k+2)
    t[idx_vec<k] = 0.0
    msk1         = (idx_vec>=k) * (idx_vec<=n)
    t[msk1]      = (idx_vec[msk1]-1) - k + 1
    t[idx_vec>n] = n - k + 2
    t            = (t - min(t)) / (max(t) - min(t))

    # -------- Create trajectory position curve using NURBS -------- #
    pVector = np.zeros((2, len(wpx)))
    pVector[0, :] = eta
    pVector[1, :] = wpx
    traj_x = evaluate(k, t, weightVector, pVector, basis_length=basis_length)
    
    pVector[1, :] = wpy
    traj_y = evaluate(k, t, weightVector, pVector, basis_length=basis_length)
    
    pVector[1, :] = wpz
    traj_z = evaluate(k, t, weightVector, pVector, basis_length=basis_length)

    # ------- Create trajectory time vector ----------- #
    time = np.arange(eta[0], eta[-1]+delta_t, delta_t)

    # --------- Interpolate trajectory ---------- #
    traj_x[:, 0] = np.array([0.0, wpx[0]])
    traj_y[:, 0] = np.array([0.0, wpy[0]])
    traj_z[:, 0] = np.array([0.0, wpz[0]])

    px = interp.interp1d(traj_x[0, :], traj_x[1, :], kind='cubic', fill_value='extrapolate')(time)
    py = interp.interp1d(traj_y[0, :], traj_y[1, :], kind='cubic', fill_value='extrapolate')(time)
    pz = interp.interp1d(traj_z[0, :], traj_z[1, :], kind='cubic', fill_value='extrapolate')(time)
    print('complete.')
    return {'px': px, 'py':py, 'pz': pz, 'time': time, 'wyaw': wyaw, 'weta': eta}
    """


    """
    Generate NURBS curve defining the trajectory along x, y and z-axis using the waypoints wpx, wpy, and wpz.
    The additional information necessary to generate the curves are:
    ETA at each waypoint, delta_t to generate the time vector, order of the NURBS curve, weight_vector to assign
    weights to the way-points, and basis_length used to define the resolution of the NURBS basis functions.

    :param wpx:             n x 1, way-points along local x-axis
    :param wpy:             n x 1, way-points along local y-axis
    :param wpz:             n x 1, way-points along local z-axis
    :param eta:             n x 1, eta for each way-point
    :param delta_t:         scalar, dt used to define time vector
    :param order:           scalar, int, order of the NURBS curve
    :param weight_vector:   n x 1, weights to be assigned to each way-point when generating the curve
    :param basis_length:    length of the basis function, number of points used to define independent variable u. Default is 1000
    :return:                dictionary with the 3D position profile of the NURBS curve: {'px', 'py', 'pz', 'time'}
    """
    print('Generating position profile (NURBS)', end=" ")
    # --------- get some useful dimensions ----------- #
    n = len(wpx)-1              # Define number of waypoints-1
    k = order                   # Define order of NURBS
    t = knot_vector(n, k)       # Define knot vector

    # -------- Create trajectory position curve using NURBS -------- #
    p_vector = np.zeros((2, len(wpx)))
    p_vector[0, :] = eta
    p_vector[1, :] = wpx

    traj_x = evaluate(k, t, weight_vector, p_vector, basis_length=basis_length)
    
    p_vector[1, :] = wpy
    traj_y = evaluate(k, t, weight_vector, p_vector, basis_length=basis_length)
    
    p_vector[1, :] = wpz
    traj_z = evaluate(k, t, weight_vector, p_vector, basis_length=basis_length)

    # ------- Create trajectory time vector ----------- #
    time = np.arange(eta[0], eta[-1]+delta_t, delta_t)

    # --------- Interpolate trajectory ---------- #
    traj_x[:, 0] = np.array([0.0, wpx[0]])
    traj_y[:, 0] = np.array([0.0, wpy[0]])
    traj_z[:, 0] = np.array([0.0, wpz[0]])

    px = interp.interp1d(traj_x[0, :], traj_x[1, :], kind='cubic', fill_value='extrapolate')(time)
    py = interp.interp1d(traj_y[0, :], traj_y[1, :], kind='cubic', fill_value='extrapolate')(time)
    pz = interp.interp1d(traj_z[0, :], traj_z[1, :], kind='cubic', fill_value='extrapolate')(time)

    print('complete.')
    return {'px': px, 'py': py, 'pz': pz, 'time': time}

def knot_vector(n, k):
    """
    Generate knot vector of k-th order for a trajectory with n-points

    :param n:       number of points in the trajectory
    :param k:       order of the NURBS
    :return:        normalized knot vector, [0,1]
    """
    t              = np.zeros((n + k + 1,))             # initialize vector
    idx_vec        = np.arange(1, n + k + 2)            # generate indexes of the vector
    # t[idx_vec < k] = 0.0                              # set all values in t below index k equal to 0
    msk1           = (idx_vec >= k) * (idx_vec <= n)    # create a mask for values between k and n
    t[msk1]        = (idx_vec[msk1] - 1) - k + 1        # assign values to the masked elements
    t[idx_vec > n] = n - k + 2                          # assign fixed value n-k+2 for everything above index n
    t              = (t - min(t)) / (max(t) - min(t))   # normalize vector
    return t


if __name__ == '__main__':

    print('NURBS functions')
