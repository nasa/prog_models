"""
Geometric functions
Matteo Corbetta
"""

# from abc import abstractmethod
# import scipy.spatial as spatial
# import scipy.interpolate as interp
# import numpy as np
from prog_models.models.uav_model.utilities.imports_ import np, spatial, interp

"""

"""


# DISTANCE FUNCTIONS
# ====================
# def sqeucdist(x, xprime):
#     """
#     squared distance of two vectors x and xprime. Equivalent to || x - xprime ||_2^2
#     :param x:       n x 1, input vector, doubles
#     :param xprime:  n x 1, query point vector, doubles
#     :return:        n x n, norm-2 of distance of vectors x and xprime
#     """
#     return spatial.distance.cdist(x, xprime, metric='sqeuclidean')

# def eucdist(x, xprime):
#     """
#     euclidean distance of two vectors x and prime. Equivalent to ||x - xprime||
#     :param x:       n x 1, input vector, doubles
#     :param xprime:  n x 1, query input vector, doubles
#     :return:        n x n, norm of distance of vectors x and xprime
#     """
#     return spatial.distance.cdist(x, xprime, metric='euclidean')

# def compute_spherical_law_terms(phi1, phi2, lam1, lam2):
#     sin1 = np.sin(phi1)
#     sin2 = np.sin(phi2)
#     cos1 = np.cos(phi1)
#     cos2 = np.cos(phi2)
#     cosdlam = np.cos(lam1 - lam2)
#     return sin1, cos1, sin2, cos2, cosdlam

# def arg_spherical_law_cos(sin_phi1, cos_phi1, sin_phi2, cos_phi2, cos_dlon):
#     return sin_phi1 * sin_phi2 + cos_phi1 * cos_phi2 * cos_dlon

# def great_circle_central_angle(sin_phi1, cos_phi1, sin_phi2, cos_phi2, cos_dlon):
#     return np.arccos(arg_spherical_law_cos(sin_phi1, cos_phi1, sin_phi2, cos_phi2, cos_dlon))

# def great_circle_sphericalawcos(r, phi1, phi2, lam1, lam2):
#     sin1, cos1, sin2, cos2, cosdlam = compute_spherical_law_terms(phi1, phi2, lam1, lam2)
#     dsigma = arg_spherical_law_cos(sin1, cos1, sin2, cos2, cosdlam)
#     return r * np.arccos(dsigma)



# def compute_distance_between_trajs(ref_traj, new_traj, time_bound=None):
#     if time_bound is None:              time_bound = ['avg', 'inf', 'sup', ['inf', 'sup'], ['sup', 'inf']]
#     elif type(time_bound) == str:       time_bound = [time_bound,]
#     distance_vectors, time_vectors = [], []
#     for item in time_bound:
#         time_unix_interp, ref_pos_interp, new_pos_interp = interpolate_position_at_timestamps(ref_traj, new_traj, time_bound=item)
#         distance_vector = greatcircle_distance(ref_pos_interp[:, 0], new_pos_interp[:, 0], ref_pos_interp[:, 1], new_pos_interp[:, 1])
#         distance_vectors.append(distance_vector)
#         time_vectors.append(time_unix_interp)
#     return distance_vectors, time_vectors

"""
def extract_pos_time_from_traj(traj1, traj2, timebound, coords='geodetic_pos'):
    if any([name in timebound for name in ['inf', 'sup']]):
        if type(timebound) == str:
            pos1  = traj1[coords + '_' + timebound]
            pos2  = traj2[coords + '_' + timebound]
            time1 = traj1['timestamps_' + timebound]
            time2 = traj2['timestamps_' + timebound]
        elif type(timebound) == list and len(timebound) == 2:
            pos1  = traj1[coords + '_' + timebound[0]]
            pos2  = traj2[coords + '_' + timebound[1]]
            time1 = traj1['timestamps_' + timebound[0]]
            time2 = traj2['timestamps_' + timebound[1]]
        else:
            raise Exception("The only option for varriabel time_bound are strings 'inf' or 'sup', or lists ['inf', 'sup'] or ['sup', 'inf'].")
    else:
        pos1  = traj1[coords]
        time1 = traj1['timestamps']
        pos2  = traj2[coords]   
        time2 = traj2['timestamps'] 
    return pos1, time1, pos2, time2
"""
"""
def interpolate_position_at_timestamps(ref_traj, new_traj, coordinates='geodetic_pos', time_bound='avg'):

    assert any([name in time_bound for name in ['avg', 'inf', 'sup']]), "Variable time_bound can be one of the following: avg, inf, or sup"
    # Extract position and time vectors for both trrajectories according to time bound (average, inferior or superior)
    ref_pos, ref_timevec, new_pos, new_timevec = extract_pos_time_from_traj(ref_traj, new_traj, time_bound, coords=coordinates)

    ndims = ref_pos.shape[1]

    ref_time_unix = np.asarray([ref_timevec[ii].timestamp() for ii in range(len(ref_timevec))])
    new_time_unix = np.asarray([new_timevec[ii].timestamp() for ii in range(len(new_timevec))])

    time0 = max(ref_time_unix[0],  new_time_unix[0])
    timeF = min(ref_time_unix[-1], new_time_unix[-1])

    index_both_aircraft_flying = (ref_time_unix>time0) * (ref_time_unix<timeF)
    ref_time_unix_interp = ref_time_unix[index_both_aircraft_flying]
    ref_pos_at_interp    = ref_pos[index_both_aircraft_flying]
    new_pos_at_interp    = np.zeros((len(ref_time_unix_interp), ndims))
    for ii in range(ndims):
        new_post_interp_fun      = interp.interp1d(new_time_unix, new_pos[:, ii], kind='linear')
        new_pos_at_interp[:, ii] = new_post_interp_fun(ref_time_unix_interp)

    return ref_time_unix_interp, ref_pos_at_interp, new_pos_at_interp 
"""

# EARTH-RELATED DISTANCE FUNCTIONS
# ================================
def greatcircle_distance(lat1, lat2, lon1, lon2):
    R       = 6371e3 # meters
    phi1    = lat1
    phi2    = lat2
    dphi    = (lat2 - lat1) 
    dlambda = (lon2 - lon1)

    a = np.sin(dphi / 2.0) * np.sin(dphi / 2.0) + \
            np.cos(phi1) * np.cos(phi2) * \
                np.sin(dlambda / 2.0) * np.sin(dlambda / 2.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))
    d = R * c # distance in meters
    return d


# def vincenty_distance(p1, p2, tol=1e-12, max_iter=200):
#     """ 
#     Compute distance between two geodetic coordinates p1=(lat1, lon1), p2=(lat2, lon2) using ellipsoid equations from Vincenty, 1975.
#         T. Vincenty, 1975. Direct and inverse solution of geodesics on the ellipsoid with application of nested equations. Survey Review 23(176), 88-93. 

#     The algorithm does not converge for two nearly antipodal points, for example: (0.0, 0.0), (0.5, 179.5).

#     :param p1:          tuple, list or 1D array, latitude and longitude of first point on Earth
#     :param p2:          tuple, list or 1D array, latitude and longitude of second point on Earth
#     :param tol:         scalar, tolerance to distance precision, default = 1e-12 typically refers to <1cm error
#     :param max_iter:    int, number of maximum iterations before breaking algorithm. Default = 200
#     :return:            distance between p1 and p2, in meters.
#     """
    
    # Extract latitude and longitude values
    # ------------------------------------
#     lat1, lon1 = p1
#     lat2, lon2 = p2

    # If points are identical, return distance=0
    # ------------------------------------------
#     if lat1 == lat2 and lon1 == lon2:
#         return 0.0
    
    # Define ellipsoid constants
    # -----------------------------
#     a  = 6378137.0          # semi-major Earth axis (radius at Equator), meters, according to WGS84
#     f  = 1/298.257223563    # flat parameter of the Ellipsoid, according to WGS84
#     b  = (1.0 - f)*a        # semi-minor axis of the ellipsoid (radius at the poles), meters, according to WGS84 = 6356752.314245 
    
    # Define coordinate-dependent values
    # ------------------------------------
#     U1 = np.arctan((1.0-f)*np.tan(lat1))    # reduced latitude (latitude on auxiliary sphere, lat1)
#     U2 = np.arctan((1.0-f)*np.tan(lat2))    # reduced latitude (latitude on auxiliary sphere, lat2)
#     L  = lon2 - lon1                        # difference over longitude of the two points
    # Compute trigonometric values for U1, U2
#     sin_U1 = np.sin(U1)
#     cos_U1 = np.cos(U1)
#     sin_U2 = np.sin(U2)
#     cos_U2 = np.cos(U2)

#     lam  = L    # initialize longitude difference between p1 and p2 on auxiliary sphere. It should asymptotically converge to 0
#     iter = 0    # initialize iterator
#     while iter < max_iter:
        # Trigonometry of lambda
#         sin_lam = np.sin(lam)
#         cos_lam = np.cos(lam)

        # trigonometry of sigma
#         sin_sigma = np.sqrt( (cos_U2 * sin_lam )**2.0 + (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lam)**2.0 )
#         if sin_sigma == 0.0:        return 0.0  # coincident points
#         cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lam
#         sigma      = np.arctan2(sin_sigma, cos_sigma)

        # trigonometry of alpha
#         sin_alpha  = cos_U1 * cos_U2 * sin_lam / sin_sigma
#         cos_alpha2 = 1.0 - sin_alpha**2.0
        # Compute cos(2 \sigma_m)
#         try:                            cos_2sigma_m = cos_sigma - 2.0 * sin_U1 * sin_U2 / cos_alpha2
#         except ZeroDivisionError:       cos_2sigma_m = 0.0
        
        # Compute new lambda
#         C       = f/16.0 * cos_alpha2 * ( 4.0 + f * (4.0 - 3.0 * cos_alpha2) )
#         lam_old = lam
#         lam     = L + (1.0 - C) * f * sin_alpha * ( sigma + C * sin_sigma * ( cos_2sigma_m + C * cos_sigma * ( -1.0 + 2.0 * cos_2sigma_m**2.0 ) ) )
        
        # Evaluate difference
#         d_lam = abs(lam - lam_old)
#         if d_lam < tol: break
#         iter += 1   # update iterator
    
    # Return value
    # ----------
#     if d_lam > tol or iter == max_iter: # Failure to converge
#         return None 
#     else:   # After lambda converged, compute the following:
#         u2     = cos_alpha2 * (a**2.0 - b**2.0) / b**2.0
#         A      = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-786.0 + u2 * (320.0 - 175.0*u2)))
#         B      = u2/1024.0 * ( 256.0 + u2 * ( -128.0 + u2 * (74.0 - 47.0*u2) ) )
#         dsigma = B * sin_sigma * ( cos_2sigma_m + 1.0/4.0 * B * ( cos_sigma * ( -1.0 + 2.0 * cos_2sigma_m**2.0) - B / 6.0 * cos_2sigma_m * (-3.0 + 4.0 * sin_sigma**2.0) * (-3.0 + 4.0 * cos_2sigma_m**2.0) ) )
#         s      = b * A * (sigma - dsigma)
#     return np.round(s, 6)


def geodetic_distance(lats, lons, alts, method='greatcircle', return_surf_vert=False):
    """
    Compute geodetic distance between two points defined by latitude, longitude, and altitude
            dist                = geodetic_distance(lats, lons, alts, method='greatcircle', return_surf_vert=False)
            surf_dist, ver_dist = geodetic_distance(lats, lons, alts, method='greatcircle', return_surf_vert=True)

    Distance in 3D using: add altitude as cartesian coordinate
    dist = sqrt(surface_distance**2 + (alt_1 - alt_2)**2)

    :param lats:                    tuple, list or 1D array, latitudes of the two points (radians)
    :param lons:                    tuple, list or 1D array, longitudes of the two points (radians)
    :param alts:                    tuple, list or 1D array, altitudes of the two points (meters)
    :param method:                  string, measurement method to compute distance. Options: 'greatcircle' and 'vincenty'
    :param return_surf_vert:        Boolean, whether to return surface distance and vertical distance separately (default=False)
    :return:                        distance between two points in meters. Tuple of values (surface and alt distance) if return_surf_vert==True; absolute distance otherwise 
    """
    assert len(lats)==2 and len(lons)==2 and len(alts)==2, "Latitudes, longitudes and altitude values must be 2-element lists or arrays."
    if type(alts[0])==np.ndarray:   # if altitudes are vectors, compute point-wise difference (must be same length)
        assert len(alts[0])==len(alts[1]), "If altitudes are vectors, their length must coincide."
        vert_dist = alts[1]-alts[0]
    else:   # if alts are two points, compute difference between them
        vert_dist = np.diff(alts)   # compute difference in altitude
    # Compute geodetic distance according to method
    if method=='greatcircle':       surface_dist = greatcircle_distance(lats[0], lats[1], lons[0], lons[1])
    elif method=='vincenty':        surface_dist = vincenty_distance([lats[0], lons[0]], [lats[1], lons[1]])
    else:                           raise Exception("Geodetic distance method " + method + " not recognized.")
    # return horizontal and vertical distance or total distance
    if return_surf_vert:            return surface_dist, vert_dist
    else:                           return np.sqrt(surface_dist**2.0 + vert_dist**2.0)
    

# def geodetic_distance_fast(lat1, lat2, lon1, lon2, alt1, alt2):
#     return np.sqrt(greatcircle_distance(lat1, lat2, lon1, lon2)**2.0 + (alt1 - alt2)**2.0)



# REFERENCE FRAMES
# ================
"""
def velocity_body_frame(phi, theta, psi, as_x, as_y, as_z):
    return np.dot( rot_earth2body(phi, theta, psi), np.array([as_x, as_y, as_z]).reshape((-1,)))
"""
    
def rot_earth2body(phi, theta, psi):
    R = np.zeros((3, 3))
    R[0, :] = np.array([np.cos(theta) * np.cos(phi),                                               
                        np.cos(theta) * np.sin(phi),                                                
                       -np.sin(theta)])
    R[1, :] = np.array([-np.cos(psi) * np.sin(phi) + np.sin(psi) * np.sin(theta) * np.cos(phi),
                         np.cos(psi) * np.cos(phi) + np.sin(psi) * np.sin(theta) * np.sin(phi),
                         np.sin(psi) * np.cos(theta)])
    R[2, :] = np.array([np.sin(psi) * np.sin(phi) + np.cos(psi) * np.sin(theta) * np.cos(phi),
                       -np.sin(psi) * np.cos(phi) + np.cos(psi) * np.sin(theta) * np.sin(phi),
                        np.cos(psi) * np.cos(theta)])
    return R

def rot_body2earth(phi, theta, psi):
    R = np.zeros((3, 3))
    R[0, :] = np.array([np.cos(psi) * np.cos(theta),
                        np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi),
                        np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)])
    R[1, :] = np.array([np.sin(psi) * np.cos(theta),
                        np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi),
                        np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)])
    R[2, :] = np.array([-np.sin(theta),
                         np.cos(theta) * np.sin(phi),
                         np.cos(theta) * np.cos(phi)])
    return R

"""
def R_phiX(phi):
    return np.array([ [1.0, 0.0, 0.0], 
                      [0.0, np.cos(phi), np.sin(phi)],
                      [0.0, -np.sin(phi), np.cos(phi)]])

def R_thetaY(theta):
    return np.array([ [np.cos(theta), 0.0, -np.sin(theta)],
                      [0.0, 1.0, 0.0],
                      [np.sin(theta), 0.0, np.cos(theta)]])

def R_psiZ(psi):
    return np.array([ [np.cos(psi), np.sin(psi), 0.0],
                      [-np.sin(psi), np.cos(psi), 0.0],
                      [0.0, 0.0, 1.0]])
"""                      

# def body_ang_vel_from_eulers(phidot, thetadot, psidot):
def body_ang_vel_from_eulers(phi, theta, psi, phidot, thetadot, psidot):
    """ 
    Compute the desired body angular velocities p, q, r given the desired Euler's angular velocities with respect to the inertial (Earth) reference frame.
    The desired body angular velocities will be used to compute the desired moments and total thrust given the rotor configuration. From the desired
    thrust and moments, it is possible to compute the desired rotor speed given the thrust allocation matrix (UAV dependent).

    :param phidot:      first Euler's angle (phi) rate of change
    :param thetadot:    second Euler's angle (theta) rate of change
    :param psidot:      third Euler's angle (psi) rate of change
    :return:            body angular velocities organized in column vector [p, q, r]^{\top}
    """
    # phi_vec   = np.array([phidot, 0.0, 0.0]).reshape((-1,))
    # theta_vec = np.array([0.0, thetadot, 0.0]).reshape((-1,))
    # psi_vec   = np.array([0.0, 0.0, psidot]).reshape((-1,))
    # return phi_vec + np.dot(R_phiX(phidot), theta_vec) + np.dot(np.dot(R_phiX(phidot), R_thetaY(thetadot)), psi_vec)
    p = phidot - psidot * np.sin(theta)
    q = thetadot * np.cos(phi) + psidot * np.sin(phi) * np.cos(theta)
    r = - thetadot * np.sin(phi) + psidot * np.cos(phi) * np.cos(theta)
    return p, q, r

# COORDINATE TRANSFORMATION
# ==========================

# def cart2circ(x, y, wrap_to=None, ang_unit='rad'):
#     """
#     Conversion from cartesian to circular coordinates of a vector in 2 dimension.
#     The function receives x and y, dimensions of the vector along horizontal (x) and vertical (y) directions (on the
#     other hand, horizonal and vertical are totally arbitrary and they are used as a convention).

#     The function returns the vector in circular coordinates (polar coordinates, but just in 2D), which is amplitude and
#     direction. wrap_to is used to constrain the angle between certain limits, namely (0, 180), (0,360), or the corresponding
#     radian version (0,pi), (0,2pi). Default for wrap_to is None, which means that no wrapping is applied.

#     ang_unit defines the unit of the angle, either radians ('rad', default), or degrees, 'deg'.
#     Parameters ang_unit and wrap_to must be aligned; if ang_unit is 'rad', then wrap_to must either be None or a value in radians.
#     Similarly, if ang_unit='deg' then wrap_to must either be None or a value in degrees.

#     if x and y are two 1D arrays, then the conversion is performed element-wise.

#     :param x:           (n,) array, doubles, values of vector along x-direction
#     :param y:           (n,) array, doubles, values of vector along y-direction
#     :param wrap_to:      None or scalar, wrapping of direction.  if None (default) no wrapping, otherwise follow unit of ang_unit.
#     :param ang_unit:     string, 'rad' (default) or 'deg', unit of angle defining vector direction.
#     :return amp:        (n,) array, doubles amplitude of vector.
#     :return angle:      (n,) array, doubles, direction of vector (either radians or degrees according to ang_unit).
#     """
#     # Compute amplitude
#     amp = np.sqrt(x ** 2. + y ** 2.)  # Compute magnitude of wind speed

#     # Calculating direction as positive counterclockwise from North (= 0)
#     angle = np.arctan2(y / amp, x / amp) - np.pi / 2.  # compute angle in radians
    
#     # Convert, wrapt and return
#     if ang_unit == 'deg':
#         angle = angle * 180.0 / np.pi  # convert to degees
#     if wrap_to is not None:  # wrap angle between (0,wrap) in case is wanted
#         return amp, angle % wrap_to
#     else:
#        return amp, angle

"""
def circ2cart(rho, theta, theta_from_east=True):
    if theta_from_east:     theta += np.pi / 2.0
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def gen_compass_angles(wps_enu):
    n         = wps_enu.shape[0]
    yaw_angle = np.zeros((n,))  # ??? actually length is 1 less, so we repeat last angle at end
    piO2      = np.pi / 2.0 #   atan2 returns -pi..pi. We subtract pi/2 so zero is +Y axis
    for jj in range(1, n):
        dy = wps_enu[jj, 1] - wps_enu[jj-1, 1]
        dx = wps_enu[jj, 0] - wps_enu[jj-1, 0]
        if dx or dy:
            yaw_angle[jj] = np.arctan2(dy, dx) - piO2
        else:
            yaw_angle[jj] = yaw_angle[jj-1]
    return yaw_angle
"""

def gen_heading_angle(lat, lon, alt):
    
    print('Generating heading angle ', end=" ")
    """
    # Compute heading using lat-lon coordinates
    # -----------------------------------------
    n    = len(lat)
    head = np.zeros((n,))
    for jj in range(1, n):
        dlon_ = lon[jj] - lon[jj-1]
        X = np.cos(lat[jj]) * np.sin(dlon_)
        Y = np.cos(lat[jj-1]) * np.sin(lat[jj]) - np.sin(lat[jj-1]) * np.cos(lat[jj]) * np.cos(dlon_)
        head[jj-1] = np.arctan2(X, Y)
    
    # Adjust heading based on minimum rotation between current and next heading.
    # ---------------------------------------------------------------------------
    
    # first, remove non-sense 0 which comes from same lat-lon position within the flight
    for idx in range(1, n): 
        prev_p = head[idx-1]
        curr_p = head[idx]
        if idx < n-1 and curr_p==0:
            head[idx] = prev_p
    
    # Now select the sign of the angle based on the shortest rotation the UAV is supposed to move
    for idx in range(n-1):
        curr_p = head[idx]
        next_p = head[idx+1]
        # If angles are identical move on 
        if curr_p == next_p:    
            continue
        else:
            next_p_2 = 2.0*np.pi - abs(next_p) # angle in opposite direction
            # Select angle based on mininum rotation necessary
            if abs(curr_p - next_p) < abs(curr_p - next_p_2):   head[idx+1] = next_p
            else:                                               head[idx+1] = next_p_2
    print('complete.')
    return head
    """
    # Compute heading using lat-lon coordinates
    # -----------------------------------------
    # This heading is calculated from North,
    # while reference frame is ENU (East-North-Up), therefore, the first direction is EAST, not NORTH.
    # Need to adjust heading for ENU reference frame after calculating it.
    head = heading_compute_geodetic(lat, lon)
    # Adjust first heading based on altitude (avoid issues with fictitious way-points later on)
    head = heading_adjust_first_nonzero(head, alt)
    # Adjust heading based on minimum rotation between current and next heading.
    # ---------------------------------------------------------------------------
    # Select the sign of the angle based on the shortest rotation the UAV is supposed to move
    head = heading_adjust_rotation(head)

    print('complete.')
    return head

def heading_adjust_first_nonzero(heading, altitude):
    n = len(heading)
    for jj in range(n):
        if altitude[jj] > 1.0 and heading[jj] == 0 and heading[jj + 1] != 0:
            heading[jj] = heading[jj + 1]
            break
    return heading

def heading_adjust_rotation(heading):
    n = len(heading)
    for idx in range(n-1):
        curr_p = heading[idx]
        next_p = heading[idx+1]
        # If angles are identical move on
        if curr_p == next_p:
            continue
        else:
            next_p_2 = next_p - 2.0*np.pi   # angle in opposite direction
            # Select angle based on mininum rotation necessary
            if abs(curr_p - next_p) < abs(curr_p - next_p_2):   heading[idx+1] = next_p
            else:                                               heading[idx+1] = next_p_2
    return heading

def heading_compute_geodetic(lat, lon):
    n = len(lat)
    heading = np.zeros((n,))
    for jj in range(1, n):
        dlon_ = lon[jj] - lon[jj - 1]
        dlat_ = lat[jj] - lat[jj - 1]
        X = np.cos(lat[jj]) * np.sin(dlon_)
        Y = np.cos(lat[jj - 1]) * np.sin(lat[jj]) - np.sin(lat[jj - 1]) * np.cos(lat[jj]) * np.cos(dlon_)
        head_temp = np.arctan2(X, Y)
        if Y != 0:
            head_temp -= np.pi / 2.0
            head_temp *= -1.0
        if jj < n - 1 and ((dlat_ != 0 or dlon_ != 0) and head_temp == 0):
            head_temp = heading[jj - 2]
        heading[jj - 1] = head_temp
    heading[-1] = heading[-2]
    return heading


# def coord_distance(lat, lon, alt):
#     """
#     distance in 3D using great circle: add altitude as cartesian coordinate
#                                 dist = sqrt(great_circle((lat_1, lon_1), (lat_2, lon_2)).m**2, (alt_1 - alt_2)**2)
#     Using ECEF coordinates:     
#                                 d = sqrt{(X_2-X_1)^2 + (Y_2-Y_1)^2 + (Z_2-Z_1)^2}
#     """
#     dh = []
#     dv = np.diff(alt)
#     for ii in range(1, len(lat)):
#         dh_tmp = greatcircle_distance(lat[ii-1], lat[ii], lon[ii-1], lon[ii])
#         dh.append(dh_tmp)
#     return np.asarray(dh), dv


def transform_from_cart_to_geo(cartesian_matrix, lat0, lon0, alt0):
    coord = Coord(lat0=lat0, lon0=lon0, alt0=alt0)
    geodetic_matrix = np.zeros((len(cartesian_matrix[:,0]), 3))
    geodetic_matrix[:, 0], geodetic_matrix[:, 1], geodetic_matrix[:, 2] = coord.enu2geodetic(cartesian_matrix[:,0],
                                                                                             cartesian_matrix[:,1],
                                                                                             cartesian_matrix[:,2])
    return geodetic_matrix


# Coordinate class
# ===============
class Coord():

    def __init__(self, lat0, lon0, alt0):
        self.a = 6378137.0                      # [m], equatorial radius
        self.f = 1.0 / 298.257223563            # [-], ellipsoid flatness
        self.b = self.a * (1.0 - self.f)        # [m], polar radius
        self.e = np.sqrt(self.f * (2-self.f))   # [-], eccentricity
        self.lat0 = lat0
        self.lon0 = lon0
        self.alt0 = alt0
        self.N0   = self.a / np.sqrt(1 - self.e**2.0 * np.sin(self.lat0)**2.0) # [m], Radius of curvature on the Earth

    def ecef2enu(self, xecef, yecef, zecef):
        # N = self.a / np.sqrt(1 - self.e**2.0 * np.sin(self.lat0)**2.0) # [m], Radius of curvature on the Earth

        # Compute location of the origin of ENU reference frame in the ECEF reference frame
        x0 = (self.alt0 + self.N0) * np.cos(self.lat0) * np.cos(self.lon0)
        y0 = (self.alt0 + self.N0) * np.cos(self.lat0) * np.sin(self.lon0)
        z0 = (self.alt0 + (1.0 - self.e**2.0) * self.N0) * np.sin(self.lat0)

        # Compute relative distance between the data points and the origin of the
        # ENU reference frame
        xd = xecef - x0
        yd = yecef - y0
        zd = zecef - z0

        # Compute coordinates in ENU reference frame
        x  = - np.sin(self.lon0) * xd + np.cos(self.lon0) * yd
        y  = - np.cos(self.lon0) * np.sin(self.lat0) * xd - np.sin(self.lat0) * np.sin(self.lon0) * yd + np.cos(self.lat0) * zd
        z  =   np.cos(self.lat0) * np.cos(self.lon0) * xd + np.cos(self.lat0) * np.sin(self.lon0) * yd + np.sin(self.lat0) * zd

        return x, y, z

    def enu2ecef(self, xenu, yenu, zenu):
        # N = self.a / np.sqrt(1.0 - self.e**2.0 * np.sin(self.lat0)**2.0)  # [m], Radius of curvature on the Earth

        # Compute coordinates of origin of ENU in the ECEF reference frame
        x0 = (self.alt0 + self.N0) * np.cos(self.lat0) * np.cos(self.lon0)
        y0 = (self.alt0 + self.N0) * np.cos(self.lat0) * np.sin(self.lon0)
        z0 = (self.alt0 + (1.0 - self.e**2.0) * self.N0) * np.sin(self.lat0)

        # Compute relative coordinates in ECEF reference frame
        xd = -np.sin(self.lon0) * xenu - np.cos(self.lon0) * np.sin(self.lat0) * yenu + np.cos(self.lat0) * np.cos(self.lon0) * zenu
        yd =  np.cos(self.lon0) * xenu - np.sin(self.lat0) * np.sin(self.lon0) * yenu + np.cos(self.lat0) * np.sin(self.lon0) * zenu
        zd =  np.cos(self.lat0) * yenu + np.sin(self.lat0) * zenu

        # Compute global coordinates in ECEF reference frame by adding the location
        # of the ENU to the relative ECEF coordinates
        xecef = xd + x0
        yecef = yd + y0
        zecef = zd + z0
        return xecef, yecef, zecef


    def ecef2geodetic(self, X, Y, Z):
        """
        lat, lon, alt = Coord.ecef2geodetic(X, Y, Z)
    
        Convert ECEF coordinates into geodetic coordinates.
    
        The equations used in this function come from the report:
            Geomatics Guidance Note 7, part 2 Coordinate Conversions & Transformations including Formulas
            International Association of Oil & Gas producers, Report 373-7-2,
            April 2018.
            See Chapter 2, Subsection 2.2, Sub-subsection 2.2.1
    
        INPUT
            X       n x 1 array, real, [m], X coordinate in ECEF reference system
            Y       n x 1 array, real, [m], Y coordinate in ECEF reference system
            Z       n x 1 array, real, [m], Z coordinate in ECEF reference system
    
        OUTPUT
            lat     n x 1 array, real, [rad], latitude values
            lon     n x 1 array, real, [rad], longitude values
            alt     n x 1 array, real, [m], altitudevalues
        """

        # Ancillary variables
        epsilon = self.e**2.0 / (1.0 - self.e**2.0)
        b       = self.a * (1.0 - self.f)
        p       = np.sqrt(X**2.0 + Y**2.0)
        q       = np.arctan2( Z * self.a , p * b)

        # Compute latitude, longitude and altitude.
        lat = np.arctan2( (Z + epsilon * b * np.sin(q)**3.0) , (p - self.e**2.0 * self.a * np.cos(q)**3.0) ) # [rad], latitude
        lon = np.arctan2( Y , X );                                                                           # [rad], longitude
        N   =  self.a /  np.sqrt(1.0 - self.e**2.0 * np.sin(lat)**2.0);                                    # [m], Radius of curvature on the Earth
        alt = (p / np.cos(lat)) - N
        return lat, lon, alt


    def geodetic2ecef(self, lat, lon, alt):
        N = self.a / np.sqrt(1.0 - self.e**2.0 * np.sin(lat)**2.0)  # [m], Radius of curvature on the Earth, phi = latitude
        X = (N + alt) * np.cos(lat) * np.cos(lon)
        Y = (N + alt) * np.cos(lat) * np.sin(lon)
        Z = (alt + (1.0 - self.e**2.0) * N) * np.sin(lat)
        return X, Y, Z


    def enu2geodetic(self, x, y, z):
        """
        Return lat, lon, alt
        """
        xecef_tmp, yecef_tmp, zecef_tmp = self.enu2ecef(x, y, z)
        return self.ecef2geodetic(xecef_tmp, yecef_tmp, zecef_tmp )


    def geodetic2enu(self, lat, lon, alt):
        X, Y, Z = self.geodetic2ecef(lat, lon, alt)
        return self.ecef2enu(X, Y, Z)
