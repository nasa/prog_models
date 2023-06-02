# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Functions for generating route - setting waypoints, calculating ETAs 
"""

import numpy as np
import datetime as dt

from prog_models.utils.traj_gen_utils import geometry as geom

# FUNCTIONS
# ==========
def check_and_adjust_eta_feasibility(lat, lon, alt, eta, vehicle_max_speed, vehicle_max_speed_vert, distance_method='greatcircle'):
    """
    Check that ETAs are feasible by verifying that the average speed necessary to 
    reach the waypoint is not beyond the maximum speed of the vehicle.
    If so, the ETA for each waypoint is increased progressively until the criterion is met.

    :param lat:                         rad, n x 1, latitude of waypoints
    :param lon:                         rad, n x 1, longitude of waypoints
    :param alt:                         m, n x 1, altitude of waypoints
    :param eta:                         s, n x 1, ETAs of waypoints
    :param vehicle_max_speed:           m/s, scalar, vehicle max horizontal speed 
    :param vehicle_max_speed_vert:      m/s, scalar, vehicle max speed ascending or descending
    :param distance_method:             string, method to calculate the distance between two points, either greatcircle (default), or vincenty.
    :return:                            s, n x 1, ETAs of waypoints corrected to ensure feasibility.
    """
    # Check if the speeds required for ETAs are beyond the max vehicle speeds.
    n = len(lat)-1
    d_eta = np.diff(eta)
    for point in range(n):
        dh, dv = geom.geodetic_distance([lat[point], lat[point + 1]],
                                        [lon[point], lon[point + 1]],
                                        [alt[point], alt[point + 1]],
                                        method=distance_method, return_surf_vert=True)
        dv = dv[0]
        # If speed is larger than maximum (possible when both dh, dv>0), increment d_eta to reduce until desired (consider margin)
        while dh / d_eta[point] > vehicle_max_speed or dv / d_eta[point] > vehicle_max_speed_vert:
            d_eta[point] += 1.0
    return np.asarray(np.cumsum(np.insert(d_eta, 0, 0.0)))


def build(lat, lon, alt, departure_time, parameters: dict = dict(), etas=None, vehicle_max_speed=None):
    """
    Generate route given waypoints (lat, lon, alt), departure time, 
    etas or speed between waypoints, additional time for hovering, takeoff, landing, and eventually adjust ETA in case 
    one wants to generate a route with ETAs in the past or in the future (for example, if one has wind data referring to a specific point in time).

    To generate a Route, one can either use ETAs or the speed between waypoints. The latter is composed of cruise_speed, ascent_speed, 
    descent_speed, landing_speed. Both ETAs and speed values as set to None as default, but one of ETAs or speed values must be provided.
    
    :param lat:         1D array or list, latitude positions
    :param lon:         1D array or list, longitude positions
    :param alt:         1D array or list, altitude positions
    :param departure_time:          timestamp, flight departure time
    The following are keywords for the dictionary 'route_parameters':
        :param cruise_speed:            scalar, cruise speed between waypoints, default is None.
        :param ascent_speed:            scalar, ascent speed, default is None.
        :param descent_speed:           scalar, descent speed, default is None.
        :param landing_speed:           scalar, landing speed when vehicle is <10m from the ground, default is None.
        :param hovering_time:           scalar, additional hovering time, default is 0.
        :param add_takeoff_time:        scalar, additional takeoff time, default is 0.
        :param add_landing_time:        scalar, additional landing time, default is 0.
        :param adjust_eta:              dictionary with keys ['hours', 'seconds'], to adjust route time 
    :param etas:                    1D array or list, ETAs at each waypoints, in seconds, default is None.
    :return:                        route, from Route class.
    """
    params = dict(
        cruise_speed=None,        # m/s, default cruise speed
        ascent_speed=None,        # m/s, default ascent speed (climb)
        descent_speed=None,       # m/s, default descent speed
        landing_speed=None,       # m/s, default landing speed (when < 10ft from ground)
        hovering_time=0.0,        # s, scalar, additional hovering time, default is 0.
        takeoff_time=0.0,         # scalar, additional takeoff time, default is 0.
        landing_time=0.0,         # scalar, additional landing time, default is 0.
        adjust_eta=None,          # dictionary with keys ['hours', 'seconds'], to adjust route time
        additional_hover_time=0.5,  # s, additional hovering time if waypoints are identical (to avoid extreme acceleration values).
    )
    params.update(parameters)

    route = Route(departure_time=departure_time, 
                  cruise_speed=params['cruise_speed'], 
                  ascent_speed=params['ascent_speed'], 
                  descent_speed=params['descent_speed'], 
                  landing_speed=params['landing_speed'])
    route.set_waypoints(lat, lon, alt) 
    route.set_eta(eta=etas,
                  hovering=params['hovering_time'],
                  add_takeoff_time=params['takeoff_time'],
                  add_landing_time=params['landing_time'],
                  same_wp_hovering_time=params['additional_hover_time'],
                  vehicle_max_speed=vehicle_max_speed)
    if params['adjust_eta'] is not None:
        if isinstance(params['adjust_eta'], dict) == False:
            raise TypeError("adjust_eta parameter must be a dictionary with keys: 'hours' and 'seconds'.")
        route.adjust_eta(hours_=params['adjust_eta']['hours'], seconds_=params['adjust_eta']['seconds'])
    return route


def reshape_route_attribute(x, dim=None, msk=None):
    """
    Reshape route attribute to ensure correct length

    :param x:           attribute to reshape
    :param dim:         scalar, int or None, desired length of attribute. Used only if x has no attribute "len"
    :param msk:         list or array of int, or None, mask to reshape attribute x. The function return x by repeating x[msk] at msk points.
    :return:            reshaped attribute
    """
    if not hasattr(x, "__len__"):   
        x = x * np.ones((dim, ))
    elif msk:
        x = np.insert(x, msk, x[msk])
    return x

# ROUTE CLASS
# ============    
class Route():
    """
    This class is used to generate a route with waypoints, corresponding ETAs, and the desired speed in between waypoints.
    """
    def __init__(self, 
                 departure_time=None, 
                 cruise_speed=None, 
                 ascent_speed=None, 
                 descent_speed=None, 
                 landing_speed=None, 
                 landing_alt=10.5):
        """
        Initialize class

        :param departure_time:       dt.datetime object, departure time according to plan.
        :param cruise_speed:         m/s, scalar, cruise speed between waypoints
        :param ascent_speed:         m/s, scalar, ascent speed between waypoints
        :param descent_speed:        m/s, scalar, descent speed between waypoints
        :param landing_speed:        m/s, scalar, landing speed, either ascending or descending speed when altitude is lower than landing_altitude
        :param landing_alt:          m, scalar, landing altitude. Below this altitude, the vertical speed of the vehicle is set to landing_speed, regardless if ascending or descending
        """
        if type(departure_time) != dt.datetime:
            departure_time = departure_time.to_pydatetime()
        self.departure_time   = departure_time
        self.landing_altitude = landing_alt
        self.cruise_speed     = cruise_speed
        self.ascent_speed     = ascent_speed
        self.descent_speed    = descent_speed
        self.landing_speed    = landing_speed
        self.hovering         = None
        self.takeoff_time     = 0.  # Default additional time for takeoff
        self.landing_time     = 0.  # Default additional time for landing
        self.lat              = []
        self.lon              = []
        self.alt              = []
        self.eta              = []

    def __call__(self):
        """
        Route calling function returns the flight plan in latitude, longitude, altitude, ETA.
        """
        return self.lat, self.lon, self.alt, self.eta
    
    def adjust_eta(self, hours_=0., seconds_=0.0):
        """
        Adjust departure time and corresponding ETAs according to hours_ and seconds_ from now.
        For example:
        
        route = Route()
        route.adjust_eta(hours_=1, seconds=_=4600)
        
        will define a route with departure time set to 1 hour and 4600 seconds after the method "route.adjust_eta()" was called.

        :param hours_:          scalar, double, number of hours ahead of "now" the route should start
        :param seconds_:        scalar, double, number of seconds ahead of "now" the route should start
        """
        self.departure_time = dt.datetime.now() + dt.timedelta(hours=hours_, seconds=seconds_) 
        timedelta            = self.departure_time - self.eta[0]
        self.eta            = [item + timedelta  for item in self.eta]
        return
    
    def add_point(self, lat, lon, alt, eta=None):
        """
        Append a waypoint composed of lat, lon, alt to the route. If eta is not None, the eta is also added to the ETAs.
        :param lat:     rad, scalar, latitude coordinate of point
        :param lon:     rad, scalar, longitude coordinate of point
        :param alt:     rad, scalar, altitude coordinate of point
        :param eta:     s, scalar, ETAs of point
        """
        self.lat.append(lat)
        self.lon.append(lon)
        self.alt.append(alt)
        if eta:
            self.eta.append(eta)

    def set_waypoints(self, lat, lon, alt, eta=None):
        """
        Set waypoints as a series of coordinates contained in input lat, lon, alt, and eta (optional, default = None)

        :param lat:     rad, n x 1, latitude coordinates
        :param lon:     rad, n x 1, longitude coordinates
        :param alt:     rad, n x 1, altitude coordinates
        :param eta:     s, n x 1, ETAs. Optional, default = None.
        """
        self.lat = lat
        self.lon = lon
        self.alt = alt
        if eta is not None:
            self.eta = eta
        return
        
    def set_waypoints_as_array(self):
        """
        Set waypoints as arrays, if passed as lists or other types
        """
        self.lat = np.asarray(self.lat)
        self.lon = np.asarray(self.lon)
        self.alt = np.asarray(self.alt)
        return
        
    def set_landing_waypoints(self, set_landing_eta=False):
        """
        Set waypoints at altitude defined by landing_altitude.
        By so doing, the trajectory can split the ascending and descending phases into segments, and assign the landing_speed to all segments that
        fall below the landing_alt mark.

        :param set_landing_eta:         Bool, whether to set the landing ETAs as well. Otherwise, they will be calculated later.
        :return:                        int, n x 1 array, indices of waypoints that define the landing.
        """
        # if waypoints are stored as list, define them as array
        if type(self.alt) == list:  
            self.set_waypoints_as_array()
        
        # get boolean flag where altitude is below the landing altitude, and get the corresponding indices
        idx_land     = np.asarray(self.alt < self.landing_altitude)
        idx_land_pos = np.where(idx_land)[0]    
        
        # if there are waypoints below landing altitude: append a landing waypoint (with altitude self.landing_altitude) accordingly.
        if idx_land_pos.size != 0:
            n_ = len(self.lat)
            counter = 0
            for item in idx_land_pos:
                if item == 0:   # if first element is below, just add a landing waypoint
                    self.lat = np.insert(self.lat, item + 1, self.lat[item])
                    self.lon = np.insert(self.lon, item + 1, self.lon[item])
                    self.alt = np.insert(self.alt, item+1, self.landing_altitude*1.0)
                    counter += 1
                elif item == n_-1: # if one before the last element, add landing waypoints right before landing.
                    if self.alt[item+counter-1] > self.landing_altitude:
                        self.lat = np.insert(self.lat, -1, self.lat[item+counter-1])
                        self.lon = np.insert(self.lon, -1, self.lon[item+counter-1])
                        self.alt = np.insert(self.alt, -1, self.landing_altitude*1.0)
                        counter += 1
                else:
                    if self.alt[item+counter] - self.alt[item+counter-1] < 0:     # descending
                        idx_delta = 0
                    else:   # ascending
                        idx_delta = +1
                    self.lat = np.insert(self.lat, item+counter + idx_delta, self.lat[item+counter])
                    self.lon = np.insert(self.lon, item+counter + idx_delta, self.lon[item+counter])
                    self.alt = np.insert(self.alt, item+counter + idx_delta, self.landing_altitude*1.0)
                    counter += 1
                    if idx_delta == 0:  # if descended, needs to go back up
                        if self.alt[item+counter+1] > self.landing_altitude:
                            self.lat = np.insert(self.lat, item+counter+1, self.lat[item+counter+1])
                            self.lon = np.insert(self.lon, item + counter+1, self.lon[item + counter+1])
                            self.alt = np.insert(self.alt, item + counter+1, self.landing_altitude*1.0)
                            counter += 1

        # Recalculate landing positions with new waypoints:
        idx_land = np.asarray(self.alt < self.landing_altitude)
        idx_land_pos = np.where(idx_land)[0]

        # Interpolate ETA at landing waypoints linearly
        if set_landing_eta:
            eta_landing = []
            counter     = 0
            for idx in idx_land_pos:
                delta_alt   = self.alt[idx+1+counter] - self.alt[idx-1+counter]
                t_0         = self.eta[idx-1].timestamp()
                t_1         = self.eta[idx].timestamp()
                delta_t     = t_1 - t_0
                delta_ratio = delta_alt / delta_t
                t_land      = 1.0/delta_ratio * ( self.landing_altitude + delta_ratio * t_0 - self.alt[idx-1])
                counter    += 1
                eta_landing.append(dt.datetime.utcfromtimestamp(t_land) + dt.timedelta(hours=-8))   # -8 because it's California time
            self.eta = np.insert(self.eta, idx_land_pos, eta_landing)
        return idx_land_pos

    def set_eta(self, eta=None, hovering=0, add_takeoff_time=None, add_landing_time=None, same_wp_hovering_time=1.0,vehicle_max_speed=None):
        """
        Assign ETAs to waypoints.
        If ETAS are provided (i.e., eta is not None), assign them.
        If they are not provided, compute them from desired speed.

        :param eta:                     s, n x 1 array or None. ETAs for all waypoints.
        :param hovering:                s, scalar or n x 1 array. Default = 0. hovering condition to add to the waypoints.
        :param add_takeoff_time         s, scalar or None, add takeoff time
        :param add_landing_time         s, scalar or None, add landing time
        :param same_wp_hovering_time    s, scalar, ancillary variable to avoid spurious accelerations when the vehicle needs to hover on the same waypoint. It adds time to the ETA, which lowers accelerations.
        """
        # Assign ETAS
        # ============
        if eta is not None: # if ETA is provided, assign to self.eta
            if hasattr(eta, "__len__") is False or len(eta)!=len(self.lat):
                raise TypeError("ETA must be vector array with same length as lat, lon and alt.")
                
            eta_unix = np.zeros_like(eta, dtype=np.float64)
            for i, eta_i in enumerate(eta):
                eta_unix[i] = dt.datetime.timestamp(eta_i) 
            # Check if speeds required for ETAs are above max speeds            
            if self.cruise_speed is None:       cruise_speed_val = 6.0
            else:                               cruise_speed_val = self.cruise_speed
            if self.ascent_speed is None:       vert_speed_val = 3.0
            else:                               vert_speed_val = self.ascent_speed
            # Get the new relative ETA given expected ETAs and distance between waypoints
            relative_eta_new = check_and_adjust_eta_feasibility(self.lat, self.lon, self.alt, eta_unix-eta_unix[0], cruise_speed_val, vert_speed_val, distance_method='greatcircle')
            self.eta = np.asarray([dt.datetime.fromtimestamp(relative_eta_new[i] + eta_unix[0]) for i in range(len(eta))])

        else:   # if ETA is not provided, compute it from desired cruise speed and other speeds
            if self.cruise_speed == None:
                raise TypeError("If ETA is not provided, desired speed (cruise, ascent, descent) must be provided.")

            idx_land_pos = self.set_landing_waypoints(set_landing_eta=False)

            if add_takeoff_time is not None:    self.takeoff_time = add_takeoff_time
            if add_landing_time is not None:    self.landing_time = add_landing_time

            # Check speed dimensions
            n = len(self.lat)
            self.cruise_speed  = reshape_route_attribute(self.cruise_speed, dim=n-1, msk=idx_land_pos)
            self.ascent_speed  = reshape_route_attribute(self.ascent_speed, dim=n-1, msk=idx_land_pos)
            self.descent_speed = reshape_route_attribute(self.descent_speed, dim=n-1, msk=idx_land_pos)
            self.hovering      = reshape_route_attribute(hovering, dim=n-1, msk=idx_land_pos)
            if self.landing_speed is None:  self.landing_speed = self.descent_speed
            else:                           self.landing_speed = reshape_route_attribute(self.landing_speed, dim=n-1, msk=idx_land_pos)

            self.eta = self.compute_etas_from_speed(takeoff_time=self.takeoff_time, landing_time=self.landing_time, hovering=self.hovering,
                                                    same_wp_hovering_time=same_wp_hovering_time)

        return self.eta

    def compute_etas_from_speed(self, hovering, takeoff_time, landing_time, distance_method='greatcircle', cruise_speed=None, ascent_speed=None, descent_speed=None, same_wp_hovering_time=1.0, assign_eta=True):
        """
        Compute the ETAs for all waypoints given the desired cruise and vertical speed.

        :param cruise_speed:        m/s, cruise speed between waypoints
        :param ascent_speed:        m/s, ascent speed between waypoints
        :param descent_speed:       m/s, descent speed between waypoints
        :param hovering:            s, extra time for hovering in between waypoints
        :param takeoff_time:        s, extra time needed to take off
        :param landing_time:        s, extra time needed to land
        :param distance_method:     string, method used to compute the distance between two points, either 'greatcircle' or 'vincenty'. default = 'greatcircle'
        :return:                    s, n x 1, ETAs for all waypoints.
        """
        if len(self.lat) <= 2:
            raise TypeError("At least 3 waypoints are required to compute ETAS from speed. Only {} were given.".format(len(self.lat)))
        
        # define margin on cruise speed
        # ----------------------------
        # If calculated ETA produces a speed that is larger than desired speed, we can accommodate it as long as is within this margin (%)
        cruise_speed_margin = 0.1   # %, 'extra' speed we can tolerate on cruise.
        vert_speed_margin = 0.05    # %, 'extra' speed we can tolerate on ascent/descent

        # Reshape speed
        if cruise_speed is None:    
            cruise_speed  = self.cruise_speed
        else:                       
            cruise_speed      = reshape_route_attribute(cruise_speed, dim=len(self.lat)-1)
            self.cruise_speed = cruise_speed

        if ascent_speed is None:    
            ascent_speed  = self.ascent_speed
        else:                       
            ascent_speed      = reshape_route_attribute(ascent_speed, dim=len(self.lat)-1)
            self.ascent_speed = ascent_speed

        if descent_speed is None:   
            descent_speed = self.descent_speed
        else:                       
            descent_speed      = reshape_route_attribute(descent_speed, dim=len(self.lat)-1)
            self.descent_speed = descent_speed

        if self.landing_speed is None:  
            self.landing_speed = descent_speed.copy()

        if hovering is None:        
            hovering      = self.hovering
        else:                       
            hovering      = reshape_route_attribute(hovering, dim=len(self.lat)-1)
            self.hovering = hovering

        # Compute relative ETAs
        # -------------------
        alt_for_land = self.alt[1:]
        n     = len(self.lat)-1
        d_eta = np.zeros((n,))
        for point in range(n):
            dh, dv = geom.geodetic_distance([self.lat[point], self.lat[point+1]], 
                                            [self.lon[point], self.lon[point+1]], 
                                            [self.alt[point], self.alt[point+1]], 
                                            method=distance_method, return_surf_vert=True)
            dv = dv[0]

            # Identify correct vertical speed
            if   dv > 0 and alt_for_land[point] > self.landing_altitude:    vert_speed = ascent_speed[point]
            elif dv > 0 and alt_for_land[point] <= self.landing_altitude:   vert_speed = self.landing_speed[point]
            elif dv < 0 and alt_for_land[point] >= self.landing_altitude:   vert_speed = descent_speed[point]
            elif dv < 0 and alt_for_land[point] < self.landing_altitude:    vert_speed = self.landing_speed[point]
            else:                                                           vert_speed = 0. # not moving vertically.

            # Define the correct speed:
            if np.isclose(dh + dv, 0.0):
                d_eta[point] = same_wp_hovering_time  # if there's no vertical / horizontal speed (waypoints are identical) add a default hovering value to avoid extreme accelerations.
            else:
                if np.isclose(dh, 0.):      speed_sq = vert_speed**2.0
                elif np.isclose(dv, 0.):    speed_sq = cruise_speed[point]**2.0
                else:                       speed_sq = cruise_speed[point]**2.0 + vert_speed**2.0
                d_eta[point] = np.sqrt( (dh**2.0 + dv**2.0) / speed_sq )
                # If speed is larger than desired (possible when both dh, dv>0), increment d_eta to reduce until desired (consider margin)
                while dh/d_eta[point] > (cruise_speed[point]*(1.+cruise_speed_margin)) or dv/d_eta[point] > (vert_speed*(1.+vert_speed_margin)):
                    d_eta[point] += 1.0

            if hovering[point] != 0:    d_eta[point] += hovering[point]

        d_eta[0]  += takeoff_time
        d_eta[-1] += landing_time
        eta_array = np.asarray(np.cumsum(np.insert(d_eta, 0, 0.0)))
        if assign_eta:  self.eta = [dt.datetime.fromtimestamp(eta_array[ii] + + self.departure_time.timestamp()) for ii in range(len(eta_array))]
        return self.eta
        