import numpy as np
import datetime as dt

import os, sys
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '/utilities'))

from prog_models.models.uav_model.utilities import geometry as geom

# FUNCTIONS
# ==========
def build(name, lat, lon, alt, departure_time, etas=None, cruise_speed=None, ascent_speed=None, descent_speed=None, landing_speed=None,
             hovering_time=0., add_takeoff_time=0., add_landing_time=0., adjust_eta=None):
    """
    Generate route given waypoints (lat, lon, alt), departure time, 
    etas or speed in-between way-points, additional time for hovering, takeoff, landing, and eventually adjust eta in case 
    one wants to generate a route with ETAs in the past or in the future (for example, if one has wind data referring to a specific point in time).

    To generate a Route, one can either use etas or the speed in-between waypoints. The latter composed of cruise_speed, ascent_speed, 
    descent_speed, landing_speed. Both etas and speed values as set to None as default, but one of etas or speed values must be provided.
    
    :param name:        str, route name
    :param lat:         1D array or list, latitude positions
    :param lon:         1D array or list, longitude positions
    :param alt:         1D array or list, altitude positions
    :param departure_time:          timestamp, flight departure time
    :param etas:                    1D array or list, etas at each waypoints, in seconds, default is None.
    :param cruise_speed:            scalar, cruise speed in-between waypoints, default is None.
    :param ascent_speed:            scalar, ascent speed, default is None.
    :param descent_speed:           scalar, descent_speed, default is None.
    :param landing_speed:           scalar, landing_speed when vehicle is <10m from the ground, default is None.
    :param hovering_time:           scalar, additional hovering time, default is 0.
    :param add_takeoff_time:        scalar, additional takeoff time, default is 0.
    :param add_landing_time:        scalar, additional landing time, default is 0.
    :param adjust_eta:              dictionary with keys ['hours', 'seconds'], to adjust route time 
    :return:                        route, from Route class.
    """
    route = Route(name=name, 
                  departure_time=departure_time, 
                  cruise_speed=cruise_speed, 
                  ascent_speed=ascent_speed, 
                  descent_speed=descent_speed, 
                  landing_speed=landing_speed)
    route.set_waypoints(lat, lon, alt) 
    route.set_eta(eta=etas, hovering=hovering_time, add_takeoff_time=add_takeoff_time, add_landing_time=add_landing_time)
    if adjust_eta is not None:
        assert adjust_eta==dict, 'adjust_eta must be a dictionary with keys: "hours" and "seconds."'
        route.adjust_eta(hours_=adjust_eta['hours'], seconds_=adjust_eta['seconds'])
    return route


def reshape_route_attribute(x, dim=None, msk=None):
    if not hasattr(x, "__len__"):   
        assert dim is not None, "Must provide dim to generate vector."
        x = x * np.ones((dim, ))
    elif msk:
        x = np.insert(x, msk, x[msk])
    return x

"""
def read_routes(fname, str_=None, decimal=True, ft2m=False):
    
    # Set default name for new route:
    # -------------------------------
    if str_ is None:   str_ = ['SF', 'OAK', 'Freemont', 'Stanford']
    
    # Extract routes from files
    # -------------------------
    file   = open(fname, 'r')
    lines  = file.readlines()
    routes = []
    for line in lines:
        if line == '\n':    continue
        if any(name in line for name in str_):
            newroute = Route(line[:line.rfind('=')-1])
            newroute.departure_time = dt.datetime.strptime(line[line.rfind('D')+2:-1], "%Y-%m-%d %H:%M:%S")
            routes.append(newroute)
            continue
        lat, lon, alt = parse_coordinate_line(line, decimal=decimal, ft2m=ft2m)
        routes[-1].add_point(lat, lon, alt)

    # Add x, y, z to routes
    # --------------------
    for idx, route in enumerate(routes):
        coord = geom.Coord(route.lat[0], route.lon[0], route.alt[0])
        route.x, route.y, route.z = coord.geodetic2enu(route.lat, route.lon, route.alt)
        routes[idx] = route
    return routes
"""

""" 
def parse_coordinate_line(l, decimal=True, ft2m=False):
    # According to how the text file with trajectory is organized.
    lat = degrees_to_decimal_from_str(l[:l.find(',')])
    lon = degrees_to_decimal_from_str(l[l.find(',')+2:-1])
    if decimal:
        lat = lat / 180.0 * np.pi
        lon = lon / 180.0 * np.pi

    # Get altitude in ft or m
    if ft2m:    alt = feet2meters_from_str(l[l.rfind(',')+2:l.rfind('\n')-2])
    else:       alt = float(l[l.rfind(',')+2:l.rfind('\n')-2])
    return lat, lon, alt
"""

# ROUTE CLASS
# ============    
class Route():
    def __init__(self, 
                 name, 
                 departure_time=None, 
                 cruise_speed=None, 
                 ascent_speed=None, 
                 descent_speed=None, 
                 landing_speed=None, 
                 landing_alt=10):
        self.name             = name
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
        return self.lat, self.lon, self.alt, self.eta
    
    """
    def adjust_eta(self, hours_=0., seconds_=0.0):
        self.departure_time = dt.datetime.now() + dt.timedelta(hours=hours_, seconds=seconds_) 
        timedelta            = self.departure_time - self.eta[0]
        self.eta            = [item + timedelta  for item in self.eta]
        return
    """

    """
    def add_point(self, lat, lon, alt, eta=None):
        self.lat.append(lat)
        self.lon.append(lon)
        self.alt.append(alt)
        if eta:
            self.eta.append(eta)
    """
        
    def set_waypoints(self, lat, lon, alt, eta=None):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        if eta is not None:
            self.eta = eta
        return
        
    """
    def set_waypoints_as_array(self):
        self.lat = np.asarray(self.lat)
        self.lon = np.asarray(self.lon)
        self.alt = np.asarray(self.alt)
        return
    """
        
    def set_landing_waypoints(self, set_landing_eta=False):
        if type(self.alt) == list:
            self.set_waypoints_as_array()
        idx_land     = np.asarray(self.alt < self.landing_altitude) * np.insert(np.sign(np.diff(self.alt)) < 0, 0, False)
        idx_land_pos = np.where(idx_land)[0]
        self.lat     = np.insert(self.lat, idx_land_pos, self.lat[idx_land_pos])
        self.lon     = np.insert(self.lon, idx_land_pos, self.lon[idx_land_pos])
        self.alt     = np.insert(self.alt, idx_land_pos, [self.landing_altitude*1.0, ]*len(idx_land_pos))
        # Interpolate eta at landing waypoints linearly
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

    
    def set_eta(self, eta=None, hovering=0, add_takeoff_time=None, add_landing_time=None):
                
        # Assign ETAS
        # ============
        if eta is not None: # if ETA is provided, assign to self.eta and that's it.
            assert hasattr(eta, "__len__") and len(eta)==len(self.lat), "ETA must be vector array with same length as lat, lon and alt."
            assert isinstance(eta[0], dt.datetime), "ETA vector must be composed of datetime objects."
            self.eta = eta
            # self.set_landing_waypoints(set_landing_eta=True)
        else:   # if ETA is not provided, compute it from desired cruise speed and other speeds
            assert self.cruise_speed is not None, "If ETA is not provided, desired speed (cruise, ascent, descent) must be provided."
            
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
            
            self.eta = self.compute_etas_from_speed(takeoff_time=self.takeoff_time, landing_time=self.landing_time, hovering=self.hovering)
            # self.eta = [self.departure_time + dt.timedelta(0, self.eta[item]) for item in range(len(self.eta))]
            
        return self.eta


    def compute_etas_from_speed(self, hovering, takeoff_time, landing_time, distance_method='greatcircle', cruise_speed=None, ascent_speed=None, descent_speed=None, assign_eta=True):
        """
        :param cruise_speed:        m/s, cruise speed in between waypoints
        :param ascent_speed:        m/s, ascent speed in between waypoints
        :param descent_speed:       m/s, descent speed in between waypoints
        :param hovering:            s, extra time for hovering in between waypoints
        :param takeoff_time:        s, extra time needed to take off
        :param landing_time:        s, extra time needed to land
        :param distance_method:     string, method used to compute the distance between two points, either 'greatcircle' or 'vincenty'. default = 'greatcircle'
        """
        assert len(self.lat)>2, "Need at least more than 2 way-points to compute ETAS with function compute_etas."
        
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

            if dv> 0:                                                       vert_speed = ascent_speed[point]
            elif dv < 0 and alt_for_land[point] > self.landing_altitude:    vert_speed = descent_speed[point]
            elif dv < 0 and alt_for_land[point] <= self.landing_altitude:   vert_speed = self.landing_speed[point]
            else:                                                           vert_speed = 0. # not moving vertically.
            # d_eta[point] = max([dh / cruise_speed[point], abs(dv / vert_speed)])
            d_eta[point] = np.sqrt( (dh**2.0 + dv**2.0) / (cruise_speed[point]**2.0 + vert_speed**2.0))
            if hovering[point] != 0:    d_eta[point] += hovering[point]

        d_eta[0]  += takeoff_time
        d_eta[-1] += landing_time
        eta_array = np.asarray(np.cumsum(np.insert(d_eta, 0, 0.0)))
        if assign_eta:  self.eta = [dt.datetime.fromtimestamp(eta_array[ii] + + self.departure_time.timestamp()) for ii in range(len(eta_array))]
        return self.eta