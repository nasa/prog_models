

import io
import numpy as np
import datetime as dt
import scipy.io as inpout
from prog_models.models.uav_model.trajectory.route import Route # , read_routes
from prog_models.models.uav_model.utilities import loadsavewrite_utils as loadsave

from prog_models.exceptions import ProgModelInputException

DEG2RAD = np.pi/180.0
FEET2MET = 0.3048

def get_flightplan(fname, **kwargs):
    """
    Flight plan in .txt file organized as follows:
    columns: latitude (deg), longitude (deg), altitude (feet), time (unix)
    rows:   header first row, data from second row onward
    
    :param fname:         str, file path including file name and txt extension
    :param  kwargs:       options for numpy loadtxt function: skiprows (default=1), comments (default #), max_rows (default None)
    :return:              flight plan dictionary with keys: lat, lon, alt, time_unix, time_stamp, name (i.e., filename).
    """
    
    params = dict(skiprows=1, comments='#', max_rows=None)
    params.update(kwargs)

    if fname[fname.rfind('.')+1:] == 'txt':
        lat, lon, alt, time_unix, timestamps = loadsave.load_traj_from_txt(fname, params['skiprows'], params['comments'], params['max_rows'])
    elif fname[fname.rfind('.')+1:] == 'mat':
        lat, lon, alt, time_unix, timestamps = loadsave.load_traj_from_mat_file(fname)
    
    # If no time stamp was available from file, add current time stamp and corresponding unix time.
    if timestamps is None or time_unix is None:
        timestamps = [dt.datetime.now()]
        time_unix  = [timestamps[0].timestamp()]
    
    flightplan_ = {'lat': lat, 'lon': lon, 'alt': alt, 'time_unix': time_unix, 'timestamp': timestamps, 'name': fname}
    return flightplan_

def convert_dict_inputs(input_dict):
    # Check units and return warnings if incorrect:
    if 'lat_deg' not in input_dict.keys() and 'lat_rad' not in input_dict.keys():
        raise ProgModelInputException("Waypoints latitude must be defined in degrees (with lat_deg) or radians (with lat_rad).")
    elif 'lon_deg' not in input_dict.keys() and 'lon_rad' not in input_dict.keys():
        raise ProgModelInputException("Waypoints longitude must be defined in degrees (with lon_deg) or radians (with lon_rad).")
    elif 'alt_ft' not in input_dict.keys() and 'alt_m' not in input_dict.keys():
        raise ProgModelInputException("Waypoints altitude must be defined in feet (with alt_ft) or meters (with alt_m).")
    if len(input_dict.keys()) > 3 and 'time_unix' not in input_dict.keys():
        raise ProgModelInputException("Waypoints input incorrectly. Use lat_deg, lon_deg, alt_ft, and time_unix to specify.")
    
    # Convert, if necessary
    if 'lat_deg' in input_dict.keys():
        lat = input_dict['lat_deg'] * DEG2RAD
        lon = input_dict['lon_deg'] * DEG2RAD
    else: 
        lat = input_dict['lat_rad']
        lon = input_dict['lon_rad']
    if 'alt_ft' in input_dict.keys():
        alt = input_dict['alt_ft'] * FEET2MET
    else: 
        alt = input_dict['alt_m']
        
    if 'time_unix' in input_dict.keys():
        time_unix = input_dict['time_unix']
        timestamps = [dt.datetime.fromtimestamp(time_unix[ii]) for ii in range(len(time_unix))]
    else: 
        # If no time stamp was available from file, add current time stamp and corresponding unix time.
        timestamps = [dt.datetime.now()]
        time_unix  = [timestamps[0].timestamp()]

    return {'lat_rad': lat, 'lon_rad': lon, 'alt_m': alt, 'timestamp': timestamps, 'time_unix': time_unix}


# LARC flight loading function
# ============================
"""
def larc_flight(file='data/LARC_data.mat'):
    # d = io.loadmat(file)
    d = inpout.loadmat(file)
    lat = d['waypoints'][0][0][0][0] * DEG2RAD
    lon = d['waypoints'][0][0][1][0] * DEG2RAD
    alt = d['waypoints'][0][0][2][0]
    eta = d['waypoints'][0][0][3][0]

    sd = np.insert(np.diff(lat) + np.diff(lon) + np.diff(alt), 0, 0.)
    lat = lat[sd!=0]
    lon = lon[sd!=0]
    alt = alt[sd!=0]
    eta = eta[sd!=0]

    route = Route(name='LARC flight', departure_time=dt.datetime.fromtimestamp(eta[0]))
    route.lat = lat
    route.lon = lon
    route.alt = alt
    route.eta = [dt.datetime.fromtimestamp(eta[ii]) for ii in range(len(eta))]
    return route


def load_SFBA_routes(cruise_speeds=[30, 30, 25], ascent_speeds=[5.0, 5.0, 3.0], descent_speeds=[5.0, 5.0, 3.0], 
                     landing_speeds=[1.5, 1.5, 1.5], takeofftimes=[60.0, 60.0, 60.0], landingtimes=[60.0, 60.0, 60.0]):
    route_filename = 'data/SF_SJ_OAK_routes.txt'
    routes         = read_routes(route_filename, ft2m=True)
    route1, route2, route3 = routes
    
    route1.cruise_speed  = cruise_speeds[0]
    route1.ascent_speed  = ascent_speeds[0]
    route1.descent_speed = descent_speeds[0]
    route1.landing_speed = landing_speeds[0]

    route2.cruise_speed  = cruise_speeds[1]
    route2.ascent_speed  = ascent_speeds[1]
    route2.descent_speed = descent_speeds[1]
    route2.landing_speed = landing_speeds[1]

    route3.cruise_speed  = cruise_speeds[2]
    route3.ascent_speed  = ascent_speeds[2]
    route3.descent_speed = descent_speeds[2]
    route3.landing_speed = landing_speeds[2]
    
    route1.set_eta(eta=None, add_takeoff_time=takeofftimes[0], add_landing_time=landingtimes[0])
    route2.set_eta(eta=None, add_takeoff_time=takeofftimes[1], add_landing_time=landingtimes[1])
    route3.set_eta(eta=None, add_takeoff_time=takeofftimes[2], add_landing_time=landingtimes[2])

    return route1, route2, route3


def get_small_drone_flight_route():
    # Small drone flight
    lat, lon, alt, _, tstamps = get_flightplan()
    # Add fictitious hover
    hover = np.zeros((len(lat)-1,))
    hover[(np.diff(lon)==0) * (np.diff(lat)==0)] = 20.0   # add 20 seconds to all hovering points

    # Generate Route from flight plan
    route = Route('traj1')
    route.set_waypoints(lat, lon, alt) 
    route.set_eta(tstamps[0],  cruise_speed=6, ascent_speed=3, descent_speed=3,
                  hover=hover, add_takeoff_time=40, add_landing_time=40)
    return route
"""
