

import io
import numpy as np
import datetime as dt
import scipy.io as inpout
from prog_models.models.trajectory_generation.trajectory.route import Route, read_routes

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
        lat, lon, alt, time_unix, timestamps = load_txt_file(fname, params['skiprows'], params['comments'], params['max_rows'])
    elif fname[fname.rfind('.')+1:] == 'mat':
        lat, lon, alt, time_unix, timestamps = load_mat_file(fname)

    flightplan_ = {'lat': lat, 'lon': lon, 'alt': alt, 'time_unix': time_unix, 'timestamp': timestamps, 'name': fname}
    return flightplan_


def load_txt_file(fname, skiprows=1, comments='#', max_rows=None):
    d          = np.loadtxt(fname=fname, skiprows=skiprows, comments=comments, max_rows=max_rows)
    lat        = d[:, 0] * DEG2RAD      # covert deg 2 rad
    lon        = d[:, 1] * DEG2RAD      # covert deg 2 rad
    alt        = d[:, 2] * FEET2MET     # covert feet 2 meters
    time_unix  = d[:, -1]
    timestamps = [dt.datetime.fromtimestamp(time_unix[ii]) for ii in range(len(time_unix))]
    return lat, lon, alt, time_unix, timestamps

def load_mat_file(fname):
    d   = inpout.loadmat(fname)
    lat = d['waypoints'][0][0][0] * DEG2RAD
    lon = d['waypoints'][0][0][1] * DEG2RAD
    alt = d['waypoints'][0][0][2] * FEET2MET
    eta = d['waypoints'][0][0][3]
    
    eta = eta.astype(float).reshape((-1,))
    if all(name in list(d.keys()) for name in ['cepic_date', 'etime']):
        datetime_0 = dt.datetime.strptime(d['cepic_date'][0] + ' ' + d['etime'][0], '%Y_%m-%d %H:%M:%S')  # 
    elif eta[0] != 0:
        datetime_0 = dt.datetime.fromtimestamp(eta[0])
    else:
        raise Exception('Mat file does not contain date information.')

    timestamps = [datetime_0 + dt.timedelta(seconds=eta[ii]-eta[0]) for ii in range(len(eta))]
    return lat.reshape((-1,)), lon.reshape((-1,)), alt.reshape((-1,)), eta, timestamps


# LARC flight loading function
# ============================
def larc_flight(file='data/LARC_data.mat'):
    d = io.loadmat(file)
    lat = d['waypoints'][0][0][0][0] * np.pi/180.0
    lon = d['waypoints'][0][0][1][0] * np.pi/180.0
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
