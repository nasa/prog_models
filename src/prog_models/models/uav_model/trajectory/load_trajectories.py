# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Functions to extract user-defined waypoint information and convert to appropriate units for trajectory generation
"""
# Imports 
import numpy as np
import datetime as dt
from prog_models.models.uav_model.utilities import loadsavewrite_utils as loadsave
from prog_models.exceptions import ProgModelInputException

DEG2RAD = np.pi/180.0
FEET2MET = 0.3048

def get_flightplan(fname, **kwargs):
    """
    Flight plan is input by user in text file format. 
    Flight plan in .txt file must be organized as follows (with the following column headers):
    columns: latitude (lat_deg or lat_rad), longitude (lon_deg or lon_rad), altitude (alt_ft or alt_m), and (optional) time (time_unix)
    rows:   header first row, data from second row onward
    
    :param fname:         str, file path including file name and txt extension
    :param  kwargs:       options for numpy loadtxt function: skiprows (default=1), comments (default #), max_rows (default None)
    :return:              flight plan dictionary with keys: lat (in rad), lon (in rad), alt (in m), time_unix, time_stamp, name (i.e., filename).
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
    """
    Flight plan is input by user in dictionary format. 
    Dictionary must contain keys for: latitude ('lat_deg' or 'lat_rad'), longitude ('lon_deg' or 'lon_rad'), altitude ('alt_ft' or 'alt_m'), and (optional) time ('time_unix')
    Each dictionary key must have a corresponding numpy array of the appropriate values 
    
    :param input_dict:    dictionary with waypoints latitude, longitude, altitude, and optional time defined as numpy arrays        
    :return:              flight plan dictionary with keys: lat (in rad), lon (in rad), alt (in m), time_unix, time_stamp.
    """

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
