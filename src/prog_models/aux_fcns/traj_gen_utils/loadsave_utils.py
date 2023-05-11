# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Loading utility functions
"""

# IMPORTS
# =========
import numpy as np
import datetime as dt

# AUXILIARY CONVERSION
# ====================
DEG2RAD  = np.pi / 180.0
RAD2DEG  = 1.0 / DEG2RAD
FEET2MET = 0.3048
MET2FEET = 1.0 / FEET2MET


# FUNCTIONS
# ==========
# Trajectory Loading functions
# ==============================
def load_traj_from_txt(fname, skiprows=1, comments='#', max_rows=None):
    # Check units and return warnings if incorrect:
    data_id = np.loadtxt(fname=fname, dtype='str', max_rows=1)
    if 'lat_deg' not in data_id and 'lat_rad' not in data_id:
        raise TypeError("Waypoints latitude must be defined in degrees (with 'lat_deg') or radians (with 'lat_rad').")
    if 'lon_deg' not in data_id and 'lon_rad' not in data_id:
        raise TypeError("Waypoints longitude must be defined in degrees (with 'lon_deg') or radians (with 'lon_rad').")
    if 'alt_ft' not in data_id and 'alt_m' not in data_id:
        raise TypeError("Waypoints altitude must be defined in feet (with 'alt_ft') or meters (with 'alt_m').")    
    if not (data_id[0] == 'lat_deg' or data_id[0] == 'lat_rad'):
        raise TypeError("Waypoint latitudes must be the first column in text file.")
    if not (data_id[1] == 'lon_deg' or data_id[1] == 'lon_rad'):
        raise TypeError("Waypoint longitudes must be the second column in text file.") 
    if not (data_id[2] == 'alt_m' or data_id[2] == 'alt_ft'):
        raise TypeError("Waypoint altitudes must be the third column in text file.")
    if data_id.shape[0] > 3 and data_id[3] != 'time_unix':
        raise TypeError("ETAs must be defined in unix time (with 'time_unix').")
    if data_id.shape[0] > 4:
        raise TypeError("Too much waypoint information provided. Only latitude, longitude, altitude, and time is accepted.")

    # Convert, if necessary
    d = np.loadtxt(fname=fname, skiprows=skiprows, comments=comments, max_rows=max_rows)
    if 'lat_deg' in data_id:
        lat = d[:, 0] * DEG2RAD  # covert deg 2 rad
    elif 'lat_rad' in data_id: 
        lat = d[:, 0]
    if 'lon_deg' in data_id:
        lon = d[:, 1] * DEG2RAD  # covert deg 2 rad
    elif 'lon_rad' in data_id: 
        lon = d[:, 1]
    if 'alt_ft' in data_id:
        alt = d[:, 2] * FEET2MET  # covert feet 2 meters
    elif 'alt_m' in data_id: 
        alt = d[:, 2]

    if d.shape[1] > 3:
        time_unix = d[:, -1]
        timestamps = [dt.datetime.fromtimestamp(time_unix[ii]) for ii in range(len(time_unix))]
    else:
        time_unix = None
        timestamps = None
    return lat, lon, alt, time_unix, timestamps
    