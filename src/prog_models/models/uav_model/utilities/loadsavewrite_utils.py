# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Loading utility functions
"""

# IMPORTS
# =========
import numpy as np
import datetime as dt
from prog_models.exceptions import ProgModelInputException

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
        raise ProgModelInputException("Waypoints latitude must be defined in degrees (with 'lat_deg') or radians (with 'lat_rad').")
    if 'lon_deg' not in data_id and 'lon_rad' not in data_id:
        raise ProgModelInputException("Waypoints longitude must be defined in degrees (with 'lon_deg') or radians (with 'lon_rad').")
    if 'alt_ft' not in data_id and 'alt_m' not in data_id:
        raise ProgModelInputException("Waypoints altitude must be defined in feet (with 'alt_ft') or meters (with 'alt_m').")    
    
    # Convert, if necessary
    d = np.loadtxt(fname=fname, skiprows=skiprows, comments=comments, max_rows=max_rows)
    if 'lat_deg' in data_id:
        lat = d[:, 0] * DEG2RAD  # covert deg 2 rad
        lon = d[:, 1] * DEG2RAD  # covert deg 2 rad
    else: 
        lat = d[:, 0]
        lon = d[:, 1]
    if 'alt_ft' in data_id:
        alt = d[:, 2] * FEET2MET  # covert feet 2 meters
    else: 
        alt = d[:, 2]

    if d.shape[1] > 3:
        time_unix = d[:, -1]
        timestamps = [dt.datetime.fromtimestamp(time_unix[ii]) for ii in range(len(time_unix))]
    else:
        time_unix = None
        timestamps = None
    return lat, lon, alt, time_unix, timestamps
    