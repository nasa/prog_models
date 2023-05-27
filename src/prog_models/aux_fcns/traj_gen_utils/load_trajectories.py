# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Functions to extract user-defined waypoint information and convert to appropriate units for trajectory generation
"""
# Imports 
import datetime as dt
import numpy as np

# Conversion values 
DEG2RAD = np.pi/180.0
FEET2MET = 0.3048

def convert_df_inputs(input_df):
    """
    Flight plan is input by user in pandas dataframe format. 
    Column headers must contain: latitude ('lat_deg' or 'lat_rad'), longitude ('lon_deg' or 'lon_rad'), altitude ('alt_ft' or 'alt_m'), and (optional) time ('time_unix')
    
    :param input_df:      pandas dataframe with columns for waypoints latitude, longitude, altitude, and optional time        
    :return:              flight plan dictionary with keys: lat (in rad), lon (in rad), alt (in m), time_unix, time_stamp.
    """

    # Check units and return warnings if incorrect:
    if 'lat_deg' not in input_df.columns and 'lat_rad' not in input_df.columns:
        raise TypeError("Waypoints latitude must be defined in degrees (with lat_deg) or radians (with lat_rad).")
    elif 'lon_deg' not in input_df.columns and 'lon_rad' not in input_df.columns:
        raise TypeError("Waypoints longitude must be defined in degrees (with lon_deg) or radians (with lon_rad).")
    elif 'alt_ft' not in input_df.columns and 'alt_m' not in input_df.columns:
        raise TypeError("Waypoints altitude must be defined in feet (with alt_ft) or meters (with alt_m).")
    if len(input_df.columns) > 3 and 'time_unix' not in input_df.columns:
        raise TypeError("Waypoints input incorrectly. Use lat_deg, lon_deg, alt_ft, and time_unix to specify.")

    # Convert units, if necessary
    if 'lat_deg' in input_df.columns:
        lat = (input_df['lat_deg'] * DEG2RAD).to_numpy()
        lon = (input_df['lon_deg'] * DEG2RAD).to_numpy()
    else: 
        lat = (input_df['lat_rad']).to_numpy()
        lon = (input_df['lon_rad']).to_numpy()
    if 'alt_ft' in input_df.columns:
        alt = (input_df['alt_ft'] * FEET2MET).to_numpy()
    else: 
        alt = (input_df['alt_m']).to_numpy()
        
    if 'time_unix' in input_df.columns:
        time_unix = (input_df['time_unix']).to_numpy()
        timestamps = [dt.datetime.fromtimestamp(time_unix[ii]) for ii in range(len(time_unix))]
    else: 
        # If no time stamp was available from file, add current time stamp and corresponding unix time.
        timestamps = [dt.datetime.now()]
        time_unix  = [timestamps[0].timestamp()]

    return {'lat_rad': lat, 'lon_rad': lon, 'alt_m': alt, 'timestamp': timestamps, 'time_unix': time_unix}
   