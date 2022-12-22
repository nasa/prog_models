"""Loading utility functions"""

# IMPORTS
# =========
import numpy as np
import datetime as dt
import scipy.io as inpout

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
    d = np.loadtxt(fname=fname, skiprows=skiprows, comments=comments, max_rows=max_rows)
    lat = d[:, 0] * DEG2RAD  # covert deg 2 rad
    lon = d[:, 1] * DEG2RAD  # covert deg 2 rad
    alt = d[:, 2] * FEET2MET  # covert feet 2 meters
    if d.shape[1] > 3:
        time_unix = d[:, -1]
        timestamps = [dt.datetime.fromtimestamp(time_unix[ii]) for ii in range(len(time_unix))]
    else:
        time_unix = None
        timestamps = None
    return lat, lon, alt, time_unix, timestamps


def load_traj_from_mat_file(fname):
    d = inpout.loadmat(fname)
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

    timestamps = [datetime_0 + dt.timedelta(seconds=eta[ii] - eta[0]) for ii in range(len(eta))]
    return lat.reshape((-1,)), lon.reshape((-1,)), alt.reshape((-1,)), eta, timestamps
