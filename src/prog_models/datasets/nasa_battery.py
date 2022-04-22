# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import io
import requests
import numpy as np
from scipy.io import loadmat
import zipfile

# Map of battery to url for data
urls = {
    'RW1': "https://ti.arc.nasa.gov/c/27/",
    'RW2': "https://ti.arc.nasa.gov/c/27/",
    'RW3': "https://ti.arc.nasa.gov/c/26/",
    'RW4': "https://ti.arc.nasa.gov/c/26/",
    'RW5': "https://ti.arc.nasa.gov/c/26/",
    'RW6': "https://ti.arc.nasa.gov/c/26/",
    'RW7': "https://ti.arc.nasa.gov/c/27/",
    'RW8': "https://ti.arc.nasa.gov/c/27/",
    'RW9': "https://ti.arc.nasa.gov/c/25/",
    'RW10': "https://ti.arc.nasa.gov/c/25/",
    'RW11': "https://ti.arc.nasa.gov/c/25/",
    'RW12': "https://ti.arc.nasa.gov/c/25/",
    'RW13': "https://ti.arc.nasa.gov/c/31/",
    'RW14': "https://ti.arc.nasa.gov/c/31/",
    'RW15': "https://ti.arc.nasa.gov/c/31/",
    'RW16': "https://ti.arc.nasa.gov/c/31/",
    'RW17': "https://ti.arc.nasa.gov/c/29/",
    'RW18': "https://ti.arc.nasa.gov/c/29/",
    'RW19': "https://ti.arc.nasa.gov/c/29/",
    'RW20': "https://ti.arc.nasa.gov/c/29/",
    'RW21': "https://ti.arc.nasa.gov/c/30/",
    'RW22': "https://ti.arc.nasa.gov/c/30/",
    'RW23': "https://ti.arc.nasa.gov/c/30/",
    'RW24': "https://ti.arc.nasa.gov/c/30/",
    'RW25': "https://ti.arc.nasa.gov/c/28/",
    'RW26': "https://ti.arc.nasa.gov/c/28/",
    'RW27': "https://ti.arc.nasa.gov/c/28/",
    'RW28': "https://ti.arc.nasa.gov/c/28/",
}

cache = {}  # Cache for downloaded data
# Cache is used to prevent files from being downloaded twice

def load_data(batt_id):
    """Loads data for one or more batteries from NASA's PCoE Dataset
    https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

    Args:
        batt_id (str): Battery name from dataset (RW1-28)

    Raises:
        ValueError: Battery not in dataset (should be RW1-28)

    Returns:
        tuple[dict, list[np.array]]: (description, data)
            0: description of the data. 
            1: List of numpy arrays (order 2 tensors) such that data[i] is the data for run i, corresponding with details[i], above. Each element is in the format [time_index][value] where values are ('relativeTime', 'current' (amps), 'voltage', 'temperature' (°C)) in that order.
    """
    if isinstance(batt_id, int):
        # Convert to string
        batt_id = 'RW' + str(batt_id)
    if not isinstance(batt_id, str):
        raise ValueError('Battery ID must be a string')

    if batt_id not in urls:
        raise ValueError('Unknown battery ID: {}'.format(batt_id))

    url = urls[batt_id]

    if url not in cache:
        # Download data
        try:
            response = requests.get(url, allow_redirects=True)
        except requests.exceptions.RequestException as e: # handle chain of errors
            raise ConnectionRefusedError("Data download failed. This may be because of issues with your internet connection or the datasets may have moved. Please check your internet connection and make sure you're using the latest version of prog_models.")

        # Unzip response
        cache[url] = zipfile.ZipFile(io.BytesIO(response.content))

    f = cache[url].open(f'{cache[url].infolist()[0].filename}Matlab/{batt_id}.mat')

    # Load matlab file
    result = loadmat(f)['data']

    # Reformat
    desc = {
        'procedure': str(result['procedure'][0,0][0]),
        'description': str(result['description'][0,0][0]),
        'runs': 
        [
            {
                'type': str(run_type[0]), 
                'desc': str(desc[0]),
                'date': str(date[0])
            } for (run_type, desc, date) in zip(result['step'][0,0]['type'][0], result['step'][0,0]['comment'][0], result['step'][0,0]['date'][0])
        ]
    }

    result = result['step'][0,0]
    result = [
        np.array([
            result[key][0, i][0] for key in ('relativeTime', 'current', 'voltage', 'temperature')
        ], np.float64).T for i in range(result.shape[1])
    ]

    return desc, result

def clear_cache():
    """Clears the cache of downloaded data"""
    cache.clear()
