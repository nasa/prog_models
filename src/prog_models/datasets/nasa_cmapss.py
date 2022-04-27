# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import io
import numpy as np
import requests
import zipfile

cache = None
URL = "https://ti.arc.nasa.gov/c/6/"


def load_data(dataset_id : int) -> tuple:
    """
    Loads data for one CMAPSS trajectory from NASA's PCoE Dataset
    https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

    Data Set: 1
        Train trjectories: 100
        Test trajectories: 100
        Conditions: ONE (Sea Level)
        Fault Modes: ONE (HPC Degradation)

    Data Set: 2
        Train trjectories: 260
        Test trajectories: 259
        Conditions: SIX 
        Fault Modes: ONE (HPC Degradation)

    Data Set: 3
        Train trjectories: 100
        Test trajectories: 100
        Conditions: ONE (Sea Level)
        Fault Modes: TWO (HPC Degradation, Fan Degradation)

    Data Set: 4
        Train trjectories: 248
        Test trajectories: 249
        Conditions: SIX 
        Fault Modes: TWO (HPC Degradation, Fan Degradation)

    DAta sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine ñ i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

    The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

    Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, ìDamage Propagation Modeling for Aircraft Engine Run-to-Failure Simulationî, in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

    Args:
        dataset_id (int): Dataset id

    Raises:
        ValueError: Data not in dataset (should be 1-4)

    Returns:
        tuple[np.array, np.array, np.array]: (train, test, results)
            0: Training data from dataset,
            1: Testing data from dataset
            2: Time of end of life 
    
    The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:
        1)	unit number
        2)	time, in cycles
        3)	operational setting 1
        4)	operational setting 2
        5)	operational setting 3
        6)	sensor measurement  1
        7)	sensor measurement  2
        ...
        26)	sensor measurement  26
    """
    global cache

    if dataset_id not in range(1, 5):
        raise ValueError(f"Invalid dataset id {dataset_id}")

    dataset_id = f"FD0{dataset_id:02d}"
    if cache is None:
        # Download data
        try:
            response = requests.get(URL, allow_redirects=True)
        except requests.exceptions.RequestException as e: # handle chain of errors
            raise ConnectionError("Data download failed. This may be because of issues with your internet connection or the datasets may have moved. Please check your internet connection and make sure you're using the latest version of prog_models.")

        # Unzip response
        cache = zipfile.ZipFile(io.BytesIO(response.content))

    # Read Files
    with cache.open(f'test_{dataset_id}.txt', mode='r') as f:
        with io.BufferedReader(f) as f2:
            test = np.loadtxt(f2)

    with cache.open(f'train_{dataset_id}.txt', mode='r') as f:
        with io.BufferedReader(f) as f2:
            train = np.loadtxt(f2)

    with cache.open(f'RUL_{dataset_id}.txt', mode='r') as f:
        with io.BufferedReader(f) as f2:
            rul = np.loadtxt(f2)

    # Return results 
    return (test, train, rul)

def clear_cache() -> None:
    """
    Clears the cache of downloaded data
    """
    global cache
    cache = None
