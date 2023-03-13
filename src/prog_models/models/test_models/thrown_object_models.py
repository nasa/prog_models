# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

import numpy as np

from prog_models import LinearModel
from prog_models.models.thrown_object import ThrownObject



class defaultParams(ThrownObject):
    times = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    inputs = [{}]*9
    outputs = [
        {'x': 1.83},
        {'x': 36.95},
        {'x': 62.36},
        {'x': 77.81},
        {'x': 83.45},
        {'x': 79.28},
        {'x': 65.3},
        {'x': 41.51},
        {'x': 7.91},
    ]
    thrower_height = 20
    

class wrongTimeValues(defaultParams):
    times = [0 , 2, 3, 4, 5, 1, 6, 7, 8]

class wrongTimeLength(defaultParams):
    times = [0, 1, 2, 3, 4, 5, 6, 7]

class wrongInputStorage(defaultParams):
    inputs = [{}]*10

class wrongInputStorage2(defaultParams):
    inputs = [{}]*8

class wrongOutPuts(defaultParams):
    outputs = [
        {'x': 7.91}    
    ]


