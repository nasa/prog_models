# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Functions for aerodynamics effects
"""

import numpy as np

# Aerodynamic forces
# ====================
class DragModel():
    def __init__(self, bodyarea=None, Cd=None, air_density=None):
        self.area = bodyarea
        self.cd = Cd
        self.rho = air_density
        return

    def __call__(self, air_v):
        vsq = air_v * np.abs(air_v)
        return 0.5 * self.rho * self.area * self.cd * vsq
 