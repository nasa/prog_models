# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

import chaospy as cp
import numpy as np
from prog_models.models import BatteryElectroChemEOD
from prog_models.data_models import PCE
import scipy as sp

DT = 0.1

# Build Model
m = BatteryElectroChemEOD() 
m2 = PCE.from_model(m, {'i': cp.Uniform(0, 8)}, dt=DT, max_time = 4000, N=100)

# Test
results = []
gt = []
x0 = m.initialize()

def future_loading(t, x=None):
    return m.InputContainer(interpolator(t)[np.newaxis].T)

for i in range(100):
    samples = m2.parameters['J'].sample(size=m2.parameters['discretization'], rule='latin_hypercube')
    interpolator = sp.interpolate.interp1d(m2.parameters['times'], samples)
    
    gt.append(m.time_of_event(x0, future_loading, dt = DT)['EOD'])
    results.append(m2.time_of_event(x0, future_loading)['EOD'])
    print(results[-1], gt[-1])

# Plot
import matplotlib.pyplot as plt
plt.scatter(gt, results)
max_val = max(max(gt), max(results))
plt.plot([0, max_val], [0, max_val], 'k--')
plt.xlabel("Ground Truth (s)")
plt.ylabel("PCE (s)")
plt.show()
