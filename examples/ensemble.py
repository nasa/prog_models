# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from matplotlib import pyplot as plt
import numpy as np
from prog_models import EnsembleModel
from prog_models.datasets import nasa_battery
from prog_models.models import BatteryElectroChemEOD, BatteryCircuit

# Download data
print('downloading data (this may take a while)...')
data = nasa_battery.load_data(8)[1]

# Prepare data
RUN_ID = 0
test_input = [{'i': i} for i in data[RUN_ID]['current']]
test_time = data[RUN_ID]['relativeTime']
test_output = [{'v': v, 't': t} for v, t in zip(data[RUN_ID]['voltage'], data[RUN_ID]['temperature'])]

# Setup physics-based models
print('Setting up models...')
m_electro = BatteryElectroChemEOD(process_noise = 0, measurement_noise = 0)
m_circuit = BatteryCircuit(process_noise = 0, measurement_noise = 0)
m_ensemble = EnsembleModel((m_electro, m_circuit), process_noise = 0, measurement_noise=0)

# Evaluate models
print('Evaluating models...')
def future_loading(t, x=None):
    for i, mission_time in enumerate(test_time):
        if mission_time > t:
            return m_electro.InputContainer(test_input[i])
print('\tEnsemble')
results_ensemble = m_ensemble.simulate_to(test_time.iloc[-1], future_loading)
print('\tElectrochem')
results = m_electro.simulate_to(test_time.iloc[-1], future_loading)
print('\tCircuit')
results_circuit = m_circuit.simulate_to(test_time.iloc[-1], future_loading)

# Plot results
print('Producing figures...')
plt.plot(test_time, data[RUN_ID]['voltage'], color='green', label='ground truth')
plt.plot(results.times, [z['v'] for z in results.outputs], color='blue', label='electro chem')
plt.plot(results_circuit.times, [z['v'] for z in results_circuit.outputs], color='red', label='circuit')
plt.plot(results_ensemble.times, [z['v'] for z in results_ensemble.outputs], color='yellow', label='ensemble')
plt.legend()
