# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np

# ---------------------------
# Measurement Noise Functions
# ---------------------------
def uniform_measurement_noise(self, z : dict):
    return self.OutputContainer(z.matrix + np.random.uniform(-1*self.parameters['measurement_noise'].matrix, self.parameters['measurement_noise'].matrix, size=z.matrix.shape))

def triangular_measurement_noise(self, z : dict):
    return self.OutputContainer(z.matrix + np.random.triangular(-1*self.parameters['measurement_noise'].matrix, 0, self.parameters['measurement_noise'].matrix, size=z.matrix.shape))

def normal_measurement_noise(self, z : dict):
    return self.OutputContainer(z.matrix + np.random.normal(0, self.parameters['measurement_noise'].matrix, size=z.matrix.shape))

def no_measurement_noise(self, z : dict) -> dict:
    return z

measurement_noise_functions = {
    'uniform': uniform_measurement_noise,
    'triangular': triangular_measurement_noise,
    'normal': normal_measurement_noise,
    'gaussian': normal_measurement_noise,
    'none': no_measurement_noise,
}

# ---------------------------
# Process Noise Functions
# ---------------------------

def triangular_process_noise(self, x : dict, dt : int =1):
    return self.StateContainer(x.matrix + dt*np.random.triangular(-1*self.parameters['process_noise'].matrix, 0, self.parameters['process_noise'].matrix, size=x.matrix.shape))

def uniform_process_noise(self, x : dict, dt : int =1):
    return self.StateContainer(x.matrix + dt*np.random.uniform(-1*self.parameters['process_noise'].matrix, self.parameters['process_noise'].matrix, size=x.matrix.shape))

def normal_process_noise(self, x : dict, dt : int =1):
    return self.StateContainer(x.matrix + dt*np.random.normal(0, self.parameters['process_noise'].matrix, size=x.matrix.shape))

def no_process_noise(self, x : dict, dt :int =1) -> dict:
    return x

process_noise_functions = {
    'uniform': uniform_process_noise,
    'triangular': triangular_process_noise,
    'normal': normal_process_noise,
    'gaussian': normal_process_noise,
    'none': no_process_noise,
}
