# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np

# ---------------------------
# Measurement Noise Functions
# ---------------------------
def uniform_measurement_noise(self, z : dict):
    return self.OutputContainer({key: z[key] + \
        np.random.uniform(-self.parameters['measurement_noise'][key], self.parameters['measurement_noise'][key], size=None if np.isscalar(z[key]) else len(z[key])) \
            for key in self.outputs}) 

def triangular_measurement_noise(self, z : dict):
    return self.OutputContainer({key: z[key] + \
        np.random.triangular(-self.parameters['measurement_noise'][key], 0, self.parameters['measurement_noise'][key], size=None if np.isscalar(z[key]) else len(z[key])) \
            for key in self.outputs})

def normal_measurement_noise(self, z : dict):
    return self.OutputContainer({key: z[key] \
        + np.random.normal(
            0, self.parameters['measurement_noise'][key],
            size=None if np.isscalar(z[key]) else len(z[key]))
            for key in z.keys()})

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
    return self.StateContainer({key: x[key] + \
        dt*np.random.triangular(-self.parameters['process_noise'][key], 0, self.parameters['process_noise'][key], size=None if np.isscalar(x[key]) else len(x[key])) \
            for key in self.states})

def uniform_process_noise(self, x : dict, dt : int =1):
    return self.StateContainer({key: x[key] + \
        dt*np.random.uniform(-self.parameters['process_noise'][key], self.parameters['process_noise'][key], size=None if np.isscalar(x[key]) else len(x[key])) \
            for key in self.states})

def normal_process_noise(self, x : dict, dt : int =1):
    return self.StateContainer({key: x[key] +
            dt*np.random.normal(
                0, self.parameters['process_noise'][key],
                size=None if np.isscalar(x[key]) else len(x[key]))
                for key in x.keys()})

def no_process_noise(self, x : dict, dt :int =1) -> dict:
    return x

process_noise_functions = {
    'uniform': uniform_process_noise,
    'triangular': triangular_process_noise,
    'normal': normal_process_noise,
    'gaussian': normal_process_noise,
    'none': no_process_noise,
}
