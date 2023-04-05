# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections.abc import Iterable
import numpy as np
from warnings import warn


def MAX_E(m, times, inputs, outputs, **kwargs):
    """
    Calculate the Maximum Error between model behavior and some collected data.

    Args:
        m (PrognosticsModel): Model to use for comparison
        times (list[float]): array of times for each sample
        inputs (list[dict]): array of input dictionaries where input[x] corresponds to time[x]
        outputs (list[dict]): array of output dictionaries where output[x] corresponds to time[x]

    Keyword Args:
        x0 (StateContainer): Current State of the model
        dt (float, optional): Minimum time step in simulation. Defaults to 1e99.

    Returns:
        float: Maximum error between model and data
    """
    if isinstance(times[0], Iterable):
        # Calculate error for each
        error = [MAX_E(t, i, z, **kwargs) for (t, i, z) in zip(times, inputs, outputs)]
        return max(error)

    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)

    if not isinstance(x, m.StateContainer):
        x = [m.StateContainer(x_i) for x_i in x]

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]
    
    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    counter = 0 
    t_last = times[0]
    err_max = 0
    z_obs = m.output(x)  # Initialize
    for t, u, z in zip(times, inputs, outputs):
        while t_last < t:
            t_new = min(t_last + dt, t)
            x = m.next_state(x, u, t_new-t_last)
            t_last = t_new
            if t >= t_last:
                # Only recalculate if required
                z_obs = m.output(x)
        if not (None in z_obs.matrix or None in z.matrix):
            if any(np.isnan(z_obs.matrix)):
                warn(f"Model unstable- NaN reached in simulation (t={t})")
                break
            err_max = max(err_max, np.max(
                np.abs(z.matrix - z_obs.matrix)
            ))
            counter += 1

    if counter == 0:
        return np.nan

    return err_max


def RMSE(m, times, inputs, outputs, **kwargs):
    """
    Calculate the Root Mean Squared Error between model behavior and some collected data.

    Args:
        m (PrognosticsModel): Model to use for comparison
        times (list[float]): array of times for each sample
        inputs (list[dict]): array of input dictionaries where input[x] corresponds to time[x]
        outputs (list[dict]): array of output dictionaries where output[x] corresponds to time[x]

    Keyword Args:
        x0 (StateContainer): Current State of the model
        dt (float, optional): Minimum time step in simulation. Defaults to 1e99.

    Returns:
        float: RMSE between model and data
    """
    return np.sqrt(MSE(m, times, inputs, outputs, **kwargs))


def MSE(m, times, inputs, outputs, **kwargs):
    """
    Calculate the Mean Squared Error between model behavior and some collected data.

    Args:
        m (PrognosticsModel): Model to use for comparison
        times (list[float]): array of times for each sample
        inputs (list[dict]): array of input dictionaries where input[x] corresponds to time[x]
        outputs (list[dict]): array of output dictionaries where output[x] corresponds to time[x]

    Keyword Args:
        x0 (StateContainer): Current State of the model
        dt (float, optional): Minimum time step in simulation. Defaults to 1e99.

    Returns:
        float: MSE between model and data
    """
    if isinstance(times[0], Iterable):
        # Calculate error for each
        error = [MSE(t, i, z, **kwargs) for (t, i, z) in zip(times, inputs, outputs)]
        return sum(error)/len(error)

    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)

    if not isinstance(x, m.StateContainer):
        x = [m.StateContainer(x_i) for x_i in x]

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]
    
    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    counter = 0  # Needed to account for skipped (i.e., none) values
    t_last = times[0]
    err_total = 0
    z_obs = m.output(x)  # Initialize
    for t, u, z in zip(times, inputs, outputs):
        while t_last < t:
            t_new = min(t_last + dt, t)
            x = m.next_state(x, u, t_new-t_last)
            t_last = t_new
            if t >= t_last:
                # Only recalculate if required
                z_obs = m.output(x)
        if not (None in z_obs.matrix or None in z.matrix):
            if any(np.isnan(z_obs.matrix)):
                warn(f"Model unstable- NaN reached in simulation (t={t})")
                break
            err_total += np.sum(
                np.square(z.matrix - z_obs.matrix),
                where=~np.isnan(z.matrix))
            counter += 1

    return err_total/counter


def MAE(m, times, inputs, outputs, **kwargs):
    """
    Calculate the Mean Absolute Error between model behavior and some collected data.

    Args:
        m (PrognosticsModel): Model to use for comparison
        times (list[float]): array of times for each sample
        inputs (list[dict]): array of input dictionaries where input[x] corresponds to time[x]
        outputs (list[dict]): array of output dictionaries where output[x] corresponds to time[x]

    Keyword Args:
        x0 (StateContainer): Current State of the model
        dt (float, optional): Minimum time step in simulation. Defaults to 1e99.

    Returns:
        float: MAE between model and data
    """
    if isinstance(times[0], Iterable):
        # Calculate error for each
        error = [MAE(t, i, z, **kwargs) for (t, i, z) in zip(times, inputs, outputs)]
        return sum(error)/len(error)

    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)

    if not isinstance(x, m.StateContainer):
        x = [m.StateContainer(x_i) for x_i in x]

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]
    
    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    counter = 0  # Needed to account for skipped (i.e., none) values
    t_last = times[0]
    err_total = 0
    z_obs = m.output(x)  # Initialize
    for t, u, z in zip(times, inputs, outputs):
        while t_last < t:
            t_new = min(t_last + dt, t)
            x = m.next_state(x, u, t_new-t_last)
            t_last = t_new
            if t >= t_last:
                # Only recalculate if required
                z_obs = m.output(x)
        if not (None in z_obs.matrix or None in z.matrix):
            if any(np.isnan(z_obs.matrix)):
                warn(f"Model unstable- NaN reached in simulation (t={t})")
                break
            err_total += np.sum(
                np.abs(z.matrix - z_obs.matrix))
            counter += 1
    return err_total/counter

def MAPE(m, times, inputs, outputs, **kwargs):
    """
    Calculate the Mean Absolute Percentage Error between model behavior and some collected data.

    Args:
        m (PrognosticsModel): Model to use for comparison
        times (list[float]): array of times for each sample
        inputs (list[dict]): array of input dictionaries where input[x] corresponds to time[x]
        outputs (list[dict]): array of output dictionaries where output[x] corresponds to time[x]

    Keyword Args:
        x0 (StateContainer): Current State of the model
        dt (float, optional): Minimum time step in simulation. Defaults to 1e99.

    Returns:
        float: MAPE between model and data
    """
    if isinstance(times[0], Iterable):
        # Calculate error for each
        error = [MAPE(t, i, z, **kwargs) for (t, i, z) in zip(times, inputs, outputs)]
        return sum(error)/len(error)

    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)

    if not isinstance(x, m.StateContainer):
        x = [m.StateContainer(x_i) for x_i in x]

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]
    
    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    counter = 0  # Needed to account for skipped (i.e., none) values
    t_last = times[0]
    err_total = 0
    z_obs = m.output(x)  # Initialize
    for t, u, z in zip(times, inputs, outputs):
        while t_last < t:
            t_new = min(t_last + dt, t)
            x = m.next_state(x, u, t_new-t_last)
            t_last = t_new
            if t >= t_last:
                # Only recalculate if required
                z_obs = m.output(x)
        if not (None in z_obs.matrix or None in z.matrix):
            if any(np.isnan(z_obs.matrix)):
                warn(f"Model unstable- NaN reached in simulation (t={t})")
                break
            err_total += np.sum(
                np.abs(z.matrix - z_obs.matrix)/z)
            counter += 1
    return err_total/counter
