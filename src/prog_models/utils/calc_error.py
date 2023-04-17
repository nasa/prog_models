# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections.abc import Iterable
from warnings import warn
import math
import numpy as np

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
        stability_tol (double, optional): Configurable parameter.
            Configurable cutoff value, between 0 and 1, that determines the fraction of the data points for which the model must be stable.
            In some cases, a prognostics model will become unstable under certain conditions, after which point the model can no longer represent behavior. 
            stability_tol represents the fraction of the provided argument `times` that are required to be met in simulation, 
            before the model goes unstable in order to produce a valid estimate of mean squared error. 

            If the model goes unstable before stability_tol is met, NaN is returned. 
            Else, model goes unstable after stability_tol is met, the mean squared error calculated from data up to the instability is returned.

    Returns:
        float: Maximum error between model and data
    """
    if isinstance(times[0], Iterable):
        # Calculate error for each
        error = [MAX_E(t, i, z, **kwargs) for (t, i, z) in zip(times, inputs, outputs)]
        return max(error)

    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)
    stability_tol = kwargs.get('stability_tol', 0.95)

    if not isinstance(x, m.StateContainer):
        x = [m.StateContainer(x_i) for x_i in x]

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]
    
    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    # Checks stability_tol is within bounds
    # Throwing a default after the warning.
    if stability_tol >= 1 or stability_tol < 0:
        warn(f"configurable cutoff must be some float value in the domain (0, 1]. Received {stability_tol}. Resetting value to 0.95")
        stability_tol = 0.95

    counter = 0 
    t_last = times[0]
    err_max = 0
    z_obs = m.output(x)  # Initialize
    cutoffThreshold = math.floor(stability_tol * len(times))

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
                if counter < cutoffThreshold:
                    raise ValueError(f"""Model unstable- NAN reached in simulation (t={t}) before cutoff threshold.
                    Cutoff threshold is {cutoffThreshold}, or roughly {stability_tol * 100}% of the data""")                
                else: 
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
        stability_tol (double, optional): Configurable parameter.
            Configurable cutoff value, between 0 and 1, that determines the fraction of the data points for which the model must be stable.
            In some cases, a prognostics model will become unstable under certain conditions, after which point the model can no longer represent behavior. 
            stability_tol represents the fraction of the provided argument `times` that are required to be met in simulation, 
            before the model goes unstable in order to produce a valid estimate of mean squared error. 

            If the model goes unstable before stability_tol is met, NaN is returned. 
            Else, model goes unstable after stability_tol is met, the mean squared error calculated from data up to the instability is returned.

    Returns:
        float: RMSE between model and data
    """
    return np.sqrt(MSE(m, times, inputs, outputs, **kwargs))


def MSE(m, times, inputs, outputs, **kwargs) -> float:
    """Calculate Mean Squared Error (MSE) between simulated and observed

    Args:
        times (list[float]): Array of times for each sample.
        inputs (list[dict]): Array of input dictionaries where input[x] corresponds to time[x].
        outputs (list[dict]): Array of output dictionaries where output[x] corresponds to time[x].

    Keyword Args:
        x0 (dict, optional): Initial state.
        dt (double, optional): Maximum time step.
        stability_tol (double, optional): Configurable parameter.
            Configurable cutoff value, between 0 and 1, that determines the fraction of the data points for which the model must be stable.
            In some cases, a prognostics model will become unstable under certain conditions, after which point the model can no longer represent behavior. 
            stability_tol represents the fraction of the provided argument `times` that are required to be met in simulation, 
            before the model goes unstable in order to produce a valid estimate of mean squared error. 

            If the model goes unstable before stability_tol is met, NaN is returned. 
            Else, model goes unstable after stability_tol is met, the mean squared error calculated from data up to the instability is returned.

    Returns:
        double: Total error
    """
    if isinstance(times[0], Iterable):
        # Calculate error for each
        error = [m.calc_error(t, i, z, **kwargs) for (t, i, z) in zip(times, inputs, outputs)]
        return sum(error)/len(error)

    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)
    stability_tol = kwargs.get('stability_tol', 0.95)

    if not isinstance(x, m.StateContainer):
        x = [m.StateContainer(x_i) for x_i in x]

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]

    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    # Checks stability_tol is within bounds
    # Throwing a default after the warning.
    if stability_tol >= 1 or stability_tol < 0:
        warn(f"configurable cutoff must be some float value in the domain (0, 1]. Received {stability_tol}. Resetting value to 0.95")
        stability_tol = 0.95

    counter = 0  # Needed to account for skipped (i.e., none) values
    t_last = times[0]
    err_total = 0
    z_obs = m.output(x)
    cutoffThreshold = math.floor(stability_tol * len(times))

    for t, u, z in zip(times, inputs, outputs):
        while t_last < t:
            t_new = min(t_last + dt, t)
            x = m.next_state(x, u, t_new-t_last)
            t_last = t_new
            if t >= t_last:
                # Only recalculate if required
                z_obs = m.output(x)
        if not (None in z_obs.matrix or None in z.matrix):
            if any (np.isnan(z_obs.matrix)):
                if counter < cutoffThreshold:
                    raise ValueError(f"Model unstable- NAN reached in simulation (t={t}) before cutoff threshold. "
                                     f"Cutoff threshold is {cutoffThreshold}, or roughly {stability_tol * 100}% of the data")     
                else:
                    warn("Model unstable- NaN reached in simulation (t={})".format(t))
                    break
            err_total += np.sum(np.square(z.matrix - z_obs.matrix), where= ~np.isnan(z.matrix))
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
        stability_tol (double, optional): Configurable parameter.
            Configurable cutoff value, between 0 and 1, that determines the fraction of the data points for which the model must be stable.
            In some cases, a prognostics model will become unstable under certain conditions, after which point the model can no longer represent behavior. 
            stability_tol represents the fraction of the provided argument `times` that are required to be met in simulation, 
            before the model goes unstable in order to produce a valid estimate of mean squared error. 

            If the model goes unstable before stability_tol is met, NaN is returned. 
            Else, model goes unstable after stability_tol is met, the mean squared error calculated from data up to the instability is returned.

    Returns:
        float: MAE between model and data
    """
    if isinstance(times[0], Iterable):
        # Calculate error for each
        error = [MAE(t, i, z, **kwargs) for (t, i, z) in zip(times, inputs, outputs)]
        return sum(error)/len(error)

    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)
    stability_tol = kwargs.get('stability_tol', 0.95)

    if not isinstance(x, m.StateContainer):
        x = [m.StateContainer(x_i) for x_i in x]

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]
    
    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    # Checks stability_tol is within bounds
    # Throwing a default after the warning.
    if stability_tol >= 1 or stability_tol < 0:
        warn(f"configurable cutoff must be some float value in the domain (0, 1]. Received {stability_tol}. Resetting value to 0.95")
        stability_tol = 0.95

    counter = 0  # Needed to account for skipped (i.e., none) values
    t_last = times[0]
    err_total = 0
    z_obs = m.output(x)  # Initialize
    cutoffThreshold = math.floor(stability_tol * len(times))

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
                if counter < cutoffThreshold:
                    raise ValueError(f"""Model unstable- NAN reached in simulation (t={t}) before cutoff threshold.
                    Cutoff threshold is {cutoffThreshold}, or roughly {stability_tol * 100}% of the data""")     
                else:
                    warn("Model unstable- NaN reached in simulation (t={})".format(t))
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
        stability_tol (double, optional): Configurable parameter.
            Configurable cutoff value, between 0 and 1, that determines the fraction of the data points for which the model must be stable.
            In some cases, a prognostics model will become unstable under certain conditions, after which point the model can no longer represent behavior. 
            stability_tol represents the fraction of the provided argument `times` that are required to be met in simulation, 
            before the model goes unstable in order to produce a valid estimate of mean squared error. 

            If the model goes unstable before stability_tol is met, NaN is returned. 
            Else, model goes unstable after stability_tol is met, the mean squared error calculated from data up to the instability is returned.

    Returns:
        float: MAPE between model and data
    """
    if isinstance(times[0], Iterable):
        # Calculate error for each
        error = [MAPE(t, i, z, **kwargs) for (t, i, z) in zip(times, inputs, outputs)]
        return sum(error)/len(error)

    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)
    stability_tol = kwargs.get('stability_tol', 0.95)

    if not isinstance(x, m.StateContainer):
        x = [m.StateContainer(x_i) for x_i in x]

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]
    
    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    # Checks stability_tol is within bounds
    # Throwing a default after the warning.
    if stability_tol >= 1 or stability_tol < 0:
        warn(f"configurable cutoff must be some float value in the domain (0, 1]. Received {stability_tol}. Resetting value to 0.95")
        stability_tol = 0.95

    counter = 0  # Needed to account for skipped (i.e., none) values
    t_last = times[0]
    err_total = 0
    z_obs = m.output(x)  # Initialize
    cutoffThreshold = math.floor(stability_tol * len(times))

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
                if counter < cutoffThreshold:
                    raise ValueError(f"""Model unstable- NAN reached in simulation (t={t}) before cutoff threshold.
                    Cutoff threshold is {cutoffThreshold}, or roughly {stability_tol * 100}% of the data""")     
                else:
                    warn("Model unstable- NaN reached in simulation (t={})".format(t))
                    break
            err_total += np.sum(np.abs(z.matrix - z_obs.matrix)/z.matrix)
            counter += 1
    return err_total/counter


def DTW(m, times, inputs, outputs, **kwargs):
    """
    Dynamic Time Warping Algorithm 
    """
    if isinstance(times[0], Iterable):
        # Calculate error for each
        error = [DTW(t, i, z, **kwargs) for (t, i, z) in zip(times, inputs, outputs)]
        return sum(error)/len(error)

    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)
    stability_tol = kwargs.get('stability_tol', 0.95)

    if not isinstance(x, m.StateContainer):
        x = [m.StateContainer(x_i) for x_i in x]

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]
    
    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    # Checks stability_tol is within bounds
    # Throwing a default after the warning.
    if stability_tol >= 1 or stability_tol < 0:
        warn(f"configurable cutoff must be some float value in the domain (0, 1]. Received {stability_tol}. Resetting value to 0.95")
        stability_tol = 0.95

    counter = 0  # Needed to account for skipped (i.e., none) values
    t_last = times[0]
    err_total = 0
    z_obs = m.output(x)  # Initialize
    cutoffThreshold = math.floor(stability_tol * len(times))

    def helperDTW(s, t):
        n, m = len(s), len(t)
        dtw_matrix = np.zeros((n+1, m+1))
        for i in range(n+1):
            for j in range(m+1):
                dtw_matrix[i, j] = np.inf
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(s[i-1] - t[j-1])
                # take last min from a square box
                last_min = np.min([dtw_matrix[i-1, j],
                                dtw_matrix[i, j-1], 
                                dtw_matrix[i-1, j-1]])
                dtw_matrix[i, j] = cost + last_min
        return dtw_matrix[n, m]

    # for t, u, z in zip(times, inputs, outputs):
    for t, u, z in zip(times, inputs, outputs):
        while t_last < t:
            t_new = min(t_last + dt, t)
            x = m.next_state(x, u, t_new-t_last)
            t_last = t_new
            if t >= t_last:
                # Only recalculate if required
                z_obs = m.output(x)
        if not (None in z_obs.matrix or None in z.matrix):
            if any (np.isnan(z_obs.matrix)):
                if counter < cutoffThreshold:
                    raise ValueError(f"""Model unstable- NAN reached in simulation (t={t}) before cutoff threshold. Cutoff threshold is {cutoffThreshold}, or roughly {stability_tol * 100}% of the data""")     
                else:
                    warn("Model unstable- NaN reached in simulation (t={})".format(t))
                    break
            if len(z.matrix) != len(z_obs.matrix):
                err_total += helperDTW(z.matrix, z_obs.matrix)
            else:
                err_total += np.sum(np.square(z.matrix - z_obs.matrix), where= ~np.isnan(z.matrix))
            counter += 1

    return err_total/counter

