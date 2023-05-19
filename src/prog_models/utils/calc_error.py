# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from prog_models.sim_result import SimResult, LazySimResult
from collections.abc import Iterable
from warnings import warn
import math
import numpy as np

acceptable_types = {int, float, tuple, np.ndarray, list, SimResult, LazySimResult}

def MAX_E(m, times, inputs, outputs, **kwargs) -> float:
    """
    Calculate the Maximum Error between model behavior and some collected data.

    Args:
        m (PrognosticsModel): Model to use for comparison
        times (list[float]): array of times for each sample
        inputs (list[dict]): array of input dictionaries where input[x] corresponds to time[x]
        outputs (list[dict]): array of output dictionaries where output[x] corresponds to time[x]

    Keyword Args:
        x0 (StateContainer): Current State of the model
        dt (float, optional): Maximum time step in simulation. Defaults to 1e99.
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
                    raise ValueError(f"Model unstable- NAN reached in simulation (t={t}) before cutoff threshold. Cutoff threshold is {cutoffThreshold}, or roughly {stability_tol * 100}% of the data")    
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


def RMSE(m, times, inputs, outputs, **kwargs) -> float:
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


def MSE(self, times, inputs, outputs, _runs = None, **kwargs) -> float:
    """Calculate Mean Squared Error (MSE) between simulated and observed

    Args:
        times (list[float], list[list[float]]): Array of times for each sample.
        inputs (list[dict, SimResult]): Array of input dictionaries where input[x] corresponds to time[x].
        outputs (list[dict, SimResult]): Array of output dictionaries where output[x] corresponds to time[x].

    Keyword Args:
        x0 (StateContainer, dict, optional): Initial state.
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

    # _runs = kwargs.get('_runs', None)
    
    x = kwargs.get('x0', self.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 10)
    stability_tol = kwargs.get('stability_tol', 0.95)

    types = {type(times), type(inputs), type(outputs)}
    if not all(t in acceptable_types for t in types):
        raise TypeError(f"Types passed in must be from the following list: np.ndarray, list, SimResult, or LazySimResult. Current types are: times = {type(times).__name__}, inputs = {type(inputs).__name__}, and outputs = {type(outputs).__name__}")

    if len(times) != len(inputs) or len(inputs) != len(outputs):
        if _runs is not None:
            raise ValueError(f"Times, inputs, and outputs must all be the same length. At run {_runs}, current lengths are times = {len(times)}, inputs = {len(inputs)}, outputs = {len(outputs)}")

    if not isinstance(x, self.StateContainer):
        x = self.StateContainer(x)

    if not isinstance(inputs[0], self.InputContainer):
        inputs = [self.InputContainer(u_i) for u_i in inputs]
    
    if not isinstance(outputs[0], self.OutputContainer):
        outputs = [self.OutputContainer(z_i) for z_i in outputs]

    counter = 0  # Needed to account for skipped (i.e., none) values
    t_last = times[0]
    err_total = 0
    z_obs = self.output(x)
    cutoffThreshold = stability_tol * times[-1]

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
                if t <= cutoffThreshold:
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
                    raise ValueError(f"Model unstable- NAN reached in simulation (t={t}) before cutoff threshold. Cutoff threshold is {cutoffThreshold}, or roughly {stability_tol * 100}% of the data")
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
                    raise ValueError(f"Model unstable- NAN reached in simulation (t={t}) before cutoff threshold. Cutoff threshold is {cutoffThreshold}, or roughly {stability_tol * 100}% of the data")  
                else:
                    warn("Model unstable- NaN reached in simulation (t={})".format(t))
                    break
            err_total += np.sum(np.abs(z.matrix - z_obs.matrix)/z.matrix)
            counter += 1
    return err_total/counter
