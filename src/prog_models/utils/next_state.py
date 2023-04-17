# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

def euler_next_state(model, x, u, dt: float):
    """
    State transition equation using simple euler integration: Calls next_state(), calculating the next state, and then adds noise and applies limits

    Parameters
    ----------
    x : StateContainer
        state, with keys defined by model.states \n
        e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
    u : InputContainer
        Inputs, with keys defined by model.inputs \n
        e.g., u = m.InputContainer({'i':3.2}) given inputs = ['i']
    dt : float
        Timestep size in seconds (≥ 0) \n
        e.g., dt = 0.1

    Returns
    -------
    x : StateContainer
        Next state, with keys defined by model.states
        e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

    See Also
    --------
    PrognosticsModel.next_state
    """
    # Calculate next state and add process noise
    next_state = model.apply_process_noise(model.next_state(x, u, dt), dt)

    # Apply Limits
    return model.apply_limits(next_state)

def euler_next_state_wrapper(model, x, u, dt: float):
    """
    State transition equation using simple euler integration: Calls next_state(), calculating the next state, and then adds noise and applies limits

    Parameters
    ----------
    x : StateContainer
        state, with keys defined by model.states \n
        e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
    u : InputContainer
        Inputs, with keys defined by model.inputs \n
        e.g., u = m.InputContainer({'i':3.2}) given inputs = ['i']
    dt : float
        Timestep size in seconds (≥ 0) \n
        e.g., dt = 0.1

    Returns
    -------
    x : StateContainer
        Next state, with keys defined by model.states
        e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

    See Also
    --------
    PrognosticsModel.next_state
    """
    # Calculate next state and add process noise
    next_state = model.StateContainer(model.next_state(x, u, dt))
    next_state = model.apply_process_noise(next_state, dt)

    # Apply Limits
    return model.apply_limits(next_state)

def rk4_next_state(model, x, u, dt: float):
    """
    State transition equation using rungekutta4 integration: Calls next_state(), calculating the next state, and then adds noise and applies limits

    Parameters
    ----------
    x : StateContainer
        state, with keys defined by model.states \n
        e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
    u : InputContainer
        Inputs, with keys defined by model.inputs \n
        e.g., u = m.InputContainer({'i':3.2}) given inputs = ['i']
    dt : float
        Timestep size in seconds (≥ 0) \n
        e.g., dt = 0.1

    Returns
    -------
    x : StateContainer
        Next state, with keys defined by model.states
        e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

    See Also
    --------
    PrognosticsModel.next_state
    """
    dx1 = model.StateContainer(model.dx(x, u))
    x2 = x.matrix + dx1.matrix*dt/2
    dx2 = model.dx(x2, u)

    x3 = model.StateContainer({key: x[key] + dt*dx_i/2 for key, dx_i in dx2.items()})
    dx3 = model.dx(x3, u)
    
    x4 = model.StateContainer({key: x[key] + dt*dx_i for key, dx_i in dx3.items()})
    dx4 = model.dx(x4, u)

    x = model.StateContainer({key: x[key] + dt/3*(dx1[key]/2 + dx2[key] + dx3[key] + dx4[key]/2) for key in dx1.keys()})
    return model.apply_limits(model.apply_process_noise(x))

def rk4_next_state_wrapper(model, x, u, dt: float):
    """
    State transition equation using rungekutta4 integration: Calls next_state(), calculating the next state, and then adds noise and applies limits

    Parameters
    ----------
    x : StateContainer
        state, with keys defined by model.states \n
        e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
    u : InputContainer
        Inputs, with keys defined by model.inputs \n
        e.g., u = m.InputContainer({'i':3.2}) given inputs = ['i']
    dt : float
        Timestep size in seconds (≥ 0) \n
        e.g., dt = 0.1

    Returns
    -------
    x : StateContainer
        Next state, with keys defined by model.states
        e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

    See Also
    --------
    PrognosticsModel.next_state
    """
    dx1 = model.StateContainer(model.dx(x, u))
                
    x2 = model.StateContainer({key: x[key] + dt*dx_i/2 for key, dx_i in dx1.items()})
    dx2 = model.dx(x2, u)

    x3 = model.StateContainer({key: x[key] + dt*dx_i/2 for key, dx_i in dx2.items()})
    dx3 = model.dx(x3, u)
    
    x4 = model.StateContainer({key: x[key] + dt*dx_i for key, dx_i in dx3.items()})
    dx4 = model.dx(x4, u)

    x = model.StateContainer({key: x[key] + dt/3*(dx1[key]/2 + dx2[key] + dx3[key] + dx4[key]/2) for key in dx1.keys()})
    return model.apply_limits(model.apply_process_noise(x))
