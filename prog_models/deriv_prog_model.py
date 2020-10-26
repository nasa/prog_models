from . import prognostics_model
from abc import abstractmethod

class DerivProgModel(prognostics_model.PrognosticsModel):
    """
    A Prognostics Model where the first derivative of state can be defined for any time

    The DerivProgModel class is a wrapper around a mathematical model of a
    system as represented by a dx, output, input, and threshold equations.
    It is a subclass of the Model class, with the addition of a threshold
    equation, which defines when some condition, such as end-of-life, has
    been reached.
    """

    @abstractmethod
    def dx(self, t, x, u):
        """
        Returns the first derivative of state `x` at a specific time `t`, given state and input

        Parameters
        ----------
        t : number
            Current timestamp in seconds (≥ 0)
            e.g., t = 3.4
        x : dict
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        u : dict
            Inputs, with keys defined by model.inputs.
            e.g., u = {'i':3.2} given inputs = ['i']

        Returns
        -------
        dx : dict
            First derivitive of state, with keys defined by model.states
            e.g., dx = {'abc': 3.1, 'def': -2.003} given states = ['abc', 'def']
        """
        pass

    def next_state(self, t, x, u, dt): 
        dx = self.dx(t, x, u)
        return {key: x[key] + dx[key]*dt for key in x.keys()}
    