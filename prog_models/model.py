# Todo(CT): Should we name this SytemModel?
class Model:
        """A general time-variant state space model of system behavior.

        The Model class is a wrapper around a mathematical model of a system as
        represented by a state and output equation. Optionally, it may also
        include an input equation, which defines what the system input should be
        at any given time, and an initialize equation, which computes the initial
        system state given the inputs and outputs.

        A Model also has a parameters structure, which contains fields for
        various model parameters. The parameters structure is always given as a
        first argument to all provided equation handles. However, when calling
        the methods for these equations, it need not be specified as it is passed
        by default since it is a property of the class.
        
        The process and sensor noise variances are represented by vectors. When
        using the generate noise methods, samples are generated from zero-mean
        uncorrelated Gaussian noise as specified by these variances.
        """

        name = 'myModel'
        parameters = {} # Configuration Parameters for model
        inputs = []
        states = []
        outputs = []

        def __init__(self):
            pass

        def initialize(self, u, z):
            pass

        def state(self, t, x, u, dt): 
            pass

        def output(self, t, x):
            pass

        def event_state(self, t, x):
            pass

        def simulate_to(self, time, future_loading_eqn, first_output, options = {}):
            config = { # Defaults
                'step_size': 1,
                'save_freq': 10
            }
            config.update(options)
            # TODO(CT): Add checks (e.g., stepsize, save_freq > 0)

            t = 0
            u = future_loading_eqn(t)
            x = self.initialize(u, first_output)
            times = [t]
            inputs = [u]
            states = [x]
            outputs = [first_output]
            event_states = [self.event_state(t, x)]
            next_save = config['save_freq']
            while t < time:
                t += config['step_size']
                u = future_loading_eqn(t)
                x = self.state(t, x, u, config['step_size'])
                if (t >= next_save):
                    next_save += config['save_freq']
                    times.append(t)
                    inputs.append(u)
                    states.append(x)
                    outputs.append(self.output(t, x))
                    event_states.append(self.event_state(t, x))
            if times[-1] != t:
                times.append(t)
                inputs.append(u)
                states.append(x)
                outputs.append(self.output(t, x))
                event_states.append(self.event_state(t, x))
            return {
                't': times,
                'u': inputs,
                'x': states,
                'z': outputs, 
                'event_state': event_states
            }