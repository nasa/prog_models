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