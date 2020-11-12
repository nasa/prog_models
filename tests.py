import unittest
from prog_models import *
import prog_models
import copy

class MockModel(model.Model):
    states = ['a', 'b', 'c']
    inputs = ['i1', 'i2']
    outputs = ['o1']
    parameters = {
        'p1': 1.2,
        'x0': {'a': 1, 'b': [3, 2], 'c': -3.2}
    }

    def __init__(self, options = {}):
        self.parameters.update(options)
        super().__init__()

    def initialize(self, u, z):
        return copy.deepcopy(self.parameters['x0'])

    def next_state(self, t, x, u, dt):
        x['a']+= u['i1']*dt
        x['c']-= u['i2']
        return x

    def output(self, t, x):
        return {'o1': x['a'] + sum(x['b']) + x['c']}

class TestModels(unittest.TestCase):
    def test_broken_models(self):
        class missing_states(model.Model):
            inputs = ['i1', 'i2']
            outputs = ['o1']
            parameters = {'process_noise':0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, t, x, u, dt):
                pass
            def output(self, t, x):
                pass
        class empty_states(model.Model):
            states = []
            inputs = ['i1', 'i2']
            outputs = ['o1']
            parameters = {'process_noise':0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, t, x, u, dt):
                pass
            def output(self, t, x):
                pass
        class missing_inputs(model.Model):
            states = ['x1', 'x2']
            outputs = ['o1']
            parameters = {'process_noise':0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, t, x, u, dt):
                pass
            def output(self, t, x):
                pass
        class empty_inputs(model.Model):
            inputs = []
            states = ['x1', 'x2']
            outputs = ['o1']
            parameters = {'process_noise':0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, t, x, u, dt):
                pass
            def output(self, t, x):
                pass
        class missing_outputs(model.Model):
            states = ['x1', 'x2']
            inputs = ['i1']
            parameters = {'process_noise':0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, t, x, u, dt):
                pass
            def output(self, t, x):
                pass
        class empty_outputs(model.Model):
            inputs = ['i1']
            states = ['x1', 'x2']
            outputs = []
            parameters = {'process_noise':0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, t, x, u, dt):
                pass
            def output(self, t, x):
                pass
        class missing_initiialize(model.Model):
            inputs = ['i1']
            states = ['x1', 'x2']
            outputs = ['o1']
            parameters = {'process_noise':0.1}
            def next_state(self, t, x, u, dt):
                pass
            def output(self, t, x):
                pass
        class missing_next_state(model.Model):
            inputs = ['i1']
            states = ['x1', 'x2']
            outputs = ['o1']
            parameters = {'process_noise':0.1}
            def initialize(self, u, z):
                pass
            def output(self, t, x):
                pass
        class missing_output(model.Model):
            inputs = ['i1']
            states = ['x1', 'x2']
            outputs = ['o1']
            parameters = {'process_noise':0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, t, x, u, dt):
                pass

        try: 
            m = missing_states()
            self.fail("Should not have worked, missing 'states'")
        except prog_models.ProgModelTypeError:
            pass

        try: 
            m = empty_states()
            self.fail("Should not have worked, empty 'states'")
        except prog_models.ProgModelTypeError:
            pass

        try: 
            m = missing_inputs()
            self.fail("Should not have worked, missing 'inputs'")
        except prog_models.ProgModelTypeError:
            pass

        try: 
            m = empty_inputs()
            self.fail("Should not have worked, empty 'inputs'")
        except prog_models.ProgModelTypeError:
            pass

        try: 
            m = missing_outputs()
            self.fail("Should not have worked, missing 'outputs'")
        except prog_models.ProgModelTypeError:
            pass

        try: 
            m = empty_outputs()
            self.fail("Should not have worked, empty 'outputs'")
        except prog_models.ProgModelTypeError:
            pass

        try: 
            m = missing_initiialize()
            self.fail("Should not have worked, missing 'initialize' method")
        except TypeError:
            pass

        try: 
            m = missing_next_state()
            self.fail("Should not have worked, missing 'next_state' method")
        except TypeError:
            pass

        try: 
            m = missing_output()
            self.fail("Should not have worked, missing 'output' method")
        except TypeError:
            pass
        
    def test_model(self):
        try:
            m = MockModel()
            self.fail("Should not have worked, missing `process_noise`")
        except prog_models.ProgModelTypeError:
            pass
        
        m = MockModel({'process_noise': 0.0})
        x0 = m.initialize({}, {})
        self.assertEqual(x0, m.parameters['x0'])
        x = m.next_state(0, x0, {'i1': 1, 'i2': 2.1}, 0.1)
        self.assertAlmostEqual(x['a'], 1.1, 6)
        self.assertAlmostEqual(x['c'], -5.3, 6)
        self.assertEqual(x['b'], [3, 2])
        z = m.output(0, x)
        self.assertAlmostEqual(z['o1'], 0.8, 5)

    def test_sim(self):
        m = MockModel({'process_noise': 0.0})
        def load(t):
            return {'i1': 1, 'i2': 2.1}
        
        ## Check inputs
        try:
            m.simulate_to(0, load, {'o1': 0.8})
            self.fail("Should have failed- time must be greater than 0")
        except prog_models.ProgModelInputException:
            pass

        try:
            m.simulate_to(-30, load, {'o1': 0.8})
            self.fail("Should have failed- time must be greater than 0")
        except prog_models.ProgModelInputException:
            pass

        try:
            m.simulate_to([12], load, {'o1': 0.8})
            self.fail("Should have failed- time must be a number")
        except prog_models.ProgModelInputException:
            pass

        try:
            m.simulate_to([12], load, {})
            self.fail("Should have failed- output must contain each field (e.g., o1)")
        except prog_models.ProgModelInputException:
            pass

        try:
            m.simulate_to([12], 132, {})
            self.fail("Should have failed- future_load should be callable")
        except prog_models.ProgModelInputException:
            pass

        ## Simulate
        (times, inputs, states, outputs) = m.simulate_to(3.5, load, {'o1': 0.8}, {'dt': 0.5, 'save_freq': 1.0})

        # Check times
        for t in range(0, 4):
            self.assertAlmostEqual(times[t], t, 5)
        self.assertEqual(len(times), 5)
        self.assertAlmostEqual(times[-1], 3.5, 5) # Save last step (even though it's not on a savepoint)
        
        # Check inputs
        self.assertEqual(len(inputs), 5)
        for i in inputs:
            self.assertDictEqual(i, {'i1': 1, 'i2': 2.1}, "Future loading error")

        # Check states
        self.assertEqual(len(states), 5)
        a = [1, 2, 3, 4, 4.5]
        c = [-3.2, -7.4, -11.6, -15.8, -17.9]
        for (ai, ci, x) in zip(a, c, states):
            self.assertAlmostEqual(x['a'], ai, 5)
            self.assertEqual(x['b'], [3, 2])
            self.assertAlmostEqual(x['c'], ci, 5)

        # Check outputs
        self.assertEqual(len(outputs), 5)
        o = [2.8, -0.4, -3.6, -6.8, -8.4]

        for (oi, z) in zip(o, outputs):
            self.assertAlmostEqual(z['o1'], oi, 5)

        ## Check last state saving
        (times, inputs, states, outputs) = m.simulate_to(3, load, {'o1': 0.8}, {'dt': 0.5, 'save_freq': 1.0})
        for t in range(0, 4):
            self.assertAlmostEqual(times[t], t, 5)
        self.assertEqual(len(times), 4, "Should be 4 elements in times") # Didn't save last state (because same as savepoint)

if __name__ == '__main__':
    unittest.main()