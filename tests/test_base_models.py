# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from prog_models import *
from prog_models.models import *
from copy import deepcopy


class MockModel():
    states = ['a', 'b', 'c', 't']
    inputs = ['i1', 'i2']
    outputs = ['o1']
    default_parameters = {
        'p1': 1.2,
        'x0': {'a': 1, 'b': 5, 'c': -3.2, 't': 0}
    }

    def initialize(self, u = {}, z = {}):
        return deepcopy(self.parameters['x0'])

    def next_state(self, x, u, dt):
        x['a']+= u['i1']*dt
        x['c']-= u['i2']
        x['t']+= dt
        return x

    def output(self, x):
        return {'o1': x['a'] + x['b'] + x['c']}


class MockProgModel(MockModel, prognostics_model.PrognosticsModel):
    events = ['e1', 'e2']

    def event_state(self, x):
        t = x['t']
        return {
            'e1': max(1-t/5.0,0),
            'e2': max(1-t/15.0,0)
            }

    def threshold_met(self, x):
        return {key : value < 1e-6 for (key, value) in self.event_state(x).items()}

def derived_callback(config):
    return {
        'p2': config['p1']  # New config
    }

def derived_callback2(config):
    return {  # Testing chained update
        'p3': config['p2'], 
    }

def derived_callback3(config):
    return {  # Testing 2nd chained update 
        'p4': -2 * config['p2'], 
    }


class MockModelWithDerived(MockProgModel):
    param_callbacks = {
            'p1': [derived_callback],
            'p2': [derived_callback2, derived_callback3]
        }


class TestModels(unittest.TestCase):
    def test_templates(self):
        import prog_model_template
        m = prog_model_template.ProgModelTemplate()

    def test_derived(self):
        m = MockModelWithDerived()
        self.assertAlmostEqual(m.parameters['p1'], 1.2, 5)
        self.assertAlmostEqual(m.parameters['p2'], 1.2, 5)
        self.assertAlmostEqual(m.parameters['p3'], 1.2, 5)
        self.assertAlmostEqual(m.parameters['p4'], -2.4, 5)

        m.parameters['p1'] = 2.4
        self.assertAlmostEqual(m.parameters['p1'], 2.4, 5)
        self.assertAlmostEqual(m.parameters['p2'], 2.4, 5)
        self.assertAlmostEqual(m.parameters['p3'], 2.4, 5)
        self.assertAlmostEqual(m.parameters['p4'], -4.8, 5)

        m.parameters['p2'] = 5
        self.assertAlmostEqual(m.parameters['p1'], 2.4, 5)  # No change
        self.assertAlmostEqual(m.parameters['p2'], 5, 5)
        self.assertAlmostEqual(m.parameters['p3'], 5, 5)
        self.assertAlmostEqual(m.parameters['p4'], -10, 5)

    def test_broken_models(self):

        class missing_states(prognostics_model.PrognosticsModel):
            inputs = ['i1', 'i2']
            outputs = ['o1']
            parameters = {'process_noise':0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, x, u, dt):
                pass
            def output(self, x):
                pass
        
        class empty_states(prognostics_model.PrognosticsModel):
            states = []
            inputs = ['i1', 'i2']
            outputs = ['o1']
            parameters = {'process_noise':0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, x, u, dt):
                pass
            def output(self, x):
                pass
        
        class missing_inputs(prognostics_model.PrognosticsModel):
            states = ['x1', 'x2']
            outputs = ['o1']
            parameters = {'process_noise':0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, x, u, dt):
                pass
            def output(self, x):
                pass
        
        class missing_outputs(prognostics_model.PrognosticsModel):
            states = ['x1', 'x2']
            inputs = ['i1']
            parameters = {'process_noise':0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, x, u, dt):
                pass
            def output(self, x):
                pass
        
        class missing_initiialize(prognostics_model.PrognosticsModel):
            inputs = ['i1']
            states = ['x1', 'x2']
            outputs = ['o1']
            parameters = {'process_noise':0.1}
            def next_state(self, x, u, dt):
                pass
            def output(self, x):
                pass
        
        class missing_output(prognostics_model.PrognosticsModel):
            inputs = ['i1']
            states = ['x1', 'x2']
            outputs = ['o1']
            parameters = {'process_noise':0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, x, u, dt):
                pass

        try: 
            m = missing_states()
            self.fail("Should not have worked, missing 'states'")
        except ProgModelTypeError:
            pass

        try: 
            m = empty_states()
            self.fail("Should not have worked, empty 'states'")
        except ProgModelTypeError:
            pass

        try: 
            m = missing_inputs()
            self.fail("Should not have worked, missing 'inputs'")
        except ProgModelTypeError:
            pass

        try: 
            m = missing_outputs()
            self.fail("Should not have worked, missing 'outputs'")
        except ProgModelTypeError:
            pass

        try: 
            m = missing_initiialize()
            self.fail("Should not have worked, missing 'initialize' method")
        except TypeError:
            pass

        try: 
            m = missing_output()
            self.fail("Should not have worked, missing 'output' method")
        except TypeError:
            pass

    def __noise_test(self, noise_key, dist_key, keys):
        m = MockProgModel(**{noise_key: 0.0})
        for key in keys:
            self.assertIn(key, m.parameters[noise_key])
            self.assertAlmostEqual(m.parameters[noise_key][key], 0.0)

        i = 0
        noise = {}
        for key in keys:
            noise[key] = i
            i += 1
        m = MockProgModel(**{noise_key: noise})
        for key in keys:
            self.assertIn(key, m.parameters[noise_key])
            self.assertAlmostEqual(m.parameters[noise_key][key], noise[key])

        def add_one(self, x):
            return {key: value + 1 for (key, value) in x.items()}
        m = MockProgModel(**{noise_key: add_one})
        x = getattr(m, "apply_{}".format(noise_key))({key: 1 for key in keys})
        self.assertEqual(x[keys[0]], 2)

        try:
            noise = {}
            for i in range(len(keys)-1):
                noise[keys[i]] = i
            m = MockProgModel(**{noise_key: noise})
            self.fail("Should have raised exception at missing process_noise key")
        except ProgModelTypeError:
            pass

        try:
            noise = []
            m = MockProgModel(**{noise_key: noise})
            self.fail("Should have raised exception - inproper format")
        except Exception:
            pass            

        # Test that it ignores process_noise_dist in case where process_noise is a function
        m = MockProgModel(**{noise_key: add_one, dist_key: 'invalid one'})
        x = getattr(m, "apply_{}".format(noise_key))({key: 1 for key in keys})
        self.assertEqual(x[keys[0]], 2)

        # Invalid dist
        try:
            noise = {key : 0.0 for key in keys}
            m = MockProgModel(**{noise_key: noise, dist_key: 'invalid one'})
            self.fail("Invalid noise distribution")
        except ProgModelTypeError:
            pass

        # Invalid dist
        try:
            m = MockProgModel(**{noise_key: 0, dist_key: 'invalid one'})
            self.fail("Invalid noise distribution")
        except ProgModelTypeError:
            pass

        # Valid distributions
        m = MockProgModel(**{noise_key: 0, dist_key: 'uniform'})
        m = MockProgModel(**{noise_key: 0, dist_key: 'gaussian'})
        m = MockProgModel(**{noise_key: 0, dist_key: 'normal'})
        m = MockProgModel(**{noise_key: 0, dist_key: 'triangular'})
        
    def test_process_noise(self):
        self.__noise_test('process_noise', 'process_noise_dist', MockProgModel.states)

    def test_measurement_noise(self):
        self.__noise_test('measurement_noise', 'measurement_noise_dist', MockProgModel.outputs)

    def test_prog_model(self):
        m = MockProgModel() # Should work- sets default
        m = MockProgModel(process_noise = 0.0)
        x0 = m.initialize()
        self.assertDictEqual(x0, m.parameters['x0'])
        x = m.next_state(x0, {'i1': 1, 'i2': 2.1}, 0.1)
        self.assertAlmostEqual(x['a'], 1.1, 6)
        self.assertAlmostEqual(x['c'], -5.3, 6)
        self.assertEqual(x['b'], 5)
        z = m.output(x)
        self.assertAlmostEqual(z['o1'], 0.8, 5)
        e = m.event_state({'t': 0})
        t = m.threshold_met({'t': 0})
        self.assertAlmostEqual(e['e1'], 1.0, 5)
        self.assertFalse(t['e1'])
        e = m.event_state({'t': 5})
        self.assertAlmostEqual(e['e1'], 0.0, 5)
        t = m.threshold_met({'t': 5})
        self.assertTrue(t['e1'])
        t = m.threshold_met({'t': 10})
        self.assertTrue(t['e1'])

    def test_model_gen(self):
        keys = {
            'states': ['a', 'b', 'c', 't'],
            'inputs': ['i1', 'i2'],
            'outputs': ['o1'],
            'events': ['e1']
        }

        def initialize(u, z):
            return {'a': 1, 'b': 3, 'c': -3.2, 't': 0}

        def next_state(x, u, dt):
            x['a']+= u['i1']*dt
            x['c']-= u['i2']
            x['t']+= dt
            return x

        def dx(x, u):
            return {'a': u['i1'], 'b': 0, 'c': u['i2'], 't':1}

        def output(x):
            return {'o1': x['a'] + x['b'] + x['c']}

        def event_state(x):
            t = x['t']
            return {'e1': max(1-t/5.0,0)}

        def threshold_met(x):
            t = x['t']
            return {'e1': max(1-t/5.0,0) < 1e-6}

        m = prognostics_model.PrognosticsModel.generate_model(keys, initialize, output, next_state_eqn = next_state, event_state_eqn = event_state, threshold_eqn = threshold_met)
        x0 = m.initialize({}, {})
        x = m.next_state(x0, {'i1': 1, 'i2': 2.1}, 0.1)
        self.assertAlmostEqual(x['a'], 1.1, 6)
        self.assertAlmostEqual(x['c'], -5.3, 6)
        self.assertEqual(x['b'], 3)
        z = m.output(x)
        self.assertAlmostEqual(z['o1'], -1.2, 5)
        e = m.event_state({'t': 0})
        t = m.threshold_met({'t': 0})
        self.assertAlmostEqual(e['e1'], 1.0, 5)
        self.assertFalse(t['e1'])
        e = m.event_state({'t': 5})
        self.assertAlmostEqual(e['e1'], 0.0, 5)
        t = m.threshold_met({'t': 5.1})
        self.assertTrue(t['e1'])
        t = m.threshold_met({'t': 10})
        self.assertTrue(t['e1'])

        # Without event_state or threshold
        keys = {
            'states': ['a', 'b', 'c'],
            'inputs': ['i1', 'i2'],
            'outputs': ['o1']
        }
        m = prognostics_model.PrognosticsModel.generate_model(keys, initialize, output, next_state_eqn=next_state)
        x0 = m.initialize({}, {})
        x = m.next_state(x0, {'i1': 1, 'i2': 2.1}, 0.1)
        self.assertAlmostEqual(x['a'], 1.1, 6)
        self.assertAlmostEqual(x['c'], -5.3, 6)
        self.assertEqual(x['b'], 3)
        z = m.output(x)
        self.assertAlmostEqual(z['o1'], -1.2, 5)
        e = m.event_state({'t': 5})
        self.assertDictEqual(e, {})
        t = m.threshold_met({'t': 5.1})
        self.assertDictEqual(t, {})

        # Deriv Model
        m = prognostics_model.PrognosticsModel.generate_model(keys, initialize, output, dx_eqn=dx)
        x0 = m.initialize({}, {})
        dx = m.dx(x0, {'i1': 1, 'i2': 2.1})
        self.assertAlmostEqual(dx['a'], 1, 6)
        self.assertAlmostEqual(dx['b'], 0, 6)
        self.assertAlmostEqual(dx['c'], 2.1, 6)
        x = m.next_state(x0, {'i1': 1, 'i2': 2.1}, 0.1)
        self.assertAlmostEqual(x['a'], 1.1, 6)
        self.assertAlmostEqual(x['c'], -2.99, 6)
        self.assertEqual(x['b'], 3)

    def test_broken_model_gen(self):
        keys = {
            'states': ['a', 'b', 'c', 't'],
            'inputs': ['i1', 'i2'],
            'outputs': ['o1'],
            'events': ['e1']
        }

        def initialize(u, z):
            return {'a': 1, 'b': 5, 'c': -3.2, 't': 0}

        def next_state(x, u, dt):
            x['a']+= u['i1']*dt
            x['c']-= u['i2']
            x['t']+= dt
            return x

        def output(x):
            return {'o1': x['a'] + x['b'] + x['c']}

        def event_state(x):
            t = x['t']
            return {'e1': max(1-t/5.0,0)}

        def threshold_met(x):
            t = x['t']
            return {'e1': max(1-t/5.0,0) < 1e-6}

        try:
            m = prognostics_model.PrognosticsModel.generate_model(keys, 7, output, next_state_eqn=next_state)
            self.fail("Should have failed- non-callable initialize eqn")
        except ProgModelTypeError:
            pass

        try:
            m = prognostics_model.PrognosticsModel.generate_model(keys, initialize, output, next_state_eqn=[])
            self.fail("Should have failed- non-callable next_state eqn")
        except ProgModelTypeError:
            pass
        try:
            m = prognostics_model.PrognosticsModel.generate_model(keys, initialize, output)
            self.fail("Should have failed- missing next_state and dx eqn")
        except ProgModelTypeError:
            pass

        try:
            m = prognostics_model.PrognosticsModel.generate_model(keys, initialize, {}, next_state_eqn = next_state)
            self.fail("Should have failed- non-callable output eqn")
        except ProgModelTypeError:
            pass

        try:
            incomplete_keys = {
                'states': ['a', 'b', 'c'],
                'outputs': ['o1'],
                'events': ['e1']
            }
            m = prognostics_model.PrognosticsModel.generate_model(incomplete_keys, initialize, {}, next_state_eqn = next_state)
            self.fail("Should have failed- missing inputs keys")
        except ProgModelTypeError:
            pass

        try:
            incomplete_keys = {
                'inputs': ['a'],
                'outputs': ['o1'],
                'events': ['e1']
            }
            m = prognostics_model.PrognosticsModel.generate_model(incomplete_keys, initialize, {}, next_state_eqn = next_state)
            self.fail("Should have failed- missing states keys")
        except ProgModelTypeError:
            pass

        try:
            incomplete_keys = {
                'inputs': ['a'],
                'states': ['a', 'b', 'c'],
                'events': ['e1']
            }
            m = prognostics_model.PrognosticsModel.generate_model(incomplete_keys, initialize, {}, next_state_eqn = next_state)
            self.fail("Should have failed- missing outputs keys")
        except ProgModelTypeError:
            pass

        try:
            extra_keys = {
                'inputs': ['a'],
                'states': ['a', 'b', 'c'],
                'outputs': ['o1'],
                'events': ['e1'],
                'abc': 'def'
            }
            m = prognostics_model.PrognosticsModel.generate_model(extra_keys, initialize, output, next_state_eqn = next_state)
        except ProgModelTypeError:
            self.fail("Should not have failed- extra keys")

        try:
            incomplete_keys = {
                'inputs': ['a'],
                'states': ['a', 'b', 'c'],
                'outputs': ['o1'],
                'events': ['e1']
            }
            m = prognostics_model.PrognosticsModel.generate_model(incomplete_keys, initialize, output, event_state_eqn=-3, next_state_eqn = next_state)
            self.fail("Should have failed- not callable event_state eqn")
        except ProgModelTypeError:
            pass

        try:
            incomplete_keys = {
                'inputs': ['a'],
                'states': ['a', 'b', 'c'],
                'outputs': ['o1'],
                'events': ['e1']
            }
            m = prognostics_model.PrognosticsModel.generate_model(incomplete_keys, initialize, output, event_state_eqn=event_state, threshold_eqn=-3, next_state_eqn = next_state)
            self.fail("Should have failed- not callable threshold eqn")
        except ProgModelTypeError:
            pass

        # Non-iterable state
        try:
            keys = {
                'states': 10, # should be a list
                'inputs': ['i1', 'i2'],
                'outputs': ['o1'],
                'events': ['e1']
            }
            def dx(t, x, u):
                return {'a': u['i1'], 'b': 0, 'c': u['i2']}
            m = prognostics_model.PrognosticsModel.generate_model(keys, initialize, output, dx_eqn=dx)
            self.fail("Should have failed- non iterable states")
        except TypeError:
            pass

        try:
            keys = {
                'states': ['a', 'b', 'c'], 
                'inputs': ['i1', 'i2'],
                'outputs': -2,# should be a list
                'events': ['e1']
            }
            def dx(t, x, u):
                return {'a': u['i1'], 'b': 0, 'c': u['i2']}
            m = prognostics_model.PrognosticsModel.generate_model(keys, initialize, output, dx_eqn=dx)
            self.fail("Should have failed- non iterable outputs")
        except ProgModelTypeError:
            pass


    def test_sim_to_thresh(self):
        m = MockProgModel(process_noise = 0.0)
        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

        # Any event, default
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0})
        self.assertAlmostEqual(times[-1], 5.0, 5)

        # Any event, initial state 
        x0 = {'a': 1, 'b': 5, 'c': -3.2, 't': -1}
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0, 'x': x0})
        self.assertAlmostEqual(times[-1], 6.0, 5)
        self.assertAlmostEqual(states[0]['t'], -1.0, 5)

        # Any event, manual
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0}, threshold_keys=['e1', 'e2'])
        self.assertAlmostEqual(times[-1], 5.0, 5)

        # Only event 2
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0}, threshold_keys=['e2'])
        self.assertAlmostEqual(times[-1], 15.0, 5)

        # Threshold before event
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0, 'horizon': 5.0}, threshold_keys=['e2'])
        self.assertAlmostEqual(times[-1], 5.0, 5)

        # Threshold after event
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0, 'horizon': 20.0}, threshold_keys=['e2'])
        self.assertAlmostEqual(times[-1], 15.0, 5)

        # No thresholds
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0, 'horizon': 20.0}, threshold_keys=[])
        self.assertAlmostEqual(times[-1], 20.0, 5)

        # Custom thresholds met eqn- both keys
        def thresh_met(thresholds):
            return all(thresholds.values())
        config = {'dt': 0.5, 'save_freq': 1.0, 'horizon': 20.0, 'thresholds_met_eqn': thresh_met}
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **config, threshold_keys=['e1', 'e2'])
        self.assertAlmostEqual(times[-1], 15.0, 5)

        try:
            (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, threshold_keys=['e1', 'e2', 'e3'], **{'dt': 0.5, 'save_freq': 1.0})
            self.fail("Should fail- extra threshold key")
        except ProgModelInputException:
            pass

    def test_sim_past_thresh(self):
        m = MockProgModel(process_noise = 0.0)
        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

        (times, inputs, states, outputs, event_states) = m.simulate_to(6, load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0})
        self.assertAlmostEqual(times[-1], 6.0, 5)
        
    def test_sim_prog(self):
        m = MockProgModel(process_noise = 0.0)
        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}
        
        ## Check inputs
        (times, inputs, states, outputs, event_states) = m.simulate_to(0, load, {'o1': 0.8})
        self.assertEqual(len(times), 1)

        try:
            m.simulate_to(-30, load, {'o1': 0.8})
            self.fail("Should have failed- time must be greater than 0")
        except ProgModelInputException:
            pass

        try:
            m.simulate_to([12], load, {'o1': 0.8})
            self.fail("Should have failed- time must be a number")
        except ProgModelInputException:
            pass

        try:
            m.simulate_to(12, load, {})
            self.fail("Should have failed- output must contain each field (e.g., o1)")
        except ProgModelInputException:
            pass

        try:
            m.simulate_to(12, 132, {'o1': 0.8})
            self.fail("Should have failed- future_load should be callable")
        except ProgModelInputException:
            pass

        ## Simulate
        (times, inputs, states, outputs, event_states) = m.simulate_to(3.5, load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0})

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
            self.assertEqual(x['b'], 5)
            self.assertAlmostEqual(x['c'], ci, 5)

        # Check outputs
        self.assertEqual(len(outputs), 5)
        o = [2.8, -0.4, -3.6, -6.8, -8.4]
        for (oi, z) in zip(o, outputs):
            self.assertAlmostEqual(z['o1'], oi, 5)

        # Check event_states
        self.assertEqual(len(event_states), 5)
        e = [1.0, 0.8, 0.6, 0.4, 0.3]
        for (ei, es) in zip(e, event_states):
            self.assertAlmostEqual(es['e1'], ei, 5)

        ## Check last state saving
        (times, inputs, states, outputs, event_states) = m.simulate_to(3, load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0})
        for t in range(0, 4):
            self.assertAlmostEqual(times[t], t, 5)
        self.assertEqual(len(times), 4, "Should be 4 elements in times") # Didn't save last state (because same as savepoint)

        ## Check dt > save_freq
        (times, inputs, states, outputs, event_states) = m.simulate_to(3, load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 0.1})
        for t in range(0, 7):
            self.assertAlmostEqual(times[t], t/2, 5)
        self.assertEqual(len(times), 7, "Should be 7 elements in times") # Didn't save last state (because same as savepoint)

        ## Custom Savepoint test - with last state saving
        (times, inputs, states, outputs, event_states) = m.simulate_to(3, load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 99.0, 'save_pts': [1.45, 2.45]})
        # Check times
        self.assertAlmostEqual(times[0], 0, 5)
        self.assertAlmostEqual(times[1], 1.5, 5)
        self.assertAlmostEqual(times[2], 2.5, 5)
        self.assertEqual(len(times), 4)
        self.assertAlmostEqual(times[-1], 3.0, 5) # Save last step (even though it's not on a savepoint)
        
        ## Custom Savepoint test
        (times, inputs, states, outputs, event_states) = m.simulate_to(2.5, load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 99.0, 'save_pts': [1.45, 2.45]})
        # Check times
        self.assertAlmostEqual(times[0], 0, 5)
        self.assertAlmostEqual(times[1], 1.5, 5)
        self.assertAlmostEqual(times[2], 2.5, 5)
        self.assertEqual(len(times), 3)
        # Last step is a savepoint        

    def test_sim_prog_inproper_config(self):
        m = MockProgModel(process_noise = 0.0)
        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}
        
        ## Check inputs
        config = {'dt': [1, 2]}
        try:
            (times, inputs, states, outputs, event_states) = m.simulate_to(0, load, {'o1': 0.8}, **config)
            self.fail("should have failed - dt must be number")
        except ProgModelInputException:
            pass

        config = {'dt': -1}
        try:
            (times, inputs, states, outputs, event_states) = m.simulate_to(0, load, {'o1': 0.8}, **config)
            self.fail("Should have failed- dt must be positive")
        except ProgModelInputException:
            pass

        config = {'save_freq': [1, 2]}
        try:
            (times, inputs, states, outputs, event_states) = m.simulate_to(0, load, {'o1': 0.8}, **config)
            self.fail("Should have failed- save_freq must be number")
        except ProgModelInputException:
            pass

        config = {'save_freq': -1}
        try:
            (times, inputs, states, outputs, event_states) = m.simulate_to(0, load, {'o1': 0.8}, **config)
            self.fail("Should have failed- save_freq must be positive")
        except ProgModelInputException:
            pass

        config = {'horizon': [1, 2]}
        try:
            (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **config)
            self.fail("Should have failed Horizon should be number")
        except ProgModelInputException:
            pass

        config = {'horizon': -1}
        try:
            (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **config)
            self.fail("Should have failed- horizon must be positive")
        except ProgModelInputException:
            pass
        
        config = {'thresholds_met_eqn': -1}
        try:
            (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **config)
            self.fail("Should have failed- thresholds_met_eqn must be callable")
        except ProgModelInputException:
            pass

        # incorrect number of arguments
        config = {'thresholds_met_eqn': lambda a, b: print(a, b)}
        try:
            (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **config)
            self.fail()
        except ProgModelInputException:
            pass

    # when range specified when state doesnt exist or entered incorrectly
    def test_state_limits(self):
        m = MockProgModel()
        m.state_limits = {
            't': (-100, 100)
        }
        x0 = m.initialize()

        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

        # inside bounds
        x0['t'] = 0
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
        self.assertGreaterEqual(states[1]['t'], -100)
        self.assertLessEqual(states[1]['t'], 100)

        # outside low boundary
        x0['t'] = -200
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
        self.assertAlmostEqual(states[1]['t'], -100)

        # outside high boundary
        x0['t'] = 200
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
        self.assertAlmostEqual(states[1]['t'], 100)

        # at low boundary
        x0['t'] = -100
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
        self.assertGreaterEqual(states[1]['t'], -100)
        self.assertLessEqual(states[1]['t'], 100)

        # at high boundary
        x0['t'] = 100
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
        self.assertGreaterEqual(states[1]['t'], -100)
        self.assertLessEqual(states[1]['t'], 100)

        # when state doesn't exist
        try:
            x0['n'] = 0
            (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
            self.fail()
        except Exception:
            pass

        # when state entered incorrectly
        try:
            x0['t'] = 'f'
            (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
            self.fail()
        except Exception:
            pass

        # when boundary entered incorrectly
        try:
            m.state_limits = { 't': ('f', 100) }
            x0['t'] = 0
            (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
            self.fail()
        except Exception:
            pass

        try:
            m.state_limits = { 't': (-100, 'f') }
            x0['t'] = 0
            (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
            self.fail()
        except Exception:
            pass

        try:
            m.state_limits = { 't': (100) }
            x0['t'] = 0
            (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
            self.fail()
        except Exception:
            pass


# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    # This ensures that the directory containing ProgModelTemplate is in the python search directory
    import sys
    from os.path import dirname, join
    sys.path.append(join(dirname(__file__), ".."))

    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Base Models")
    result = runner.run(l.loadTestsFromTestCase(TestModels)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
