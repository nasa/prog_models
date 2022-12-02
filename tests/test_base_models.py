# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from copy import deepcopy
import io
import numpy as np
from os.path import dirname, join
import pickle
import sys
import unittest

# This ensures that the directory containing ProgModelTemplate is in the python search directory
sys.path.append(join(dirname(__file__), ".."))

from prog_models import *
from prog_models.models import *


class MockModel():
    states = ['a', 'b', 'c', 't']
    inputs = ['i1', 'i2']
    outputs = ['o1']
    default_parameters = {
        'p1': 1.2,
        'x0': {'a': 1, 'b': 5, 'c': -3.2, 't': 0}
    }

    def initialize(self, u = {}, z = {}):
        return self.StateContainer(self.parameters['x0'])

    def next_state(self, x, u, dt):
        return self.StateContainer({
                    'a': x['a'] + u['i1']*dt,
                    'b': x['b'],
                    'c': x['c'] - u['i2'],
                    't': x['t'] + dt
                })

    def output(self, x):
        return self.OutputContainer({'o1': x['a'] + x['b'] + x['c']})


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

class LinearThrownObject(LinearModel):
    inputs = [] 
    states = ['x', 'v']
    outputs = ['x']
    events = ['impact']

    A = np.array([[0, 1], [0, 0]])
    E = np.array([[0], [-9.81]])
    C = np.array([[1, 0]])
    F = None # Will override method

    default_parameters = {
        'thrower_height': 1.83,  # m
        'throwing_speed': 40,  # m/s
        'g': -9.81  # Acceleration due to gravity in m/s^2
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            'x': self.parameters['thrower_height'],  # Thrown, so initial altitude is height of thrower
            'v': self.parameters['throwing_speed']  # Velocity at which the ball is thrown - this guy is a professional baseball pitcher
            })
    
    def threshold_met(self, x):
        return {
            'falling': x['v'] < 0,
            'impact': x['x'] <= 0
        }

    def event_state(self, x): 
        x_max = x['x'] + np.square(x['v'])/(-self.parameters['g']*2) # Use speed and position to estimate maximum height
        return {
            'falling': np.maximum(x['v']/self.parameters['throwing_speed'],0),  # Throwing speed is max speed
            'impact': np.maximum(x['x']/x_max,0) if x['v'] < 0 else 1  # 1 until falling begins, then it's fraction of height
        }

class MockModelWithDerived(MockProgModel):
    param_callbacks = {
            'p1': [derived_callback],
            'p2': [derived_callback2, derived_callback3]
        }


class TestModels(unittest.TestCase):
    def setUp(self):
        # set stdout (so it wont print)
        sys.stdout = io.StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_non_container(self):
        class MockProgModelStateDict(MockProgModel):
            def next_state(self, x, u, dt):
                return {
                    'a': x['a'] + u['i1']*dt,
                    'b': x['b'],
                    'c': x['c'] - u['i2'],
                    't': x['t'] + dt
                }
        
        m = MockProgModelStateDict(process_noise_dist='none', measurement_noise_dist='none')
        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

        # Any event, default
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0})
        self.assertAlmostEqual(times[-1], 5.0, 5)
        self.assertAlmostEqual(outputs[-1]['o1'], -13.2)
        self.assertIsInstance(outputs[-1], m.OutputContainer)

        class MockProgModelStateNdarray(MockProgModel):
            def next_state(self, x, u, dt):
                return np.array([
                    [x['a'] + u['i1']*dt],
                    [x['b']],
                    [x['c'] - u['i2']],
                    [x['t'] + dt]]
                )

        m = MockProgModelStateNdarray(process_noise_dist='none', measurement_noise_dist='none')

        # Any event, default
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0})
        self.assertAlmostEqual(times[-1], 5.0, 5)


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

        m = missing_inputs()
        self.assertEqual(len(m.inputs), 0)

        try: 
            m = missing_outputs()
            self.fail("Should not have worked, missing 'outputs'")
        except ProgModelTypeError:
            pass

        m = missing_initiialize()
        # Should work- initialize is now optional

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
        self.assertSetEqual(set(x0.keys()), set(m.parameters['x0'].keys()))
        for key, value in m.parameters['x0'].items():
            self.assertEqual(value, x0[key])
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

    def test_default_es_and_tm(self):
        # Test 1: TM only
        class NoES(MockModel, prognostics_model.PrognosticsModel):
            events = ['e1', 'e2']

            def threshold_met(self, _):
                return {'e1': False, 'e2': True}

        m = NoES()

        self.assertDictEqual(m.threshold_met({}), {'e1': False, 'e2': True})
        self.assertDictEqual(m.event_state({}), {'e1': 1.0, 'e2': 0.0})

        # Test 2: ES only
        class NoTM(MockModel, prognostics_model.PrognosticsModel):
            events = ['e1', 'e2']

            def event_state(self, _):
                return {'e1': 0.0, 'e2': 1.0}
        
        m = NoTM()

        self.assertDictEqual(m.threshold_met({}), {'e1': True, 'e2': False})
        self.assertDictEqual(m.event_state({}), {'e1': 0.0, 'e2': 1.0})

        # Test 3: Neither ES or TM 
        class NoESTM(MockModel, prognostics_model.PrognosticsModel):
            events = []

        m = NoESTM()
        self.assertDictEqual(m.threshold_met({}), {})
        self.assertDictEqual(m.event_state({}), {})

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

    def test_pickle(self):
        m = MockProgModel(p1 = 1.3)
        pickle.dump(m, open('model_test.pkl', 'wb'))
        m2 = pickle.load(open('model_test.pkl', 'rb'))
        isinstance(m2, MockProgModel)
        self.assertEqual(m.parameters['p1'], m2.parameters['p1'])
        self.assertEqual(m, m2)

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

    def test_sim_namedtuple_access(self):
        m = MockProgModel(process_noise = 0.0)
        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}
        (times, inputs, states, outputs, event_states) = m.simulate_to(6, load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0})
        named_results = m.simulate_to(6, load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0})
        self.assertEquals(times, named_results.times)
        self.assertEquals(inputs, named_results.inputs)
        self.assertEquals(states, named_results.states)
        self.assertEquals(outputs, named_results.outputs)
        self.assertEquals(event_states, named_results.event_states)
        
    def test_next_time_fcn(self):
        m = MockProgModel(process_noise = 0.0)
        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}


        # Any event, default
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **{'dt': 1, 'save_freq': 1e-99})
        self.assertEqual(len(times), 6)

        def next_time(t, x):
            return 0.5

        # With next_time
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **{'save_freq': 1e-99, 'dt': next_time})
        self.assertEqual(len(times), 11)

    def test_sim_measurement_noise(self):
        m = MockProgModel(process_noise = 0.0, measurement_noise = 1)
        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

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
            i0 = {'i1': 1, 'i2': 2.1}
            for key in i.keys():
                self.assertEqual(i[key], i0[key], "Future loading error")

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
            # Noise will make output not equal the expected
            self.assertNotEqual(round(z['o1'], 6), round(oi, 6))

        ## Now with no measurmeent Noise
        m = MockProgModel(process_noise = 0.0, measurement_noise = 0.0)
        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

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
            i0 = {'i1': 1, 'i2': 2.1}
            for key in i.keys():
                self.assertEqual(i[key], i0[key], "Future loading error")

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
            # Lack of noise will make output as expected
            self.assertEqual(round(z['o1'], 6), round(oi, 6))
    
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
            m.simulate_to(12, load, {'o2': 0.9})
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
            i0 = {'i1': 1, 'i2': 2.1}
            for key in i.keys():
                self.assertEqual(i[key], i0[key], "Future loading error")

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

    def test_vectorization(self):
        m = MockProgModel(process_noise = 0.0)
        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}
        a = np.array([1, 2, 3, 4, 4.5])
        b = np.array([5]*5)
        c = np.array([-3.2, -7.4, -11.6, -15.8, -17.9])
        t = np.array([0, 0.5, 1, 1.5, 2])
        dt = 0.5
        x0 = {'a': deepcopy(a), 'b': deepcopy(b), 'c': deepcopy(c), 't': deepcopy(t)}
        x = m.next_state(x0, load(0), dt)
        for xa, xa0 in zip(x['a'], a):
            self.assertAlmostEqual(xa, xa0+dt)

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

    def test_sim_modes(self):
        m = ThrownObject(process_noise = 0, measurement_noise = 0)
        def load(t, x=None):
            return m.InputContainer({})

        # Default mode should be auto
        result = m.simulate_to_threshold(load, save_freq = 0.75, save_pts = [1.5, 2.5])
        self.assertListEqual(result.times, [0, 0.75, 1.5, 2.25, 2.5, 3, 3.75])  

        # Auto step size
        result = m.simulate_to_threshold(load, dt = 'auto', save_freq = 0.75, save_pts = [1.5, 2.5])
        self.assertListEqual(result.times, [0, 0.75, 1.5, 2.25, 2.5, 3, 3.75])  

        # Auto step size with a max of 2
        result = m.simulate_to_threshold(load, dt = ('auto', 2), save_freq = 0.75, save_pts = [1.5, 2.5])
        self.assertListEqual(result.times, [0, 0.75, 1.5, 2.25, 2.5, 3, 3.75])  

        # Constant step size of 2
        result = m.simulate_to_threshold(load, dt = ('constant', 2), save_freq = 0.75, save_pts = [1.5, 2.5])
        self.assertListEqual(result.times, [0, 2, 4])  

        # Constant step size of 2
        result = m.simulate_to_threshold(load, dt = 2, save_freq = 0.75, save_pts = [1.5, 2.5])
        self.assertListEqual(result.times, [0, 2, 4])  

        result = m.simulate_to_threshold(load, dt = 2, save_pts = [2.5])
        self.assertListEqual(result.times, [0, 4])  

    def test_sim_rk4(self):
        # With non-linear model
        m = ThrownObject()
        def load(t, x=None):
            return m.InputContainer({})
        
        with self.assertRaises(ProgModelException):
            m.simulate_to_threshold(load, integration_method='rk4')

        # With linear model
        m = LinearThrownObject(process_noise = 0, measurement_noise = 0)

        result = m.simulate_to_threshold(load, dt = 0.1, integration_method='rk4')
        self.assertAlmostEqual(result.times[-1], 8.3)

    # when range specified when state doesnt exist or entered incorrectly
    def test_state_limits(self):
        m = MockProgModel()
        m.state_limits = {
            't': (-100, 100)
        }
        x0 = m.initialize()

        def load(t, x=None):
            return m.InputContainer({'i1': 1, 'i2': 2.1})

        # inside bounds
        x0['t'] = 0
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
        self.assertGreaterEqual(states[1]['t'], -100)
        self.assertLessEqual(states[1]['t'], 100)

        # now using the fcn
        x0['t'] = 0
        x = m.apply_limits(x0)
        self.assertAlmostEqual(x['t'], 0, 9)

        # outside low boundary
        x0['t'] = -200
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
        self.assertAlmostEqual(states[1]['t'], -100)

        x0['t'] = -200
        x = m.apply_limits(x0)
        self.assertAlmostEqual(x['t'], -100, 9)

        # outside high boundary
        x0['t'] = 200
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
        self.assertAlmostEqual(states[1]['t'], 100)

        x0['t'] = 200
        x = m.apply_limits(x0)
        self.assertAlmostEqual(x['t'], 100, 9)

        # at low boundary
        x0['t'] = -100
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
        self.assertGreaterEqual(states[1]['t'], -100)
        self.assertLessEqual(states[1]['t'], 100)

        x0['t'] = -100
        x = m.apply_limits(x0)
        self.assertAlmostEqual(x['t'], -100, 9)

        # at high boundary
        x0['t'] = 100
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x = x0)
        self.assertGreaterEqual(states[1]['t'], -100)
        self.assertLessEqual(states[1]['t'], 100)

        x0['t'] = 100
        x = m.apply_limits(x0)
        self.assertAlmostEqual(x['t'], 100, 9)

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

    def test_linear_model(self):
        m = LinearThrownObject()
        m.simulate_to_threshold(lambda t, x = None: m.InputContainer({}))
        # len() = events states inputs outputs
        #         1      2      0      1

        # Matrix overwrite type checking (Can't set attributes for B, D, G; not overwritten)
        # when matrix is not of type NumPy ndarray or standard list
        # @A
        with self.assertRaises(TypeError):
            m.A = "[[0, 1], [0, 0]]" # string
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.A = None # None
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.A = 0 # int
            m.matrixCheck()
        with self.assertRaises(TypeError):
            m.A = 3.14 # float
            m.matrixCheck()
        with self.assertRaises(TypeError):
            m.A = {} # dict
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.A = () # tuple
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.A = set() # set
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.A = True # boolean
            m.matrixCheck()
        # @C
        with self.assertRaises(TypeError):
            m.C = "[[0, 1], [0, 0]]" # string
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.C = None # None
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.C = 0 # int
            m.matrixCheck()
        with self.assertRaises(TypeError):
            m.C = 3.14 # float
            m.matrixCheck()
        with self.assertRaises(TypeError):
            m.C = {} # dict
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.C = () # tuple
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.C = set() # set
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.C = True # boolean
            m.matrixCheck()
        # @E
        with self.assertRaises(TypeError):
            m.E = "[[0, 1], [0, 0]]" # string
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.E = None # None
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.E = 0 # int
            m.matrixCheck()
        with self.assertRaises(TypeError):
            m.E = 3.14 # float
            m.matrixCheck()
        with self.assertRaises(TypeError):
            m.E = {} # dict
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.E = () # tuple
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.E = set() # set
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.E = True # boolean
            m.matrixCheck()
        # @F
        with self.assertRaises(TypeError):
            m.F = "[[0, 1], [0, 0]]" # string
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.F = 0 # int
            m.matrixCheck()
        with self.assertRaises(TypeError):
            m.F = 3.14 # float
            m.matrixCheck()
        with self.assertRaises(TypeError):
            m.F = {} # dict
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.F = () # tuple
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.F = set() # set
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.F = True # boolean
            m.matrixCheck()
        
        # Matrix Dimension Checking
        # when matrix is not proper dimensional (1-D array = C, D, G; 2-D array = A,B,E; None = F;)
        # @A 2x2
        with self.assertRaises(AttributeError):
            m.A = np.array([[0, 1]]) # 1-D array
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.A = np.array([[0, 1], [0, 0], [1, 0]]) # 3-D array
            m.matrixCheck()
        # @B 2x0
        with self.assertRaises(AttributeError):
            m.B = np.array([[]]) # 1-D array
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.B = np.array([[], [], []]) # 3-D array
            m.matrixCheck()
        # @C 1x2
        with self.assertRaises(AttributeError):
            m.C = np.array([[]]) # 0-D array
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.C = np.array([[0, 0], [1, 1]]) # 2-D array
            m.matrixCheck()
        # @D 1x1
        with self.assertRaises(AttributeError):
            m.D = np.array([]) # 0-D array
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.D = np.array([[0], [1]]) # 2-D array
            m.matrixCheck()
        # E 2x1
        with self.assertRaises(AttributeError):
            m.E = np.array([[0]]) # 1-D array
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.E = np.array([[0], [1], [2]]) # 3-D array
            m.matrixCheck()
        
        # when matrix is improperly shaped
        # @A 2x2
        with self.assertRaises(AttributeError):
            m.A = np.array([[0, 1, 2, 3], [0, 0, 1, 2]]) # extra column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.A = np.array([[0], [0]]) # less column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.A = np.array([[0, 1, 2], [0, 0]]) # one row has more columns than another
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.A = np.array([[0, 1], [0, 0], [2, 2]]) # extra row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.A = np.array([[0, 1]]) # less row
            m.matrixCheck()
        # @B 2x0
        with self.assertRaises(AttributeError):
            m.B = np.array([[0, 1 ,2]]) # extra column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.B = np.array([[0]]) # less column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.B = np.array([[0, 1, 2], [0, 0]]) # one row has more columns than another
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.B = np.array([[0, 1], [1, 1], [2, 2]]) # extra row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.B = np.array([[0, 1]]) # less row
            m.matrixCheck()
        # @C 1x2
        with self.assertRaises(AttributeError):
            m.C = np.array([[1, 0, 2]]) # extra column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.C = np.array([[1]]) # less column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.C = np.array([[0, 0], [1, 1], [2, 2]]) # extra row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.C = np.array([[]]) # less row
            m.matrixCheck()
        # @D 1x1
        with self.assertRaises(AttributeError):
            m.D = np.array([[1, 2]]) # extra column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.D = np.array([[]]) # less column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.D = np.array([[0], [1]]) # extra row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.D = np.array([[]]) # less row
            m.matrixCheck()
        # @E 2x1
        with self.assertRaises(TypeError):
            m.E = np.array([0,0], [-9.81, -1]) # extra column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.E = np.array([[], []]) # less column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.E = np.array([[0, 1, 2], [0]]) # one row has more columns than another
            m.matrixCheck() 
        with self.assertRaises(AttributeError): 
            m.E = np.array([[0, 1], [0, 0], [2, 2]]) # extra row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.E = np.array([[0, 1]]) # less row
            m.matrixCheck()
        # @G 1x1
        with self.assertRaises(AttributeError):
            m.G = np.array([0, 1]) # extra column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.G = np.array([[]]) # less column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.G = np.array([[0], [1]]) # extra row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.G = np.array([[]]) # less row
            m.matrixCheck()

    def test_F_property_not_none(self):
        class ThrownObject(LinearThrownObject):
            F = np.array([[1, 0]]) # Will override method

            default_parameters = {
                'thrower_height': 1.83,  # m
                'throwing_speed': 40,  # m/s
                'g': -9.81  # Acceleration due to gravity in m/s^2
            }

        m = ThrownObject()
        m.simulate_to_threshold(lambda t, x = None: m.InputContainer({}))
        m.matrixCheck()
        self.assertIsInstance(m.F, np.ndarray)
        self.assertTrue(np.array_equal(m.F, np.array([[1, 0]])))

    def test_init_matrix_as_list(self):
        class ThrownObject(LinearThrownObject):
            A = [[0, 1], [0, 0]]
            E = [[0], [-9.81]]
            C = [[1, 0]]

        m = ThrownObject()
        m.matrixCheck()
        self.assertIsInstance(m.A, np.ndarray)
        self.assertTrue(np.array_equal(m.A, np.array([[0, 1], [0, 0]])))
        self.assertIsInstance(m.E, np.ndarray)
        self.assertTrue(np.array_equal(m.E, np.array([[0], [-9.81]])))
        self.assertIsInstance(m.C, np.ndarray)
        self.assertTrue(np.array_equal(m.C, np.array([[1, 0]])))

    def test_event_state_function(self):
        class ThrownObject(LinearThrownObject):
            F = None # Will override method
            
            def threshold_met(self, x):
                return {
                    'falling': x['v'] < 0,
                    'impact': x['x'] <= 0
                }
        # test coverage needs testing of event_state not overridden

    def test_progress_bar(self):
        m = MockProgModel(process_noise = 0.0)
        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

        # Define output redirection
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        # Test progress bar matching
        simulate_results = m.simulate_to_threshold(load, {'o1': 0.8}, **{'dt': 0.5, 'save_freq': 1.0}, print=False, progress=True)
        capture_split =  [l+"%" for l in capturedOutput.getvalue().split("%") if l][:11]
        percentage_vals = [0, 9, 19, 30, 40, 50, 60, 70, 80, 90, 100]
        for i in range(len(capture_split)):
            actual = '%s |%s| %s%% %s' % ("Progress", "█" * percentage_vals[i] + '-' * (100 - percentage_vals[i]), str(percentage_vals[i])+".0","")
            self.assertEqual(capture_split[i].strip(), actual.strip())
        
    def test_containers(self):
        m = ThrownObject()
        c1 = m.StateContainer({'x': 1.7, 'v': 40})
        c2 = m.StateContainer(np.array([[1.7], [40]]))
        self.assertEqual(c1, c2)
        self.assertListEqual(list(c1.keys()), m.states)

        input_c1 = m.InputContainer({})
        input_c2 = m.InputContainer(np.array([]))
        self.assertEqual(input_c1, input_c2)
        self.assertListEqual(list(input_c1.keys()), m.inputs)

        output_c1 = m.OutputContainer({'x': 1.7})
        output_c2 = m.OutputContainer(np.array([[1.7]]))
        self.assertEqual(output_c1, output_c2)
        self.assertListEqual(list(output_c1.keys()), m.outputs)

    def test_thrown_object_drag(self):
        def future_load(t, x=None):
            return {}
        event = 'impact'
        m_nd = ThrownObject(process_noise_dist = 'none')
        
        # Create no drag model ('cd' = 0)
        m_nd.parameters['cd'] = 0
        simulated_results_nd = m_nd.simulate_to_threshold(future_load, threshold_keys=[event], dt=0.005, save_freq=1)
        # Create default drag model ('cd' = 0.007)
        m_df = ThrownObject(process_noise_dist = 'none')
        simulated_results_df = m_df.simulate_to_threshold(future_load, threshold_keys=[event], dt=0.005, save_freq=1)
        # Create high drag model ('cd' = 1.0)
        m_hi = ThrownObject(process_noise_dist = 'none')
        m_hi.parameters['cd'] = 1
        simulated_results_hi = m_hi.simulate_to_threshold(future_load, threshold_keys=[event], dt=0.005, save_freq=1)

        # Test no drag simulated results different from default
        self.assertNotEqual(simulated_results_nd.times, simulated_results_df.times)
        self.assertNotEqual(simulated_results_nd.states, simulated_results_df.states)
        self.assertGreater(simulated_results_nd.times[-1], simulated_results_df.times[-1])

        # Test high drag simulated results different from default
        self.assertNotEqual(simulated_results_hi.times, simulated_results_df.times)
        self.assertNotEqual(simulated_results_hi.states, simulated_results_df.states)
        self.assertLess(simulated_results_hi.times[-1], simulated_results_df.times[-1])

        # Test high drag simulated results different from no drag
        self.assertNotEqual(simulated_results_hi.times, simulated_results_nd.times)
        self.assertNotEqual(simulated_results_hi.states, simulated_results_nd.states)

# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Base Models")
    result = runner.run(l.loadTestsFromTestCase(TestModels)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
