# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from copy import deepcopy
import numpy as np
from os.path import dirname, join
import unittest


from prog_models import *
from prog_models.models import *
from prog_models.models.test_models.linear_models import (
    OneInputNoOutputNoEventLM, OneInputOneOutputNoEventLM, OneInputNoOutputOneEventLM, OneInputOneOutputNoEventLMPM)
from prog_models.models.thrown_object import LinearThrownObject

class TestEstimateParams(unittest.TestCase):
    def test_estimate_params_works(self):
        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        data = [(results.times, results.inputs, results.outputs)]
        gt = m.parameters.copy()

        self.assertTrue(m.parameters == gt)
        self.assertEqual(m.parameters, gt)

        # Now lets reset some parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        keys = ['thrower_height', 'throwing_speed']
        m.estimate_params(data, keys)
        for key in keys: # using assert not equal also works.
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        
        m.estimate_params(data, keys, bounds=((0, 4), (20, 42)))
        # Enforce if parameters are within bounds after estimate_params()
        check = m.calc_error(results.times, results.inputs, results.outputs)

        # Demonstrates furhter limitations of Parameter Estiamtion

        before = m.parameters
        m.estimate_params(data, keys, bounds=((1.243, 4.0), (-12.9234, 3.33333333)))
        check3 = m.calc_error(results.times, results.inputs, results.outputs)
        after = m.parameters

        # Now with limits that do include the true values
        # only returning the upper bounds? Why after this defined behavior?
        # m.parameters['throwing_speed'] = 15

        # Or two calls to estimate_params would resolve this issue
        m.estimate_params(data, keys, bounds=((0, 8), (20, 42)))
        # m.estimate_params(data, keys, bounds=(0, 8), (20, 42))
        ax = m.parameters
        value3 = m.calc_error(results.times, results.inputs, results.outputs)
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2, "Limits of the Bounds do not include the true values")
        
        # self.assertTrue(m.parameters == gt)


    def test_estimate_params(self):
        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        data = [(results.times, results.inputs, results.outputs)]
        gt = m.parameters.copy()

        # Reset some parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        keys = ['thrower_height', 'throwing_speed', 'g']
        m.estimate_params(data, keys)

        for key in keys: # using assert not equal also works.
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        m.estimate_params(data, keys, bounds=((0, 4), (20, 37), (-20, 0)))

        # Need at least one data point
        with self.assertRaises(ValueError):
            m.estimate_params(times=[], inputs=[], outputs=[])
        m.estimate_params(times=[], inputs=[], outputs=results.outputs)
        with self.assertRaises(ValueError):
            m.estimate_params(times=None, inputs=None, output=None)
        with self.assertRaises(ValueError):
            m.estimate_params(times='', inputs='', outputs='')
        with self.assertRaises(ValueError):
            m.estimate_params(times=[[]], inputs=[[]], outputs=[[]])

        # Now with limits that dont include the true values
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(data, keys, bounds=((0, 4), (20, 37), (-20, 0)))
        value2 = m.calc_error(results.times, results.inputs, results.outputs)
        # Not the most accurate way of seeing when these values are not accurate.
        for key in keys:
            self.assertNotAlmostEqual(m.parameters[key], gt[key], 1, "Limits of the Bounds do not include the true values")

        # Now with limits that do include the true values
        m.estimate_params(data, keys, bounds=((0, 8), (20, 42), (-20, -5)))
        value3 = m.calc_error(results.times, results.inputs, results.outputs)
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2, "Limits of the Bounds do not include the true values")
        
        # Lower Bound Greater than Upper Bound
        # Error called by minimize function.
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=((30, 7), (20, 42), (-20, 0)))

        # Testing all bounds are incorrect
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=((30, 7), (40, 20), (0, -20)))

        #Implememnt different variations of lists and tuples and see if they work as intended
        # Try incomplete list:
        with self.assertRaises(ValueError):
            # Missing bound
            m.estimate_params(data, keys, bounds=((0, 4), (20, 42)))
        with self.assertRaises(ValueError):
            # Extra bound
            m.estimate_params(data, keys, bounds=((0, 4), (20, 42), (-20, 0), (-20, 10)))

        # Dictionary bounds
        m.estimate_params(data, keys, bounds={'thrower_height': (0, 4), 'throwing_speed': (20, 42), 'g': (-20, 0)})
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Dictionary bounds - missing
        # Will fill with (-inf, inf)
        m.estimate_params(data, keys, bounds={'thrower_height': (0, 4), 'throwing_speed': (20, 42)})
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        
        # Dictionary bounds - extra
        m.estimate_params(data, keys, bounds={'thrower_height': (0, 4), 'throwing_speed': (20, 42), 'g': (-20, 0), 'dummy': (-50, 0)})
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # with self.assertRaises(ValueError):
        # Does not do anything. Probably want to include an error message or some indicator to let the user know
        # in case user is expected change however is not seeing it.

        # Behavior, sets in default values and does not error with correct number of bounds.
        # User changes a value that exists and should not be changed
        m.estimate_params(data, keys, bounds={'thrower_height': (0, 4), 'throwing_speed': (20, 42), 'g': (-20, 0), 'rho': (-100, 100)})
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Bounds - wrong type
        with self.assertRaises(ValueError):
            # bounds isn't tuple or dict
            m.estimate_params(data, keys, bounds=0)
        with self.assertRaises(ValueError):
            # bounds isn't tuple or dict
            m.estimate_params(data, keys, bounds='a')
        with self.assertRaises(ValueError):
            # Item isn't a tuple
            m.estimate_params(data, keys, bounds={'g': 7})

        m.estimate_params(data, keys, bounds= {'g': (7, 14)})
        m.estimate_params(data, keys, bounds={'g': [7, 14]})

        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=None)
        with self.assertRaises(ValueError):
            # Tuple isn't of size 2, more specfically, tuple is size less than 2
            m.estimate_params(data, keys, bounds={'g': (7,)})
        with self.assertRaises(ValueError):
            # Tuple is a size greater than 2
            m.estimate_params(data, keys, bounds={'g': (7, 8, 9)})
        with self.assertRaises(ValueError):
            # Item is a list of length 1
            m.estimate_params(data, keys, bounds={'g': [7]})

        # With inputs, outputs, and times
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(times=[results.times], inputs=[results.inputs], outputs=[results.outputs], keys=keys)
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        
        # When values are swapped between parameters
        with self.assertRaises(TypeError):
            m.estimate_params(times=[results.inputs], inputs=[results.times], outputs=[results.outputs], keys=keys)

        # Does not include a required parameter
        with self.assertRaises(ValueError):
            m.estimate_params(times=[results.times], outputs=[results.outputs], keys=keys)

        # No keys
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        # shouldn't this be erroring?
        m.estimate_params(data)
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # No Data
        with self.assertRaises(ValueError):
            m.estimate_params()

        #Testing with Arrays

        m.estimate_params(data, keys, bounds=[(0, 4), (20, 42), (-4, 15)])

        # Too little bounds given in wrapper array
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[(0, 4), (20, 42)])
        
        # Too many bounds given in wrapper array
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[(0, 4), (20, 42), (-4, 15), (-8, 8)])

        # Regardless of type of wrapper type around bounds, it should work
        m.estimate_params(data, keys, bounds=[[0, 4], [20, 42], [-4, 15]])

        # Testing tuple here.
        m.estimate_params(data, keys, bounds=[[0, 4], (20, 42), [-4, 15]])

        # Bounds wrapped by np.array
        m.estimate_params(data, keys, bounds=np.array([[0,4], [20, 42], [-4, 15]]))

        # Bounds wrapped around with an extra wrapper
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[[[0, 4], (20, 42), [-4, 15]]])

        # This should error as outputs is already a dictionary and we cannot place a hashable type within another.
        with self.assertRaises(TypeError):
            m.estimate_params(data, keys, bounds=[[-1, 5], (20, 40), {-5, 15}])

        # Lower Boung greater than Upper Bound
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[[4, 0], [-20, 20], [0, 40]])

        # Incorrect lenght given for bounds
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[[-15, 15], [-20, 20, 32], [0, 4]])
    
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[[-15], [-20, 20], [0, 4]])

        # Testing with np arrays
        npBounds = np.array([(1, 2), (2, 3), (4, 5)])

        m.estimate_params(data, keys, npBounds)

        # One of the bounds has three values.
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=np.array([(1, 2), (2, 3, 4), (4, 5)]))
        
        # 4 Bounds provided for 3 keys
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=np.array([(1, 2), (2, 3), (4,5), (-1, 20)]))

        # 2 Bounds provided for 3 keys
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=np.array([(1, 2), (2, 3)]))

        # Passing a np.array of string value that is a letter
        with self.assertRaises(TypeError):
            m.estimate_params(data, keys, bounds=np.array('a'))
        
        # Passing in string values in the correct 'bounds' format
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=np.array([('a', 'b'),('a', 'c'), ('d', 'e')]))

        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('a', 'b'),('a', 'c'), ('d', 'e')))

        # Passing in bool value.
        with self.assertRaises(TypeError):
            m.estimate_params(data, keys, bounds=np.array(True))
        
        # with self.assertRaises(TypeError):
        m.estimate_params(data, keys, bound=np.array([(False, True),(False, True), (False, True)]))

        # Having npArr defined with one list and two tuples
        m.estimate_params(data, keys, bounds=np.array([[1, 2], (2, 3), (4,5)]))

        m.estimate_params(data, keys, bounds=[np.array([1, 2]), np.array([2, 3]), np.array([4, 5])])

        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[np.array([1, 2]), np.array([2, 3]), np.array([4, 5]), np.array([-1, 20])])
        
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[np.array([1, 2]), np.array([2, 3])])

        m.estimate_params(data, keys, bounds=[np.array(['4', '9']), np.array(['2', '3']), np.array(['4', '5'])])
        
        m.estimate_params(data, keys, bounds=[np.array(['4', '9']), np.array([2, 3]), np.array([4, 5])])

        m.estimate_params(data, keys, bounds=[np.array(['4', '9']), np.array([2, 3]), np.array([4.123, 5.346])])

        # Errors not due incorrect typing but due to upper bound being less than lower bound error
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=np.array([(True, False), (False, True), (False, True)]))


        # Testing overloaded bounds equals standard foramt
        m.estimate_params(data, keys, bounds=(((([-3, 12]))), (1, 20), (-5, 30)))
        check = m.calc_error(results.times, results.inputs, results.outputs)    

        m.estimate_params(data, keys, bounds=((-3, 12), (1, 20), (-5, 30)))
        check2 = m.calc_error(results.times, results.inputs, results.outputs)
        self.assertAlmostEqual(check, check2)

        # Testing passing in strings. Warning should appear
        # Testing passing in strings with standard bounds
        m.estimate_params(data, keys, bounds=(('-3', '12'), ('1', '20'), ('-5', '30')))
        check = m.calc_error(results.times, results.inputs, results.outputs)

        # Testing with np.array
        m.estimate_params(data, keys, bounds=np.array([('-3', '12'), ('1', '20'), ('-5', '30')]))
        check2 = m.calc_error(results.times, results.inputs, results.outputs)

        self.assertAlmostEqual(check, check2)

        m.estimate_params(data, keys, bounds=(('-3.12', '12'), ('1', '20'), ('-5', '30')))
        check3 = m.calc_error(results.times, results.inputs, results.outputs)

        # Checking to make sure original equals the previous ones
        self.assertEqual(check, check3)

        # Passing in an integer for a singular test
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('a', '12'), ('1', '20'), ('-5', '30')))
        
        # Incorrect length of a singular bound
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('-3'), ('1', '20'), ('-5', '30')))
        
        # Upper bound greater than lower bound
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('9', '12'), ('30', '20'), ('-5', '30')))
        
        # Both string literals and upper bound being less than lower bound
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('a', 'b'), ('30', '20'), ('-5', '30')))
        
        # Having an incorrect bound length of three
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('-3', '12'), ('20', '30', '40'), ('-5', '30')))

        # Passing in too many bounds
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('-3', '12'), ('20', '30'), ('-5', '30'), ('-20, 20')))
        
        # Passing in not enough bounds
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('-3', '12'), ('20', '30')))
        
        # Using np.array for each bound. Different typing for each
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[np.array(['a', '9']), np.array(['2', '3']), np.array(['4', '5'])])

        # Using np.array for each bound specifically. Different typings for them
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[np.array(['a', 's']), np.array([2, 3]), np.array([4, 5])])

        # Too many bounds given in np.array for each bound
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[np.array(['4', '9']), np.array(['2', '3']), np.array(['4', '5']), np.array(['-2', '4'])])
        
        # Lower Bound greater than Upper Bound with np.array wrapper around each bound 
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[np.array(['9', '4']), np.array(['2', '3']), np.array(['4', '5'])])
        
        # Checking comparing between a string literal integer and an int
        m.estimate_params(data, keys, bounds=[np.array(['1', 9]), np.array([2, 3]), np.array([4, 5])])

        # Testing bounds equality
        m.estimate_params(data, keys, bounds=((1, 1), (2, 4), (-1, 24)))

        # Testing bounds equality with strings in np.array
        m.estimate_params(data, keys, bounds=[np.array(['9', '9']), np.array(['2', '3']), np.array(['4', '5'])])


# Testing features where keys are not parameters in the model
    def test_keys(self):
        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        data = [(results.times, results.inputs, results.outputs)]

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        keys = ['x', 'y', 'g']
        bound=((0, 8), (20, 42), (-20, -5))
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=bound)


    def test_parameters(self):
        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        data = [(results.times, results.inputs, results.outputs)]

        # Now lets reset some parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        keys = ['thrower_height', 'throwing_speed', 'g']

        # Passing in Method to see if it works
        bound=((0, 8), (20, 42), (-20, -5))
        m.estimate_params(data, keys,  method='TNC', bounds=bound)
        m.estimate_params(data, keys, bounds=bound, method='Nelder-Mead')
        # Passing in Method that does not exist
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=bound, method='madeUpName')
        # Checking options
        m.estimate_params(data, keys, bounds=bound, method='Powell', options={'maxiter': 3, 'disp': True})
        m.estimate_params(data, keys, bounds=bound, method='Powell', options={'maxiter': 0, 'disp': False})

        # Passing in arbitrary options should not error that follow our format.
        m.estimate_params(data, keys, bounds=bound, method='Powell', options= {'1':3, 'disp':1})
        
        # Not setting up 'maxiter' and/or 'disp'
        # Needs to be str: int format.
        with self.assertRaises(TypeError):
            m.estimate_params(data, keys, bounds=bound, method='Powell', options= {1:2, True:False})
        with self.assertRaises(TypeError):
            m.estimate_params(data, keys, bounds=bound, method='Powell', options={'maxiter': '3', 'disp': False})

        # Keys that are not defined in specs
        # Should this error out, maybe a warning should be provided for all the args that do not exist?
        m.estimate_params(data, keys, bounds=bound, method='TNC', options={'1':2, '2':2, '3':3})

        # Reset all progress
        m = ThrownObject()
        m1 = ThrownObject()
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m1.parameters['thrower_height'] = 1.5
        m1.parameters['throwing_speed'] = 25

        times = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        inputs = [{}]*9
        outputs = [
            {'x': 1.83},
            {'x': 36.95},
            {'x': 62.36},
            {'x': 77.81},
            {'x': 83.45},
            {'x': 79.28},
            {'x': 65.3},
            {'x': 41.51},
            {'x': 7.91},
        ]

        m.estimate_params(data, keys, bounds=bound, method='Powell', options={'maxiter': 1e-9, 'disp': False})
        m1.estimate_params(data, keys, bounds=bound, method='Powell', options={'1':2, '2':2, '3':3})

        # Ask questions about what exactly is method doing

        # Testing if results of options is being properly applied to calc_error
        self.assertNotEqual(m.calc_error(times, inputs, outputs), m1.calc_error(times, inputs, outputs))

        # using battery model to see when calc_errors do not equate each other
        m.estimate_params(data, keys, bounds=bound, method='Powell')
        m1.estimate_params(data, keys, bounds=bound, method='CG')

        # For Simple models, there shouldn't be too much change
        self.assertAlmostEqual(m.calc_error(times, inputs, outputs), m1.calc_error(times, inputs, outputs), 2)
        
        # Increasing total amount of iterations, having different methods would matter proportionally.
        m.estimate_params(data, keys, bounds=bound, method='Powell', options={'maxiter': 1e-9, 'disp': False})
        m1.estimate_params(data, keys, bounds=bound, method='CG', options={'maxiter': 1e-9, 'disp': False})

        self.assertNotAlmostEqual(m.calc_error(times, inputs, outputs), m1.calc_error(times, inputs, outputs))

        m = ThrownObject()
        # Defining wrongIntuptLen to test parameter length tests.
        wrongInputLen = [{}]*8

        wrongData = [(times, wrongInputLen, outputs)]
        
        # Wrong Input Length and Values
        with self.assertRaises(ValueError):
            m.estimate_params(times=[times], inputs=[wrongInputLen], outputs=[outputs])
        
        # Same as last example but times not in wrapper list
        with self.assertRaises(ValueError):
            m.estimate_params(times=times, inputs=[wrongInputLen], outputs=[outputs])
        
        # Passing in Runs directly
        with self.assertRaises(ValueError):
            m.estimate_params(wrongData)

        # Defining wrongOutputs to test parameter length tests.
        wrongOutputs = [
            {'x': 1.83},
            {'x': 36.95},
            {'x': 62.36},
            {'x': 77.81},
            {'x': 83.45},
            {'x': 79.28},
            {'x': 65.3},
            {'x': 41.51},
        ]

        wrongData = [(times, inputs, wrongOutputs)]

        # Wrong outputs parameter length
        with self.assertRaises(ValueError):
            m.estimate_params(times=[times], inputs=[inputs], outputs=[wrongOutputs])

        # Both inputs and outputs with incorrect lenghts
        with self.assertRaises(ValueError):
            m.estimate_params(times=[times], inputs=[wrongInputLen], outputs=[wrongOutputs])

        # Without wrapper
        with self.assertRaises(ValueError):
            m.estimate_params(times=times, inputs=wrongInputLen, outputs= wrongOutputs)

        # Passing in incorrect Runs 
        with self.assertRaises(ValueError):
            m.estimate_params(wrongData)

        # Testing functionality works without having a parent wrapper
        m.estimate_params(times=times, inputs=inputs, outputs=outputs)

        # Differnet types of wrappers should not affect function
        # Testing functionality works with having only a few parameters defined in wrapper sequences
        m.estimate_params(times=times, inputs=[inputs], outputs=[outputs])
        
        # Same as last test, but inputs is has tuple wrapper sequence insteads.
        m.estimate_params(times=times, inputs=(inputs), outputs=[outputs])

        # Cannot pass in Sets
        with self.assertRaises(TypeError):
            m.estimate_params(times=set(times), inputs=inputs, outputs=[outputs])

        # NOTE - Cannot have set() around inputs and/or outputs. Provides unhashable errors
        with self.assertRaises(TypeError):
            m.estimate_params(times=times, inputs=set(inputs), outputs=[outputs])

        with self.assertRaises(TypeError):
            m.estimate_params(times=[times], inputs=[set(inputs)], outputs=[outputs])
        
        # This fails because inputs and outputs are both dictionaries within a Set. Sometimes, an empty set within a Set.
        with self.assertRaises(TypeError):
            m.estimate_params(times=[times], inputs=inputs, outputs=set(outputs))

        # Missing inputs.
        with self.assertRaises(ValueError):
            m.estimate_params(times=[times], outputs=[outputs])

        # Passing in nothing at all
        with self.assertRaises(ValueError):
            m.estimate_params()
        
        #  'input' is not a parameter, so techincally not defining the parameter inputs.
        with self.assertRaises(ValueError):
            m.estimate_params(times=[times], input=[inputs], outputs=[outputs]) 

        # Length error expected, 1, 9, 1.
        with self.assertRaises(ValueError):
            m.estimate_params(times=[[times]], inputs=[inputs], outputs=[[outputs]]) 

        # Will work in future case, but not at the current moments
        # m.estimate_params(times=[[times]], inputs=[[inputs]], outputs=[[outputs]])

    def test_multiple_runs(self):
        m = ThrownObject()

        time1 = [0, 1, 2, 4, 5, 6, 7, 8, 9]
        time2 = [0, 1, 2, 3]

        inputs = [[{}]*9, [{}]*4]
        outputs = [[{'x': 1.83},
            {'x': 36.95},
            {'x': 62.36},
            {'x': 77.81},
            {'x': 83.45},
            {'x': 79.28},
            {'x': 65.3},
            {'x': 41.51},
            {'x': 7.91},], 
            [
                {'x': 1.83},
                {'x': 36.95},
                {'x': 62.36},
                {'x': 77.81},
            ]]
        
        # Checking too see if multiple runs can exist.
        # time1 and time2 are explciity being passed in into a parent wrapper list.
        # See definitions of variables to understand format.
        m.estimate_params(times=[time1, time2], inputs=inputs, outputs=outputs)

        # Checking too see if wrapping in tuple works.
        m.estimate_params(times=(time1, time2), inputs=inputs, outputs=outputs)

        # Adding another wrapper list around outputs. List error, 2, 2, 1 will result.
        with self.assertRaises(ValueError):
            m.estimate_params(times=[time1, time2], inputs=inputs, outputs=[outputs])

        # Adding another wrapper list around times. List error, 1, 2, 2 will result.
        with self.assertRaises(ValueError):
            m.estimate_params(times=[[time1, time2]], inputs=inputs, outputs=outputs)

        incorrectTimesRunsLen = [[0, 1, 2, 4, 5, 6, 7, 8, 9]]

        # Passing in only one run for Times whereas inputs and outputs have two runs
        with self.assertRaises(ValueError):
            m.estimate_params(times=incorrectTimesRunsLen, inputs=inputs, outputs=outputs)

        incorrectTimesLen = [[0, 1, 2, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]

        # Passing in the correct amount of runs, but one of the runs has a different length compared to other parameter's lengths
        # This test is also valdiating if we can see which run has a wrong error.
        with self.assertRaises(ValueError):
            m.estimate_params(times=incorrectTimesLen, inputs=inputs, outputs=outputs)

        with self.assertRaises(ValueError):
            m.estimate_params(times=[time1, [time2]], inputs=inputs, outputs=outputs)

        # Wrapper list becomes a set.
        timesSet = [time1, time2]
        # Unhashable type for outputs
        with self.assertRaises(TypeError):
            m.estimate_params(times=set(timesSet), inputs=inputs, outputs=set(outputs))
        
        time1 = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9])
        time2 = [0, 1, 2, 3]

        m.estimate_params(times=[time1, time2], inputs=inputs, outputs=outputs)

        # Another test case that would be fixed with future changes to Containers
        # with self.assertRaises(ValueError):
        #     m.estimate_params(times=[incorrectTimesLen], inputs=[inputs], outputs=[outputs])

def run_tests():
    unittest.main()
    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting EstimateParams Feature")
    result = runner.run(l.loadTestsFromTestCase(TestEstimateParams)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()