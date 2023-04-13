# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
import unittest

from prog_models import *
from prog_models.models import *


class TestEstimateParams(unittest.TestCase):
    def test_estimate_params_works(self):
        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        data = [(results.times, results.inputs, results.outputs)]
        gt = m.parameters.copy()

        self.assertEqual(m.parameters, gt)

        # Now lets incorrectly set some parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        keys = ['thrower_height', 'throwing_speed']
        m.estimate_params(data, keys)
        for key in keys: # using assert not equal also works.
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        
        # Incorrectly set parameters, but provide a bounds for what said values should be.
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(data, keys, bounds=((0, 4), (20, 42)))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        # Checking to see if our estimated parameters are wihtin the bounds.
        self.assertLessEqual(m.parameters['thrower_height'], 4)
        self.assertGreaterEqual(m.parameters['thrower_height'], 0)
        self.assertLessEqual(m.parameters['throwing_speed'], 42)
        self.assertGreaterEqual(m.parameters['throwing_speed'], 20)

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(data, keys, bounds=((0, 4), (40, 40)))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 1)
        self.assertLessEqual(m.parameters['thrower_height'], 4)
        self.assertGreaterEqual(m.parameters['thrower_height'], 0)
        self.assertEqual(m.parameters['throwing_speed'], 40)

        # Demonstrates further limitations of Parameter Estimation
        m.parameters['thrower_height'] = 5
        m.parameters['throwing_speed'] = 19
        m.estimate_params(data, keys, bounds=((1.231, 4), (20, 41.99)))

        # Notice how the estimated parameters equal to the upper bounds of their respective bounds.
        self.assertEqual(m.parameters['thrower_height'], 4)
        self.assertEqual(m.parameters['throwing_speed'], 41.99)

        # Or two calls to estimate_params would resolve this issue
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(data, keys, bounds=((0, 4), (20, 42)))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # When our incorrecltly set parameters that are not within the bounds themselves
        m.parameters['thrower_height'] = 10
        m.parameters['throwing_speed'] = -10
        m.estimate_params(data, keys, bounds=((0, 4), (20, 42)))
        for key in keys:
            self.assertNotAlmostEqual(m.parameters[key], gt[key], 2)
        # These returned values are still within the bounds, they are just not close to the original value at all.
        # This results in our estimate parameters to result in the upper extremas
        self.assertEqual(m.parameters['thrower_height'], 4)
        self.assertEqual(m.parameters['throwing_speed'], 42)

        # Show casing results of having a local min/max
        # Even though all our bounds accomodate for the original model params values,
        # our estimate_parmams returns lower/upper bound values.
        m.parameters['thrower_height'] = 4
        m.parameters['throwing_speed'] = 24
        m.parameters['g'] = -20
        keys = ['thrower_height', 'throwing_speed', 'g']
        m.estimate_params(data, keys, bounds=((0, 4), (20, 42), (-20, -8)))
        for key in keys:
            self.assertNotAlmostEqual(m.parameters[key], gt[key], 2)

        m.estimate_params(data, keys, bounds=((0, 4), (20, 42), (-20, -8)))

        self.assertEqual(m.parameters['thrower_height'], 4)
        self.assertEqual(m.parameters['throwing_speed'], 42)
        self.assertEqual(m.parameters['g'], -20)

        # However, this is a feature that occurs with bounds internally and is not an aspect of our original code.
        m.parameters['thrower_height'] = 4
        m.parameters['throwing_speed'] = 24
        m.parameters['g'] = -20
        m.estimate_params(data, keys)
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Furthermore, notice that these changes are regardless if the lower/upper bounds equals our initially set parameters
        m.parameters['thrower_height'] = 4
        m.parameters['throwing_speed'] = 24
        m.parameters['g'] = -20
        keys = ['thrower_height', 'throwing_speed', 'g']
        m.estimate_params(data, keys, bounds=((0, 5), (20, 42), (-21, -7)))
        for key in keys:
            self.assertNotAlmostEqual(m.parameters[key], gt[key], 2)
        self.assertEqual(m.parameters['thrower_height'], 5)
        self.assertEqual(m.parameters['g'], -7)
        
        # However, note that our throwing_speed does not equal to it's upper bound anymore.
        self.assertNotEqual(m.parameters['throwing_speed'], 42)
        self.assertLess(m.parameters['throwing_speed'], 42)
        self.assertGreaterEqual(m.parameters['throwing_speed'], 20)


    def test_estimate_params(self):
        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        data = [(results.times, results.inputs, results.outputs)]
        gt = m.parameters.copy()

        # Reset some parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        keys = ['thrower_height', 'throwing_speed', 'g']

        # Need at least one data point
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[], inputs=[], outputs=[])
        self.assertEqual(
            'Times, inputs, and outputs must have at least one element',
            str(cm.exception)
        )
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=None, inputs=None, output=None)
        self.assertEqual(
            'Missing keyword arguments times, inputs, outputs',
            str(cm.exception)
        )
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times='', inputs='', outputs='')
        self.assertEqual(
            'Times, inputs, and outputs must have at least one element',
            str(cm.exception) 
        )
        self.assertEqual(
            'Times, inputs, and outputs must have at least one element',
            str(cm.exception)
        )
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=None, inputs=None, output=None)
        self.assertEqual(
            'Missing keyword arguments times, inputs, outputs',
            str(cm.exception)
        )
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times='', inputs='', outputs='')
        self.assertEqual(
            'Times, inputs, and outputs must have at least one element',
            str(cm.exception) 
        )
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[[]], inputs=[[]], outputs=[[]])
        self.assertEqual(
            'Times, inputs, and outputs for Run 0 must have at least one element',
            str(cm.exception)
        )

        # Checking when one parameter is valid but others are not.
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[], inputs=[], outputs=results.outputs)
        self.assertEqual(
             'Times, inputs, and outputs must be same length. Length of times: 0, Length of inputs: 0, Length of outputs: 9',
             str(cm.exception)
        )
        
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=results.times, inputs=None, outputs=None)
        self.assertEqual(
            'Missing keyword arguments inputs, outputs',
            str(cm.exception)
        )
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=None)
        self.assertEqual(
            'Missing keyword arguments outputs',
            str(cm.exception)
        )
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=None, inputs=[None], outputs=[])
        self.assertEqual(
            'Missing keyword arguments times',
            str(cm.exception)
        )
        # Strings are iterables by definition so this would pass as well, regardless if they are ints
        # Later tests will ensure they must be ints.
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times= None, inputs = 'Testing', outputs='Testing')
        self.assertEqual(
            'Missing keyword arguments times',
            str(cm.exception)
        )

        # Now with limits that dont include the true values
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(data, keys, bounds=((0, 4), (20, 37), (-20, 0)))
        # Not the most accurate way of seeing when these values are not accurate.
        for key in keys:
            self.assertNotAlmostEqual(m.parameters[key], gt[key], 1)

        # Now with limits that do include the true values
        m.estimate_params(data, keys, bounds=((0, 8), (20, 42), (-20, -5)))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        
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
        m.estimate_params(data)
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # No Data
        with self.assertRaises(ValueError):
            m.estimate_params()

        # Testing with Arrays
        m.estimate_params(data, keys, bounds=[(0, 4), (20, 42), (-20, 15)])
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Too little bounds given in wrapper array
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[(0, 4), (20, 42)])
        
        # Too many bounds given in wrapper array
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[(0, 4), (20, 42), (-4, 15), (-8, 8)])


        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        # Regardless of type of wrapper type around bounds, it should work
        m.estimate_params(data, keys, bounds=[[0, 4], [20, 42], [-12, 15]])
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Testing tuple here.
        m.estimate_params(data, keys, bounds=[[0, 4], (20, 42), [-12, 15]])
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Bounds wrapped by np.array
        m.estimate_params(data, keys, bounds=np.array([[0,4], [20, 42], [-12, 15]]))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

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
        npBounds = np.array([(0, 4), (-1000, 42), (-900, 0)])
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        m.estimate_params(data, keys, npBounds)
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

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
        
        # True and False values are converted to their integer counterparts. So this does not fail.
        m.estimate_params(data, keys, bound=np.array([(False, True), (False, True), (False, True)]))

        # Having npArr defined with one list and two tuples
        m.estimate_params(data, keys, bounds=np.array([[1, 2], (2, 3), (4,5)]))

        # Correct number of bounds with np.array as their inner wrapper
        m.estimate_params(data, keys, bounds=[np.array([1, 2]), np.array([2, 3]), np.array([4, 5])])

        # Four bounds passed with three keys where inner wrapper is np.arrays
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[np.array([1, 2]), np.array([2, 3]), np.array([4, 5]), np.array([-1, 20])])
        
        # Two bounds passed with three keys where inner wrapper is np.arrays
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[np.array([1, 2]), np.array([2, 3])])

        # Passing in strings gets converted automatically, even inside an np.array
        m.estimate_params(data, keys, bounds=[np.array(['4', '9']), np.array(['2', '3']), np.array(['4', '5'])])
        
        # Passing in strings and integers into bounds
        m.estimate_params(data, keys, bounds=[np.array(['4', '9']), np.array([2, 3]), np.array([4, 5])])

        # Passing in strings, integers, and floating values in bounds.
        m.estimate_params(data, keys, bounds=[np.array(['4', '9']), np.array([2, 3]), np.array([4.123, 5.346])])

        # Errors not due incorrect typing but due to upper bound being less than lower bound error
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=np.array([(True, False), (False, True), (False, True)]))

        # Testing overloaded bounds equals standard foramt
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(data, keys, bounds=(((([-3, 4]))), (1, 400), (-20, 30)))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        value1 = m.calc_error(results.times, results.inputs, results.outputs)

        # Testing different formats of inner wrapper types works.
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(data, keys, bounds=([-3, 12], (1, 400), np.array([-20, 30])))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        value2 = m.calc_error(results.times, results.inputs, results.outputs)
        self.assertAlmostEqual(value1, value2)

        # Testing passing in strings with standard bounds
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.parameters['g'] = -8
        m.estimate_params(data, keys, bounds=(('-3', '12'), ('1', '42'), ('-12', '30')))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        check = m.calc_error(results.times, results.inputs, results.outputs)

        # Testing with np.array
        m.estimate_params(data, keys, bounds=np.array([('-3', '12'), ('1', '42'), ('-12', '30')]))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        check2 = m.calc_error(results.times, results.inputs, results.outputs)

        # Checking if passing in different inner wrapper types for bounds deviates between calls.
        self.assertEqual(check, check2)

        m.estimate_params(data, keys, bounds=(('-3.12', '12'), ('1', '42.125381'), ('-12', '30')))
        check3 = m.calc_error(results.times, results.inputs, results.outputs)

        # Checking if passing in different inner wrapper types for bounds deviates between calls,
        # but this time we are passing in floating values into our bounds
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
        
        # Checking comparing between a string literal integer and an ints
        m.estimate_params(data, keys, bounds=[np.array(['1', 9]), np.array([2, 3]), np.array([4, 5])])

        # Testing bounds equality
        m.estimate_params(data, keys, bounds=((1, 1), (2, 4), (-1, 24)))

        # Testing bounds equality with strings in np.array
        m.estimate_params(data, keys, bounds=[np.array(['9', '9']), np.array(['2', '3']), np.array(['4', '5'])])

        # Resetting parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.parameters['g'] = -8
        times = np.array([0.0, 0.5, 1.0])
        inputs = np.array([{}, {}, {}])
        outputs = np.array([{'x': 1.83}, {'x': 21.83}, {'x': 38.78612068965517}])
        m.estimate_params(times = times, inputs = inputs, outputs = outputs)


# Testing features where keys are not parameters in the model
    def test_keys(self):
        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        data = [(results.times, results.inputs, results.outputs)]
        gt = m.parameters.copy()

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        # Setting kesy to incorrect values
        keys = ['x', 'y', 'g']
        bound=((0, 8), (20, 42), (-20, -5))
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=bound)

        # Incorrect key length
        keys = ['x', 'y']
        bound=((0, 8), (20, 42), (-20, -5))
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=bound)

        # Checking a parameter that does not exist.
        # gives bounds error
        keys = ['thrower_height', 'throwing_speed', 1]
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=bound)

        # Checking same as prev test
        keys = ['thrower_height', 'throwing_speed', '1']
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=bound)
        
        # Keys within a tuple should not error
        keys = ('thrower_height', 'throwing_speed', 'g')
        m.estimate_params(data, keys, bounds=bound)

        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Reset parameters after successful estimate_params call
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        # Keys are a Set. Should throw an exception.
        keys = {'thrower_height', 'throwing_speed', 'g'}
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=bound)

        # Keys are an array
        keys = np.array(['thrower_height', 'throwing_speed', 'g'])
        m.estimate_params(data, keys, bounds=bound)

        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Reset parameters after successful estimate_params call
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        
        # Keys within a list of tuples without any commas
        # Same as writing - ['thrower_height', 'throwing_speed', 'g']
        keys = [('thrower_height'), ('throwing_speed'), ('g')]
        m.estimate_params(data, keys, bounds=bound)  
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 4)

        # Testing keys within tuples that are a length of one
        keys = [('thrower_height', ), ('throwing_speed', ), ('g', )]
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=bound)

        # Testing lists within lists
        keys = [['thrower_height'], ['throwing_speed'], ['g']]
        with self.assertRaises(TypeError):
            m.estimate_params(data, keys, bounds=bound)


    def test_parameters(self):
        """
        Testing if passing in other keyword arguments works as intended
        """

        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        data = [(results.times, results.inputs, results.outputs)]
        gt = m.parameters.copy()

        # Now lets reset some parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        keys = ['thrower_height', 'throwing_speed', 'g']
        bound=((0, 8), (20, 42), (-20, -5))

        # Note that not every one of the keys would comply with this system
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.parameters['g'] = -5
        m.estimate_params(data, keys, bounds=bound, method='TNC')

        saveError = m.calc_error(results.times, results.inputs, results.outputs)

        # Testing that default method passed in works as intended
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.parameters['g'] = -5
        
        m.estimate_params(data, keys, bounds=bound)

        self.assertNotEqual(saveError, m.calc_error(results.times, results.inputs, results.outputs))

        # Passing 'TNC' method along with 'maxfun 1000' as our options
        # to see what happens if there are changed when we pass in 
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.parameters['g'] = -5
        m.estimate_params(data, keys, bounds=bound, method='TNC', options={'maxfun': 1000, 'disp': False})
        
        self.assertGreater(saveError, m.calc_error(results.times, results.inputs, results.outputs))

        # Passing in Method that does not exist
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=bound, method='madeUpName')

        # Reset incorrect parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        keys = ['thrower_height', 'throwing_speed']
        bound=((0, 8), (20, 42))
                     
        # Not setting up 'maxiter' and/or 'disp'
        # Needs to be str: int format.
        with self.assertRaises(TypeError):
            m.estimate_params(data, keys, bounds=bound, method='Powell', options= {1:2, True:False})
        with self.assertRaises(TypeError):
            m.estimate_params(data, keys, bounds=bound, method='Powell', options={'maxiter': '9999', 'disp': False})

        # Key values for options that do not exist are heavily method dependent
        m.estimate_params(data, keys, bounds=bound, options={'1':2, '2':2, '3':3})
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        saveError = m.calc_error(results.times, results.inputs, results.outputs)

        # Resetting parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        # maxiter is set to -1 here, thus, we are not going to get to a close answer
        m.estimate_params(data, keys, bounds=bound, options={'maxiter': -1, 'disp': False})

        self.assertNotEqual(saveError, m.calc_error(results.times, results.inputs, results.outputs))

        #Resetting parameters for next estimate_params function call
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        # Passing in arbitrary options should not error that follow our format.
        m.estimate_params(data, keys, bounds=bound, options= {'1':3, 'disp':1})
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        self.assertAlmostEqual(saveError, m.calc_error(results.times, results.inputs, results.outputs), delta = 0.00001)

    def test_multiple_runs(self):
        """
        In this test, we are examining the behavior of estimate_params when there are multiple runs.
        """
        m = ThrownObject()

        # The value of time1, time2, inputs, and outputs are arbitrary values

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
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[time1, time2], inputs=inputs, outputs=[outputs])
        self.assertEqual(
            'Times, inputs, and outputs must be same length. Length of times: 2, Length of inputs: 2, Length of outputs: 1',
            str(cm.exception)
        )

        # Adding another wrapper list around times. List error, 1, 2, 2 will result.
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[[time1, time2]], inputs=inputs, outputs=outputs)
        self.assertEqual(
            'Times, inputs, and outputs must be same length. Length of times: 1, Length of inputs: 2, Length of outputs: 2',
            str(cm.exception)
        )

        incorrectTimesRunsLen = [[0, 1, 2, 4, 5, 6, 7, 8, 9]]

        # Passing in only one run for Times whereas inputs and outputs have two runs
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=incorrectTimesRunsLen, inputs=inputs, outputs=outputs)
        self.assertEqual(
            'Times, inputs, and outputs must be same length. Length of times: 1, Length of inputs: 2, Length of outputs: 2',
            str(cm.exception)
        )

        incorrectTimesLen = [[0, 1, 2, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]

        # Passing in the correct amount of runs, but one of the runs has a different length compared to other parameter's lengths
        # This test is also valdiating if we can see which run has a wrong error.
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=incorrectTimesLen, inputs=inputs, outputs=outputs)
        self.assertEqual(
            'Times, inputs, and outputs must be same length for the run at index 1. Length of times: 5, Length of inputs: 4, Length of outputs: 4',
            str(cm.exception)
        )

        # Incorrect lengths for times.
        with self.assertRaises(ValueError):
            m.estimate_params(times=[time1, [time2]], inputs=inputs, outputs=outputs)
        self.assertEqual(
            'Times, inputs, and outputs must be same length for the run at index 1. Length of times: 5, Length of inputs: 4, Length of outputs: 4',
            str(cm.exception)
        )
        # Wrapper list becomes a set.
        timesSet = [time1, time2]
        # Unhashable type for outputs
        with self.assertRaises(TypeError):
            m.estimate_params(times=set(timesSet), inputs=inputs, outputs=set(outputs))
        
        time1 = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9])
        time2 = [0, 1, 2, 3]

        # Confirming estimate_params works when different runs are passed in as different data types.
        # Passing in time1 as a np.array datatype.
        m.estimate_params(times=[time1, time2], inputs=inputs, outputs=outputs)

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
            np.array([
                {'x': 1.83},
                {'x': 36.95},
                {'x': 62.36},
                {'x': 77.81},
            ])]

        # Passing in an np.array structure as a runs for outputs.
        m.estimate_params(times=[time1, time2], inputs=inputs, outputs=outputs)

        # Test case that would be fixed with future changes to Containers
        # with self.assertRaises(ValueError):
        #     m.estimate_params(times=[incorrectTimesLen], inputs=[inputs], outputs=[outputs])


    def test_incorrect_lens(self):
        """
        Goal of this test is to compare the results of passing in incorrect lenghts into our estimate_params call.

        Furthermore, checks incorrect lenghts within multiple runs.
        """
        # Initalizing our model
        m = ThrownObject(process_noise = 0, measurement_noise = 0)
        results = m.simulate_to_threshold(save_freq=0.5)
        gt = m.parameters.copy()

        # Defined Keys and Bounds for estimate_params
        keys = ['thrower_height', 'throwing_speed']
        
        m = ThrownObject()

        # Defining wrongIntuptLen to test parameter length tests.
        wrongInputLen = [{}]*8
        wrongData = [(results.times, wrongInputLen, results.outputs)]
        
        # Wrong Input Length and Values
        with self.assertRaises(ValueError):
            m.estimate_params(times=[results.times], inputs=[wrongInputLen], outputs=[results.outputs])
        
        # Same as last example but times not in wrapper list
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=[wrongInputLen], outputs=[results.outputs])
        
        # Passing in Runs directly
        with self.assertRaises(ValueError):
            m.estimate_params(wrongData)

        # Defining wrongOutputs to test parameter length tests.
        # Arbitrary Outputs created here.
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

        # Formatting incorrect Output data.
        wrongData = [(results.times, results.inputs, wrongOutputs)]

        # Wrong outputs parameter length
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[results.times], inputs=[results.inputs], outputs=[wrongOutputs])
        self.assertEqual(
            'Times, inputs, and outputs must be same length for the run at index 0. Length of times: 9, Length of inputs: 9, Length of outputs: 8',
            str(cm.exception)
        )

        # Both inputs and outputs with incorrect lenghts
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[results.times], inputs=[wrongInputLen], outputs=[wrongOutputs])
        self.assertEqual(
            'Times, inputs, and outputs must be same length for the run at index 0. Length of times: 9, Length of inputs: 8, Length of outputs: 8',
            str(cm.exception)
        )

        # Without wrapper
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=results.times, inputs=wrongInputLen, outputs= wrongOutputs)
        self.assertEqual(
            'Times, inputs, and outputs must be same length for the run at index 0. Length of times: 9, Length of inputs: 8, Length of outputs: 8',
            str(cm.exception)
        )

        # Length error expected, 1, 9, 1.
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[[results.times]], inputs=[results.inputs], outputs=[[results.outputs]])
        self.assertEqual(
            'Times, inputs, and outputs must be same length for the run at index 0. Length of times: 1, Length of inputs: 9, Length of outputs: 1',
            str(cm.exception)
        )


        # Passing in incorrect Runs
        with self.assertRaises(ValueError):
            m.estimate_params(wrongData)

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        # Testing functionality works without having a parent wrapper
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs)
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Differnet types of wrappers should not affect function
        # Testing functionality works with having only a few parameters defined in wrapper sequences
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        m.estimate_params(times=results.times, inputs=(results.inputs), outputs=(results.outputs))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        
        # Same as last test, but inputs is has tuple wrapper sequence insteads.
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        m.estimate_params(times=results.times, inputs=(results.inputs), outputs=[results.outputs])
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Cannot pass in Sets
        with self.assertRaises(TypeError):
            m.estimate_params(times=set(results.times), inputs=results.inputs, outputs=[results.outputs])

        # Missing inputs.
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[results.times], outputs=[results.outputs])
        self.assertEqual(
            'Missing keyword arguments inputs',
            str(cm.exception)
        )

        #  'input' is not a parameter, so techincally not defining the parameter inputs.
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[results.times], input=[results.inputs], outputs=[results.outputs]) 
        self.assertEqual(
            'Missing keyword arguments inputs',
            str(cm.exception)
        )

        # Will work in future case, but not at the current moment
        # with self.assertRaises(ValueError)
            # m.estimate_params(times=[[times]], inputs=[[inputs]], outputs=[[outputs]])


# Test that specifcally looks into adding tolerance into our keyword arguments.
    def test_tolerance(self):
        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        gt = m.parameters.copy()
        keys = ['thrower_height', 'throwing_speed', 'g']

        # x0 violates bounds for 'TNC' method... Not behavior that we want to keep.
        bound = ((0, 4), (24, 42), (-20, -5))

        # there seems to be a maximum tolernace before no changes occur
        bound = ((0, 4), (24, 42), (-20, 10))

        # works as intended for everthing
        bound = ((-15, 15), (24, 42), (-20, 10))

        # Includes the correct amount of bounds needed. Might not even use for sake of how large everything is.
        # Now lets reset some parameters
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10

        # High tolerance would result in a higher calc_error
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs,
                          bounds=bound, keys = keys, tol = 100)
        if not all(abs(m.parameters[key] - gt[key]) > 0.02 for key in keys):
            raise ValueError("m.parameter shouldn't be too close to the original parameters")
        check = m.calc_error(results.times, results.inputs, results.outputs)

        # Reset parameters
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10

        # Not including bounds works as intended here, whereas including bounds does not get a good fit for the parameters.
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = 1e-9)
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        check2 = m.calc_error(results.times, results.inputs, results.outputs)

        self.assertLess(check2, check)

        # Note that tolerance does not convert here
        with self.assertRaises(TypeError):
            m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, 
                              bounds=bound, keys = keys , tol = "12")

        # When tolerance is in a list, it rasies a ValueError.
        # with self.assertRaises(TypeError):
        #     m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, 
        #                       bounds=bound, keys = keys , tol = [(1)])             

        # These cases are working?
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = [])
        hold1 = m.calc_error(results.times, results.inputs, results.outputs)

        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10

        # Tolerance works as intended as long as it is within a sequence of length 1
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = [15])
        hold2 = m.calc_error(results.times, results.inputs, results.outputs)

        # self.assertEqual(hold2, check2)
        self.assertNotEqual(hold1, hold2)

        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = 15)
        hold1 = m.calc_error(results.times, results.inputs, results.outputs)

        self.assertEqual(hold1, hold2)

        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = (15))
        hold2 = m.calc_error(results.times, results.inputs, results.outputs)

        self.assertEqual(hold1, hold2)

        # Cannot pass Sets into tolerance
        with self.assertRaises(TypeError):
            m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = {1})

        # Works when tolerance is a floating point value.

        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = 1.123543297)
        if not all(abs(m.parameters[key] - gt[key]) > 0.02 for key in keys):
            self.fail("m.parameter shouldn't be too close to the original parameters")

        # When tolerance is 
        with self.assertRaises(ValueError):
            m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = [1, 2, 3])

        # 
        with self.assertRaises(TypeError):
            m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = [1, '2', '3'])
        
        with self.assertRaises(ValueError):
            m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = (1, 2, 3))
        
        # Using TNC
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10

        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys=keys, method = 'TNC', tol = 1e-9)
        track1 = m.calc_error(results.times, results.inputs, results.outputs)

        # Defaut Values 
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10

        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = 1e-9)
        track2 = m.calc_error(results.times, results.inputs, results.outputs)
        
        self.assertNotAlmostEqual(track1, track2)

        # Reset parameters
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10

        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, 
                          bounds=bound, keys=keys, method = 'TNC', tol = 1e-4)
        track1 = m.calc_error(results.times, results.inputs, results.outputs)

        # So, for tol values between 1e-4 and bigger, we will continnue to have the same results, whereas. 
        # To detmerine it is 1e-34is the smallest, we have a tolernace check if it 1e-3 produces similar results, to which it does not.

        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        
        # Providing bounds here has some unwanted behavior. Provides worse preidictions... Just a result of having tolerance
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, 
                          bounds=bound, keys=keys, method='TNC', options = {'maxiter': 250}, tol=1e-9)
        
        track2 = m.calc_error(results.times, results.inputs, results.outputs)

        # Note that at some point, the tolerance does not keep going farther down, the tolerance does not affect thie calc_error
        self.assertNotEqual(track1, track2)


        # Anyways, here are tests that are checking for how tolerance and options work alongside one another
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs,
                           bounds=bound, keys=keys, tol = 1e-2)
        override1 = m.calc_error(results.times, results.inputs, results.outputs)

        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs,
                            bounds=bound, keys=keys, tol = 1e-5)
        override2 = m.calc_error(results.times, results.inputs, results.outputs)

        self.assertNotEqual(override1, override2)

        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs,
                           bounds = bound, keys=keys, tol = 1e-2, options={'xatol': 1e-5})
        override3 = m.calc_error(results.times, results.inputs, results.outputs)

        # The passed in options properly overrides the tolerance that is placed. 
        self.assertNotEqual(override1, override3)
        self.assertEqual(override2, override3)


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
    import cProfile
    cProfile.run('main()', "output.dat")

    import pstats
    from pstats import SortKey

    with open("output_time.txt", 'w') as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("output_calls.txt", 'w') as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()

    # main()
