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
from prog_models.models.test_models.linear_models import (
    OneInputNoOutputNoEventLM, OneInputOneOutputNoEventLM, OneInputNoOutputOneEventLM, OneInputOneOutputNoEventLMPM)
from prog_models.models.thrown_object import LinearThrownObject
from prog_models.models.test_models.thrown_object_models import defaultParams, wrongInputStorage, wrongTimeValues

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
        keys = ['thrower_height', 'throwing_speed', 'g']
        m.estimate_params(data, keys)
        for key in keys: # using assert not equal also works.
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        
        # self.assertTrue(m.parameters == gt)


    def test_estimate_params(self):
        # Some things to consider, dictionary vs tuple. The function is accepting both it would seem.
        # Personal opinion on which data type to use. Tuples make the most amount of sense.

        # gt copy works as intended. Going to a different object location

        # Potentially include something that focuses on specifically the key value pairs between gt and m.parameters? mayb not

        #MAKE SURE IT WORKS WITH AN ARRAY AND A NUMPY ARRAY.

        # Create a model that will easily raise an Exception with calc_error.

        # Write a test that fails with the examples bug.
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
        # m.estimate_params(data, keys, bounds=((1.243, 4.0), (-12.9234, 3.33333333), (0, 3)))
        # value1 = m.calc_error(results.times, results.inputs, results.outputs)
        # m.estimate_params(data, keys, bounds=((0, 4), (20, 37), (-20, 0)))
        # value5 = m.calc_error(results.times, results.inputs, results.outputs)
        # m.estimate_params(data, keys, bounds=((1.243, 4.0), (-12.9234, 3.33333333), (0, 3)))
        # sanityCheck = m.calc_error(results.times, results.inputs, results.outputs)

        # Need at least one data point
        with self.assertRaises(ValueError):
            m.estimate_params(times=[], inputs=[], outputs=[])
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
        
        m.estimate_params(data, keys, bounds=((0, 4), (20, 37), (-20, 0)))
        check = m.calc_error(results.times, results.inputs, results.outputs)
        m.estimate_params(data, keys, bounds=((1.243, 4.0), (19.13212, 39.12301), (-21.123, 2.000991)))
        check2= m.calc_error(results.times, results.inputs, results.outputs)

        # calc_error is a much bigger value compared to other values. Not including a good range?
        # Should not affect other calls...
        # m.estimate_params(data, keys, bounds=((1.243, 4.0), (-12.9234, 3.33333333), (0, 3)))
        # check3 = m.calc_error(results.times, results.inputs, results.outputs)

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

        # Regardless of type of wrapper sequence around bounds, it should work
        m.estimate_params(data, keys, bounds=[[0, 4], [20, 42], [-4, 15]])

        m.estimate_params(data, keys, bounds=[[0, 4], (20, 42), [-4, 15]])

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

        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=np.array([(1, 2), (2, 3, 4), (4, 5)]))
        
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=np.array([(1, 2), (2, 3), (4,5), (-1, 20)]))

        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=np.array([(1, 2), (2, 3)]))

        with self.assertRaises(TypeError):
            m.estimate_params(data, keys, bounds=np.array('a'))

        with self.assertRaises(TypeError):
            m.estimate_params(data, keys, bounds=np.array(True))

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

        m.estimate_params(data, keys, bounds=(('-3', '12'), ('1', '20'), ('-5', '30')))
        check3 = m.calc_error(results.times, results.inputs, results.outputs)

        # Checking to make sure original equals the previous ones
        self.assertEqual(check, check3)

        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('a', '12'), ('1', '20'), ('-5', '30')))
        
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('-3'), ('1', '20'), ('-5', '30')))
        
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('a', '12'), ('30', '20'), ('-5', '30')))
        
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('-3', '12'), ('20', '30', '40'), ('-5', '30')))

        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('-3', '12'), ('20', '30'), ('-5', '30'), ('-20, 20')))
        
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=(('-3', '12'), ('20', '30')))
        
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[np.array(['a', 's']), np.array([2, 3]), np.array([4, 5])])
        
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[np.array(['a', '9']), np.array(['2', '3']), np.array(['4', '5'])])

        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[np.array(['4', '9']), np.array(['2', '3']), np.array(['4', '5']), np.array(['-2', '4'])])
        
        # Lower Bound greater than Upper Bound
        with self.assertRaises(ValueError):
            m.estimate_params(data, keys, bounds=[np.array(['9', '4']), np.array(['2', '3']), np.array(['4', '5'])])
        
        # Same bounds
        # with self.assertRaises(ValueError):
        m.estimate_params(data, keys, bounds=[np.array(['9', '9']), np.array(['2', '3']), np.array(['4', '5'])])

        m.estimate_params(data, keys, bounds=((1, 1), (2, 4), (-1, 24)))


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
        # with self.assertRaises(TypeError):
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

        m.estimate_params(data, keys, bounds=bound, method='Powell', options={'maxiter': 1e-9, 'disp': False})
        m1.estimate_params(data, keys, bounds=bound, method='Powell', options={'1':2, '2':2, '3':3})
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

        data = [(times, wrongInputLen, outputs)]
        
        with self.assertRaises(ValueError):
            m.estimate_params(times=[times], inputs=[wrongInputLen], outputs=[outputs])
        
        with self.assertRaises(ValueError):
            m.estimate_params(times=times, inputs=[wrongInputLen], outputs=[outputs])
        
        with self.assertRaises(ValueError):
            m.estimate_params(data)

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

        data = [(times, inputs, wrongOutputs)]

        # Wrong outputs parameter length
        with self.assertRaises(ValueError):
            m.estimate_params(times=[times], inputs=[inputs], outputs=[wrongOutputs])

        # Both inputs and outputs with incorrect lenghts
        with self.assertRaises(ValueError):
            m.estimate_params(times=[times], inputs=[wrongInputLen], outputs=[wrongOutputs])

        # Without wrapper
        with self.assertRaises(ValueError):
            m.estimate_params(times=times, inputs=wrongInputLen, outputs= wrongOutputs)

        with self.assertRaises(ValueError):
            m.estimate_params(data)


        # Testing functionality works without having a parent wrapper
        m.estimate_params(times=times, inputs=inputs, outputs=outputs)

        # Differnet types of wrappers should not affect function
        # Testing functionality works with having only a few parameters defined in wrapper sequences
        m.estimate_params(times=times, inputs=[inputs], outputs=[outputs])
        
        # Same as last test, but inputs is has tuple wrapper sequence insteads.
        m.estimate_params(times=times, inputs=(inputs), outputs=[outputs])

        # Same as last test, but times in list wrapper and outputs is around dictionary wrapper.
        m.estimate_params(times=set(times), inputs=inputs, outputs=[outputs])

        m.estimate_params(times=set(times), inputs=(inputs), outputs=[outputs])

        # This fails because inputs and outputs are both dictionaries within a Set. Sometimes, an empty set within a Set.
        # m.estimate_params(times=[times], inputs=inputs, outputs=set(outputs))

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

        # Another test case that would be fixed with future changes to Containers
        # with self.assertRaises(ValueError):
        #     m.estimate_params(times=[incorrectTimesLen], inputs=[inputs], outputs=[outputs])


# mainly a sanity check
    def test_param_estimate(self):
        m = ThrownObject(thrower_height=20)

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

        keys = ['thrower_height', 'throwing_speed']

        m.estimate_params([(times, inputs, outputs)], keys, dt=0.01)
        
        value1 = m.calc_error(times, inputs, outputs, dt=1e-4)

        # After calling calc_error, we expect the total value to be less than 1.
        # This is taken from the examples
        self.assertLess(value1, 1)

        # Calling calc_error on the same object should not change the given value
        value2 = m.calc_error(times, inputs, outputs, dt=1e-4)
        self.assertEqual(value1, value2)

        # Calling estimate_params should not change parameters
        m.estimate_params([(times, inputs, outputs)], keys, dt=0.01)

        # Does not change the values
        value3 = m.calc_error(times, inputs, outputs, dt=1e-4)

        self.assertEqual(value3, value1)

        m.parameters['thrower_speed'] = 50
        

    # @unittest.skip
    def test_big_example(self):
        # Note, lowering timesteps or increasing simulate threshold may cause this model to not run (takes too long)
        m = BatteryElectroChemEOD()

        options = {
            'save_freq': 200, # Frequency at which results are saved
            'dt': 2, # Timestep
        }

        def future_loading(t, x=None):
            if (t < 600):
                i = 2
            elif (t < 900):
                i = 1
            elif (t < 1800):
                i = 4
            elif (t < 3000):
                i = 2     
            else:
                i = 3
            return m.InputContainer({'i': i})
    
        simulated_results = m.simulate_to(200, future_loading, **options)

        value = m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt=2)

        # Creating errors
        m.parameters['qMax'] = 12000
        # m.parameters['VolS'] = 300
        # Division by zero occurs here
        # keys = ['qMax', 'VolS']
        keys = ['qMax']

        # why are these values changing from value?
        error = m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt=2)

        m.estimate_params([(simulated_results.times, simulated_results.inputs, simulated_results.outputs)], keys, dt=0.5)

        value1 = m.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs, dt=2)

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