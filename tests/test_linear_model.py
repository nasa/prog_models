# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
import unittest

from prog_models.models.test_models.linear_models import *
from prog_models.models.thrown_object import LinearThrownObject, LinearThrownObject_WrongB


# TODO: IF no inputs, parameters should be not be instantied at all. e.g) Having 0 inputs means B will be a 2x0, which is not possible.

# linear_thrown_object comments...

# Figure out why its an attribute error and not a typeerror FOR paramter B. line 55.

# 

class TestLinearModel(unittest.TestCase):
    def test_linear_model(self):
        m = LinearThrownObject()

        # Parameters for reference

        # A = np.array([[0, 1], [0, 0]])
        # E = np.array([[0], [-9.81]])
        # C = np.array([[1, 0]])

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
        # Matrix Dimension Checking
        # when matrix is not proper dimensional (1-D array = C, D, G; 2-D array = A,B,E; None = F;)
        # @A 2x2
        with self.assertRaises(AttributeError):
            m.A = np.array([[0, 1]]) # 1-D array
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.A = np.array([[0, 1], [0, 0], [1, 0]]) # 3-D array
            m.matrixCheck()
        # When Matrix is improperly formed
        with self.assertRaises(AttributeError):
            m.A = np.array([[0, 1, 2, 3], [0, 0, 1, 2]]) # extra column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.A = np.array([[0], [0]]) # less column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.A = np.array([[0, 1], [0, 0], [2, 2]]) # extra row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.A = np.array([[0, 1]]) # less row
            m.matrixCheck()
        # Reset Process
        m.A = np.array([[0, 1], [0, 2]])
        m.matrixCheck()

        # @B
        with self.assertRaises(TypeError):
            m.B = "[[0, 1], [0, 0]]"
            m.matrixCheck()
        with self.assertRaises(TypeError):
            m.B = None # None
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.B = 0 # int
            m.matrixCheck()
        with self.assertRaises(TypeError):
            m.B = 3.14 # float
            m.matrixCheck()
        with self.assertRaises(TypeError):
            m.B = {} # dict
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.B = () # tuple
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.B = set() # set
            m.matrixCheck() 
        with self.assertRaises(TypeError):
            m.B = True # boolean
            m.matrixCheck()
        m.B = 'setDefault' #sets paramter B to defaulted value
        m.matrixCheck()
        # PART 2: Add a test that sees the new runtime check works from linear_model line 73
        # Tests created for @B mainly because we are exploring when optional paramters are inputted.
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
        with self.assertRaises(AttributeError):
            m.C = np.array(2) # 0-D array
            # this provides a 2 dimensional Array???
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.C = np.array([[0, 0], [1, 1]]) # 2-D array
            m.matrixCheck()
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
            # How is this any different from the first test...
            m.matrixCheck()
        m.C = np.array([[1, 0]])
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
        with self.assertRaises(AttributeError):
            m.E = np.array([[0]]) # 1-D array
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.E = np.array([[0], [1], [2]]) # 3-D array
            m.matrixCheck()
        with self.assertRaises(TypeError):
            m.E = np.array([0,0], [-9.81, -1]) # extra column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.E = np.array([[], []]) # less column values per row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.E = np.array([[0, 1], [0, 0], [2, 2]]) # extra row
            m.matrixCheck()
        with self.assertRaises(AttributeError): 
            m.E = np.array([[0, 1]]) # less row
            m.matrixCheck()
        m.E = np.array([0, -9.81])
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
        with self.assertRaises(AttributeError):
            # if F is none, we need to override event_state
            m_noes = FNoneNoEventStateLM()
        m.F = None
        m.matrixCheck()

        # @D 1x
        with self.assertRaises(AttributeError):
            m.D = np.array(1) # 0-D array
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.D = np.array([[0], [1]]) # 2-D array with incorrect values ?????
            m.matrixCheck()
        with self.assertRaises(AttributeError):
            m.D = np.array([1, 2]) # extra column values per row
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
        m.D = 'setDefault' # sets to Default Value
        m.matrixCheck()

        # G Matrix Testing

        # INCORRECT? REMOVE OUTSIDE SET OF [] / ADD OUTSIDE SET OF []
        # Figure out what exactly would error for parameter G

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
        m.G = 'setDefault' # sets to Default Value
        m.matrixCheck()

# Testing when LinearModel is initiated when no inputs are given but matrix B is defined.
    # def test_incorrect_B(self):
    #     with self.assertRaises(AttributeError):
    #         m = LinearThrownObject_WrongB()


        # with self.assertRaises(AttributeError):
        #     m.B = np.array([[]]) # 0-D array
        #     m.matrixCheck()
        # with self.assertRaises(AttributeError):
        #     m.B = np.array([[], [], []]) # 3-D array
        #     m.matrixCheck()
        # with self.assertRaises(AttributeError):
        #     m.B = np.array([[], [], [], []]) # 4-D array
        #     m.matrixCheck()
        #     # @B 2x0
        # with self.assertRaises(AttributeError):
        #     m.B = np.array([[0, 1 ,2]]) # extra column values per row
        #     m.matrixCheck()
        # with self.assertRaises(AttributeError):
        #     m.B = np.array([[0]]) # less column values per row
        #     m.matrixCheck()
        # with self.assertRaises(AttributeError): 
        #     m.B = np.array([[0, 1], [1, 1], [2, 2]]) # extra row
        #     m.matrixCheck()
        # with self.assertRaises(AttributeError): 
        #     m.B = np.array([[0, 1]]) # less row
        #     m.matrixCheck()


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
        # LinearThrown Object defines A, E, C as np arrays, thus we define with python lists instead. 
        
        class ThrownObject(LinearThrownObject):
            A = [[0, 1], [0, 0]]
            E = [0, -9.81]
            C = [[1, 0]]

        m = ThrownObject()
        m.matrixCheck()

        # Testing to see confirm python lists and np arrays have same functionality.
        self.assertIsInstance(m.A, np.ndarray)
        self.assertTrue(np.array_equal(m.A, np.array([[0, 1], [0, 0]])))
        self.assertIsInstance(m.E, np.ndarray)
        self.assertTrue(np.array_equal(m.E, np.array([0, -9.81])))
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
        # Needs more development; test coverage needs testing of event_state not overridden

# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Base Models")
    result = runner.run(l.loadTestsFromTestCase(TestLinearModel)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
