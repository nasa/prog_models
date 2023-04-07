# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest

from prog_models import EnsembleModel
from prog_models.models.test_models.linear_models import OneInputOneOutputOneEventLM


class TestEnsemble(unittest.TestCase):
    def setUp(self):
        # set stdout (so it wont print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_single_model(self):
        # An ensemble model with one model should raise an exception

        m = OneInputOneOutputOneEventLM()
        
        with self.assertRaises(ValueError):
            EnsembleModel([m])

    def test_wrong_type(self):
        # An ensemble model with a non-model should raise an exception

        m = OneInputOneOutputOneEventLM()
        with self.assertRaises(TypeError):
            EnsembleModel(m)
        with self.assertRaises(TypeError):
            EnsembleModel([m, 1])
        with self.assertRaises(TypeError):
            EnsembleModel(77)
        with self.assertRaises(TypeError):
            EnsembleModel([m, m, m, 77])

    def test_two_models_identical(self):
        m = OneInputOneOutputOneEventLM()
        m2 = OneInputOneOutputOneEventLM(x0={'x1': 2})

        # An ensemble model with two models should work
        em = EnsembleModel([m, m2])

        # Since they're identical, the inputs, state, etc. should be the same
        self.assertEqual(em.inputs, m.inputs)
        self.assertEqual(em.states, m.states)
        self.assertEqual(em.outputs, m.outputs)
        self.assertEqual(em.events, m.events)

        # The resulting initial state should be between the two:
        x0 = em.initialize()
        self.assertEqual(x0['x1'], 1)

# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Ensemble models")
    result = runner.run(l.loadTestsFromTestCase(TestEnsemble)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
