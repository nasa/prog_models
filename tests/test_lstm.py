# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

import unittest

from prog_models.lstm_model import LSTMStateTransitionModel


class TestLSTM(unittest.TestCase):
    def test_simple_case(self):
        # model = LSTMStateTransitionModel(None)
        # self.assertEqual(model.inputs, [])
        # self.assertEqual(model.states, [])
        # self.assertEqual(model.outputs, [])

    def test_improper_input(self):
        pass