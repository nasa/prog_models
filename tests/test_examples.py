# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from io import StringIO 
import sys
from examples import *

class TestExamples(unittest.TestCase):
    def test_sim_example(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        # Run example
        sim.run_example()

        # Reset stdout 
        sys.stdout = _stdout

    def test_sim_battery_eol_example(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        # Run example
        sim_battery_eol.run_example()

        # Reset stdout 
        sys.stdout = _stdout

    def test_benchmark_example(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        # Run example
        benchmarking.run_example()

        # Reset stdout 
        sys.stdout = _stdout

    def test_new_model_example(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        # Run example
        new_model.run_example()

        # Reset stdout 
        sys.stdout = _stdout

    def test_sensitivity_example(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        # Run example
        sensitivity.run_example()

        # Reset stdout 
        sys.stdout = _stdout

    def test_model_gen_example(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        # Run example
        model_gen.run_example()

        # Reset stdout 
        sys.stdout = _stdout
    
    def test_noise_example(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        # Run example
        noise.run_example()

        # Reset stdout 
        sys.stdout = _stdout

    def test_future_loading(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        # Run example
        # future_loading.run_example()

        # Reset stdout 
        sys.stdout = _stdout

    def test_param_est(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        # Run example
        param_est.run_example()

        # Reset stdout 
        sys.stdout = _stdout
