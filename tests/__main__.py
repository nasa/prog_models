# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from .test_base_models import main as base_models_main
from .test_sim_result import main as sim_result_main
from .test_dict_like_matrix_wrapper import main as dict_like_matrix_wrapper_main
from .test_examples import main as examples_main
from .test_centrifugal_pump import main as centrifugal_pump_main
from .test_pneumatic_valve import main as pneumatic_valve_main
from .test_battery import main as battery_main
from .test_tutorials import main as tutorials_main
from .test_datasets import main as datasets_main
from .test_powertrain import main as powertrain_main
from .test_surrogates import main as surrogates_main
from .test_data_model import main as lstm_main

from io import StringIO
import matplotlib.pyplot as plt
import sys
from timeit import timeit
from unittest.mock import patch

from examples import sim as sim_example

def _test_ex():
    # Run example
    sim_example.run_example()

if __name__ == '__main__':
    was_successful = True

    try:
        # set stdout (so it wont print)
        sys.stdout = StringIO()
        
        with patch('matplotlib.pyplot.show'):
            runtime = timeit(_test_ex, number=10)
            plt.close('all')

        # Reset stdout 
        sys.stdout = sys.__stdout__
        print(f"\nExample Runtime: {runtime}")
    except Exception as e:
        print("\Benchmarking Failed: ", e)
        was_successful = False

    print("\n\nTesting individual exectution of test files")

    # Run tests individually to test them and make sure they can be executed individually
    try:
        base_models_main()
    except Exception:
        was_successful = False

    try:
        sim_result_main()
    except Exception:
        was_successful = False

    try:
        examples_main()
    except Exception:
        was_successful = False
        
    try:
        battery_main()
    except Exception:
        was_successful = False

    try:
        centrifugal_pump_main()
    except Exception:
        was_successful = False

    try:
        pneumatic_valve_main()
    except Exception:
        was_successful = False

    try:
        dict_like_matrix_wrapper_main()
    except Exception:
        was_successful = False

    try:
        tutorials_main()
    except Exception:
        was_successful = False

    try:
        datasets_main()
    except Exception:
        was_successful = False
        
    try:
        powertrain_main()
    except Exception:
        was_successful = False

    try:
        surrogates_main()
    except Exception:
        was_successful = False

    try:
        lstm_main()
    except Exception:
        was_successful = False

    if not was_successful:
        raise Exception("Failed test")
