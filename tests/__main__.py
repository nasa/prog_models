# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from tests.test_base_models import main as base_models_main
from tests.test_sim_result import main as sim_result_main
from tests.test_dict_like_matrix_wrapper import main as dict_like_matrix_wrapper_main
from tests.test_examples import main as examples_main
from tests.test_battery import main as battery_main
from tests.test_centrifugal_pump import main as centrifugal_pump_main
from tests.test_pneumatic_valve import main as pneumatic_valve_main
from tests.test_tutorials import main as tutorials_main
from tests.test_datasets import main as datasets_main
from tests.test_powertrain import main as powertrain_main
from tests.test_surrogates import main as surrogates_main
from tests.test_data_model import main as lstm_main
from tests.test_direct import main as direct_main
from tests.test_linear_model import main as linear_main
from tests.test_battery import main as battery_main
from tests.test_composite import main as composite_main
from tests.test_serialization import main as serialization_main
from tests.test_estimate_params import main as estimate_params_main
from tests.test_ensemble import main as ensemble_main
from tests.test_uav_model import main as uav_main

if __name__ == '__main__':
    was_successful = True
    print("\n\nTesting individual execution of test files")

    # Run tests individually to test them and make sure they can be executed individually
    try:
        base_models_main()
    except Exception:
        was_successful = False

    try:
        direct_main()
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

    try:
        linear_main()
    except Exception:
        was_successful = False
    
    try:
        composite_main()
    except Exception:
        was_successful = False

    try:
        serialization_main()
    except Exception:
        was_successful = False

    try:
        ensemble_main()
    except Exception:
        was_successful = False

    try:
        estimate_params_main()
    except Exception:
        was_successful = False

    try:
        uav_main()
    except Exception:
        was_successful = False

    if not was_successful:
        raise Exception("Failed test")
