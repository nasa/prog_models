# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
import unittest

from prog_models import *
from prog_models.models import *


class TestCalcError(unittest.TestCase):
    """
    Main Testing class for calc_error.

    Validating that values are correctly being passed into the new calc_error calls and that we are receiving expected results!
    """

    def test_DTW(self):
        """
        Results from calc_error of DTW work as intended.
        """
        m = LinearThrownObject()

        # Consistent results simulated.
        times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
        inputs = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        outputs = [{'x': 2.83}, {'x': 22.868664648480163}, {'x': 40.422881456712254}, {'x': 55.51861687290881}, {'x': 68.06865643702567}, 
                   {'x': 78.16641111234323}, {'x': 85.89550327176332}, {'x': 91.18868647545982}, {'x': 94.01376508127296}, 
                   {'x': 94.31711597903195}, {'x': 92.22337598299377}, {'x': 87.67210201789473}, {'x': 80.62858869729064}, {'x': 71.10796509926787}, 
                   {'x': 59.19579056829866}, {'x': 44.79567793740186}, {'x': 27.97245305860176}, {'x': 8.736607826437163}, {'x': -12.879687324031048}]

        # Compare calc_error DTW method to another validated DTW algorithm
        value = m.calc_error(times, inputs, outputs, method = 'dtw')
        self.assertEqual(value, 4.8146507570483195)

        # Testing case of inconsistent simulation of data (values from times and outputs have been deleted to substantially increase error)
        times = [0.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 4.5, 5.0, 6.5, 7.5, 8.0, 8.5, 9.0]
        inputs = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        outputs = [{'x': 2.83}, {'x': 40.422881456712254}, {'x': 55.51861687290881}, {'x': 68.06865643702567}, 
                   {'x': 78.16641111234323}, {'x': 85.89550327176332}, {'x': 91.18868647545982}, {'x': 94.01376508127296}, 
                   {'x': 94.31711597903195}, {'x': 87.67210201789473}, {'x': 44.79567793740186}, 
                   {'x': 27.97245305860176}, {'x': 8.736607826437163}, {'x': -12.879687324031048}]
        
        DTW_err = m.calc_error(times, inputs, outputs, method='dtw')
        self.assertEqual(DTW_err, 79.86516870872538)

        # Since we have deleted a few values such that the results from times and outputs may not necessarily match,
        # DTW would match simulated and observed data to each other's closest counterparts. 
        # As such, DTW would have a naturally lower error than a standard error calculation method like 'mse'.
        MSE_err = m.calc_error(times, inputs, outputs)
        self.assertLess(DTW_err, MSE_err)


def run_tests():
    unittest.main()
    
def main():
    import cProfile, pstats
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Base Models")
    profiler = cProfile.Profile()

    profiler.enable()
    result = runner.run(l.loadTestsFromTestCase(TestCalcError)).wasSuccessful()
    profiler.disable()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
