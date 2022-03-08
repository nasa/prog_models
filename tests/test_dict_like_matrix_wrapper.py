# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
import numpy as np

from prog_models.utils.containers import DictLikeMatrixWrapper
from prog_models import ProgModelTypeError


class TestDictLikeMatrixWrapper(unittest.TestCase):
    def _checks(self, c1):
        self.assertListEqual(c1.keys(), ['a', 'b'])
        self.assertListEqual(list(c1.values()), [1, 2])
        self.assertListEqual(list(c1.items()), [('a', 1), ('b', 2)])
        self.assertEqual(c1['a'], 1)
        self.assertEqual(c1['b'], 2)
        self.assertTrue((c1.matrix == np.array([[1], [2]])).all())

        # Membership
        self.assertIn('a', c1)
        self.assertIn('b', c1)
        self.assertNotIn('c', c1)
        self.assertEqual(len(c1), 2)

        # Setting by dict
        c1['a'] = -1
        self.assertTrue((c1.matrix == np.array([[-1], [2]])).all())
        self.assertEqual(c1['a'], -1)
        
        # Setting by matrix
        c1.matrix[1][0] = -2
        self.assertTrue((c1.matrix == np.array([[-1], [-2]])).all())
        self.assertEqual(c1['b'], -2)

        # Pickling
        import pickle
        c2 = pickle.loads(pickle.dumps(c1))
        self.assertTrue((c2.matrix == np.array([[-1], [-2]])).all())
        self.assertEqual(c2['a'], -1)
        self.assertEqual(c2['b'], -2)

        # Equality
        c2 = DictLikeMatrixWrapper(['a', 'b'], {'a': -1, 'b': -2})
        self.assertEqual(c1, c2)

    def test_dict_init(self):
        c1 = DictLikeMatrixWrapper(['a', 'b'], {'a': 1, 'b': 2})
        self._checks(c1)
    
    def test_array_init(self):
        c1 = DictLikeMatrixWrapper(['a', 'b'], np.array([[1], [2]]))
        self._checks(c1)

    def test_matrix_init(self):
        c1 = DictLikeMatrixWrapper(['a', 'b'], np.matrix([[1], [2]]))
        self._checks(c1)

    def test_broken_init(self):
        with self.assertRaises(ProgModelTypeError):
            DictLikeMatrixWrapper(['a', 'b'], [1, 2])

# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Base Models")
    result = runner.run(l.loadTestsFromTestCase(TestDictLikeMatrixWrapper)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
