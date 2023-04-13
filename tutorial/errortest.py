from copy import deepcopy
from prog_models.utils.containers import DictLikeMatrixWrapper

from tests.test_base_models import MockProgModel
import numpy as np

m = MockProgModel(process_noise = 0.0)
def load(t, x=None):
    return {'i1': 1, 'i2': 2.1}
a = np.array([1, 2, 3, 4, 4.5])
b = np.array([5]*5)
c = np.array([-3.2, -7.4, -11.6, -15.8, -17.9])
t = np.array([0, 0.5, 1, 1.5, 2])
dt = 0.5
x0 = {'a': deepcopy(a), 'b': deepcopy(b), 'c': deepcopy(c), 't': deepcopy(t)}
x = m.next_state(x0, load(0), dt)
print(x.matrix[0].size)
for xa, xa0 in zip(x['a'], a):
    print('xa', xa)
    print('xa0', xa0)
    print('xa0+dt', xa0+dt)
    print(xa == xa0+dt)