from prog_models.models import BatteryCircuit
import pandas as pd

batt = BatteryCircuit()

batt.parameters['qMax'] = 7856
batt.parameters[
    'process_noise'] = 0  # Note: by default there is some process noise- this turns it off. Noise will be explained later in the tutorial

from pprint import pprint

print('Model configuration:')
pprint(batt.parameters)

import pickle

pickle.dump(batt.parameters, open('/Users/mstrautk/Desktop/prog_models/tutorial/battery123.cfg', 'wb'))
batt.parameters = pickle.load(open('/Users/mstrautk/Desktop/prog_models/tutorial/battery123.cfg', 'rb'))

print('inputs: ', batt.inputs)
print('outputs: ', batt.outputs)
print('event(s): ', batt.events)
print('states: ', batt.states)


def future_loading(t, x=None):
    # Variable (piece-wise) future loading scheme
    # Note: The standard interface for a future loading function is f(t, x)
    #    State (x) is set to None by default because it is not used in this future loading scheme
    #    This allows the function to be used without state (e.g., future_loading(t))
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
    # Since loading is an input to the model, we use the InputContainer for this model
    return batt.InputContainer({'i': i})


time_to_simulate_to = 200
sim_config = {
    'save_freq': 20,
    'print': True  # Print states - Note: is much faster without
}
(times, inputs, states, outputs, event_states) = batt.simulate_to(time_to_simulate_to, future_loading, **sim_config)
from prog_models.models import BatteryCircuit
import pandas as pd

batt = BatteryCircuit()

batt.parameters['qMax'] = 7856
batt.parameters[
    'process_noise'] = 0  # Note: by default there is some process noise- this turns it off. Noise will be explained later in the tutorial

from pprint import pprint

print('Model configuration:')
pprint(batt.parameters)

import pickle

pickle.dump(batt.parameters, open('/Users/mstrautk/Desktop/prog_models/tutorial/battery123.cfg', 'wb'))
batt.parameters = pickle.load(open('/Users/mstrautk/Desktop/prog_models/tutorial/battery123.cfg', 'rb'))

"""print('inputs: ', batt.inputs)
print('outputs: ', batt.outputs)
print('event(s): ', batt.events)
print('states: ', batt.states)"""


def future_loading(t, x=None):
    # Variable (piece-wise) future loading scheme
    # Note: The standard interface for a future loading function is f(t, x)
    #    State (x) is set to None by default because it is not used in this future loading scheme
    #    This allows the function to be used without state (e.g., future_loading(t))
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
    # Since loading is an input to the model, we use the InputContainer for this model
    return batt.InputContainer({'i': i})


time_to_simulate_to = 200
sim_config = {
    'save_freq': 20,
    'print': True  # Print states - Note: is much faster without
}
(times, inputs, states, outputs, event_states) = batt.simulate_to(time_to_simulate_to, future_loading, **sim_config)
