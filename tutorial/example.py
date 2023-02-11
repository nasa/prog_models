from prog_models.models import BatteryCircuit

# create new battery
batt = BatteryCircuit()

# configure
batt.parameters['qMax'] = 7856
batt.parameters['process_noise'] = 0


# formatting the battery information
from pprint import pprint
print('model configuration')
pprint(batt.parameters)

# save configuration using 'pickle'
import pickle
batt1 = BatteryCircuit()
pickle.dump(batt.parameters, open('battery123.cfg', 'wb')) 
batt1.parameters = pickle.load(open('battery123.cfg', 'rb'))

# test
batt1.parameters['qMax'] = 4000
pprint(batt1.parameters)

# inputs and outputs
print('inputs: ', batt.inputs)
print('outputs: ', batt.outputs)

# events that are predicted
print('event(s): ', batt.events)

# internal states of the model
print('states :', batt.states)

# simulating to a specific time, piece-wise function
def future_loading(t, x = None):
    if (t < 600):
        i = 2
        #print(2)
    elif (t < 900):
        i = 1
        #print(1)
    elif (t < 1800):
        i = 4
        #print(4)
    elif (t < 3000):
        i = 2
        #print(2)
    else:
        i = 3
        #print(3)
    # loading is an input to the model, 
    # we use the InputContainer for this model
    return batt.InputContainer({'i': i})

# future loading
#print(future_loading(1040))

# simulate 200 seconds
print()
time_to_simulate_to = 200
sim_config = {
    'save_freq': 20,
    'print': True # Print states - Note: is much faster without
}
(times, inputs, states, outputs, event_states) = batt.simulate_to(time_to_simulate_to, future_loading, **sim_config)
#print(inputs[1]['i'])



# to show

inputs.plot(ylabel='Current drawn (amps)')
event_states.plot(ylabel= 'SOC')
outputs.plot(ylabel= {'v': "voltage (V)", 't': 'temperature (°C)'}, compact= False)



  



