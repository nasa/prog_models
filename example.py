from prog_models import model, prognostics_model
from prog_models.models import battery_circuit

batt = battery_circuit.BatteryCircuit()
x=batt.initialize([], [])
print(x)
print(batt.state(0, x, {'i': 0}, 0.1))
print(batt.output(0, x))
print(batt.event_state(0, x))
print(batt.threshold_met(0, x))

def future_loading(t):
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
    return {'i': i}

result = batt.simulate_to_threshold(future_loading, {'t': 18.95, 'v': 4.183})

for i in range(len(result['t'])):
    print("Time: {}\n\tInput: {}\n\tState: {}\n\tOutput: {}\n\tEvent State: {}\n".format(result['t'][i], result['u'][i], result['x'][i], result['z'][i], result['event_state'][i]))

result = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})

for i in range(len(result['t'])):
    print("Time: {}\n\tInput: {}\n\tState: {}\n\tOutput: {}\n\tEvent State: {}\n".format(result['t'][i], result['u'][i], result['x'][i], result['z'][i], result['event_state'][i]))