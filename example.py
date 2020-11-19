from prog_models import model, prognostics_model
from prog_models.models import battery_circuit

batt = battery_circuit.BatteryCircuit()

def future_loading(t):
    # Variable (piece-wise) future loading scheme 
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

# simulate for 200 seconds
(times, inputs, states, outputs, event_states) = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})

for i in range(len(times)): # Print Results
    print("Time: {}\n\tInput: {}\n\tState: {}\n\tOutput: {}\n\tEvent State: {}\n".format(times[i], inputs[i], states[i], outputs[i], event_states[i]))

# Simulate to threshold
(times, inputs, states, outputs, event_states) = batt.simulate_to_threshold(future_loading, {'t': 18.95, 'v': 4.183})

for i in range(len(times)): # Print Results
    print("Time: {}\n\tInput: {}\n\tState: {}\n\tOutput: {}\n\tEvent State: {}\n".format(times[i], inputs[i], states[i], outputs[i], event_states[i]))

