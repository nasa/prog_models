from prog_models import model, prognostics_model
from prog_models.models import battery_circuit

batt = battery_circuit.BatteryCircuit()
x=batt.initialize([], [])
print(x)
print(batt.state(0, x, {'i': 0}, 0.1))
print(batt.output(0, x))
print(batt.event_state(0, x))
print(batt.threshold_met(0, x))
