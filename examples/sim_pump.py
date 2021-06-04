# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from prog_models.models import CentrifugalPump

def run_example(): 
    pump = CentrifugalPump(process_noise= 0)

    cycle_time = 3600
    def future_loading(t, x=None):
        t = t % cycle_time
        if t < cycle_time/2.0:
            V = 471.2389
        elif t < cycle_time/2 + 100:
            V = 471.2389 + (t-cycle_time/2)
        elif t < cycle_time - 100:
            V = 571.2389
        else:
            V = 471.2398 - (t-cycle_time)

        return {
            'Tamb': 290,
            'V': V,
            'pdisch': 928654, 
            'psuc': 239179, 
            'wsync': V * 0.8
        }

    (times, inputs, states, outputs, event_states) = pump.simulate_to_threshold(future_loading, pump.output(pump.initialize(future_loading(0),{})))
    pump.parameters['x0']['wA'] = 1e-2
    pump.parameters['x0']['wThrust'] = 1e-10

    for i in range(len(times)): # Print Results
            print("Time: {}\n\tInput: {}\n\tState: {}\n\tOutput: {}\n\tEvent State: {}\n".format(times[i], inputs[i], states[i], outputs[i], event_states[i]))

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
