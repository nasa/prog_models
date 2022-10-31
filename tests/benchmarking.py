# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import time

FORMAT_STR = '| {:40s} |'
CLOCK = time.process_time

if __name__ == "__main__":
    print('| Test | Time (s) |\n| --- | --- |')

    print(FORMAT_STR.format('import main'), end='')
    t = CLOCK()
    import prog_models
    t2 = CLOCK()
    print(f'{t2-t} |')

    print(FORMAT_STR.format('import thrown object'), end='')
    t = CLOCK()
    from prog_models.models import ThrownObject
    t2 = CLOCK()
    print(f'{t2-t} |')

    print(FORMAT_STR.format('model initialization'), end='')
    t = CLOCK()
    m = ThrownObject()
    t2 = CLOCK()
    print(f'{t2-t} |')

    print(FORMAT_STR.format('set noise'), end='')
    t = CLOCK()
    m.parameters['process_dist'] = 'none'
    t2 = CLOCK()
    print(f'{t2-t} |')

    def future_load(t, x=None):
        return m.InputContainer({})

    print(FORMAT_STR.format('simulate'), end='')
    t = CLOCK()
    m.simulate_to_threshold(future_load, threshold_keys='impact')
    t2 = CLOCK()
    print(f'{t2-t} |')

    print(FORMAT_STR.format('simulate with saving'), end='')
    t = CLOCK()
    m.simulate_to_threshold(future_load, threshold_keys='impact', save_freq = 0.5)
    t2 = CLOCK()
    print(f'{t2-t} |')

    print(FORMAT_STR.format('simulate with saving, dt'), end='')
    t = CLOCK()
    m.simulate_to_threshold(future_load, threshold_keys='impact', save_freq = 0.5, dt=0.1)
    t2 = CLOCK()
    print(f'{t2-t} |')

    print(FORMAT_STR.format('simulate with printing results, dt'), end='')
    temp_out = StringIO()
    sys.stdout = temp_out
    t = CLOCK()
    m.simulate_to_threshold(future_load, threshold_keys='impact', save_freq = 0.5, dt=0.1, print = True)
    t2 = CLOCK()
    sys.stdout = sys.__stdout__
    print(f'{t2-t} |')
    
    result = m.simulate_to_threshold(future_load, threshold_keys='impact', save_freq = 0.5, dt=0.1)

    print(FORMAT_STR.format('Plot results'), end='')
    t = CLOCK()
    result.outputs.plot()
    t2 = CLOCK()
    print(f'{t2-t} |')

    print(FORMAT_STR.format('Metrics'), end='')
    t = CLOCK()
    result.event_states.monotonicity()
    t2 = CLOCK()
    print(f'{t2-t} |')

    print(FORMAT_STR.format('Surrogate Model Generation'), end='')
    temp_out = StringIO()
    sys.stdout = temp_out
    sys.stderr = temp_out
    t = CLOCK()
    m2 = m.generate_surrogate([future_load], threshold_keys='impact')
    t2 = CLOCK()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f'{t2-t} |')

    print(FORMAT_STR.format('surrogate sim'), end='')
    t = CLOCK()
    m2.simulate_to_threshold(future_load, threshold_keys='impact')
    t2 = CLOCK()
    print(f'{t2-t} |')

    print(FORMAT_STR.format('surrogate sim, dt'), end='')
    t = CLOCK()
    m2.simulate_to_threshold(future_load, threshold_keys='impact', save_freq=0.25)
    t2 = CLOCK()
    print(f'{t2-t} |')

