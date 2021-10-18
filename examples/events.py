# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example further illustrating the concept of 'events' which generalizes the idea of EOL. 

'Events' is the term used to describe something to be predicted. 
Generally in the PHM community these are referred to as End of Life (EOL). 
However, they can be much more.

In the implementation from the prog_models package, events can be anything that needs to be predicted. 
Events can represent end of life (EOL), end of mission (EOM), warning thresholds, or any event of interest (EOI). 

This example shows how a events can be used in your applications. 
"""
from prog_models.models import BatteryElectroChemEOD

def run_example():
    # Example: Warning thresholds
    # In this example we will use the battery model
    # We of course are interested in end of discharge, but for this example we
    # have a requirement that says the battery must not fall below 5% SOC
    # Additionally, we have two warning thresholds (yellow and red)

    YELLOW_THRESH = 0.15
    RED_THRESH = 0.1

    # Step 1: Extend the battery model to define the additional events
    class MyBatt(BatteryElectroChemEOD):
        events = BatteryElectroChemEOD.events + ['EOD_warn_yellow', 'EOD_warn_red']

        def event_state(self, state):
            # Get event state from parent
            event_state = super().event_state(state)

            # Add yellow and red states by scaling EOD state
            event_state['EOD_warn_yellow'] = (event_state['EOD']-YELLOW_THRESH)/(1-YELLOW_THRESH) 
            event_state['EOD_warn_red'] = (event_state['EOD']-RED_THRESH)/(1-RED_THRESH)

            # Return
            return event_state

        def threshold_met(self, x):
            # Get threshold met from parent
            t_met =  super().threshold_met(x)

            # Add yell and red states from event_state
            event_state = self.event_state(x)
            t_met['EOD_warn_yellow'] = event_state['EOD_warn_yellow'] <= 0
            t_met['EOD_warn_red'] = event_state['EOD_warn_red'] <= 0

            return t_met

    # Step 2: Use it
    # 2a: Setup model
    def future_loading(t, x=None):
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
    
    first_output = {'t': 18.95, 'v': 4.183}
    m = MyBatt()

    # 2b: Simulate to threshold
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_loading, first_output, threshold_keys=['EOD'], print = True)

    # 2c: Plot results
    event_states.plot()
    import matplotlib.pyplot as plt
    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
