# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example shows the use of the advanced feature - matrix models. Matrix models represent the state of the system using matricies instead of dictionaries as are used by `PrognosticsModel`. This is important for some applications like surrogate and machine learned models where the state is represented by a tensor, and operations by matrix operations. Simulation functions propogate the state using the matrix form, preventing the inefficiency of having to convert to and from dictionaries.

In this example, a model is designed to simulate a thrown object using matrix notation (instead of dictionary notation as in the standard model). The implementation of the model is comparable to a standard model, except that, since it is a subclass of MatrixModel, it defines functions as *_matrix() to compute matrix operations within each function.
"""

def run_example():
    from prog_models import MatrixModel
    import numpy as np

    # Define the model
    class ThrownObject(MatrixModel):
        # Define the model properties, this is exactly the same as for a regular model.

        inputs = []  # no inputs, no way to control
        states = [
            'x',    # Position (m) 
            'v'    # Velocity (m/s)
            ]
        outputs = [
            'x'     # Position (m)
        ]
        events = [
            'falling',  # Event- object is falling
            'impact'    # Event- object has impacted ground
        ]

        is_vectorized = True

        # The Default parameters. Overwritten by passing parameters dictionary into constructor
        default_parameters = {
            'thrower_height': 1.83,  # m
            'throwing_speed': 40,  # m/s
            'g': -9.81,  # Acceleration due to gravity in m/s^2
            'process_noise': 0.0  # amount of noise in each step
        }

        # Define the model equations
        # Note: here we are using the *_matrix versions of the functions. This is because we are using matrix operations.
        def initialize_matrix(self, u = None, z = None):
            # Note: states are returned as column vectors
            # Note: states are in the order of model.states, above (x, v)
            return np.array([
                [self.parameters['thrower_height']], 
                [self.parameters['throwing_speed']]], 
                dtype=np.float64)

        def dx_matrix(self, x, u):
            # Note: x is a column vector
            # Note: u is a column vector
            #   and u is in the order of model.inputs, above
            # Note: dx (returned) is a column vector
            return np.array([
                [x[1]], 
                [self.parameters['g']]], 
                dtype=np.float64)
        
        def output_matrix(self, x):
            # Note: x is a column vector
            return np.array([x[0]], dtype=np.float64)

        # This is actually optional. Leaving thresholds_met empty will use the event state to define thresholds.
        #  Threshold = Event State == 0. However, this implementation is more efficient, so we included it
        def threshold_met_matrix(self, x):
            return np.array(
                [[x[1] < 0],
                [x[0] <= 0]],
                dtype=bool)

        def event_state_matrix(self, x): 
            x_max = x[0] + np.square(x[1])/(-self.parameters['g']*2) # Use speed and position to estimate maximum height
            x_max = np.where(x[1] > 0, x[0], x_max) # 1 until falling begins
            return np.array([
                [np.maximum(x[1]/self.parameters['throwing_speed'],0)],  # Throwing speed is max speed
                [np.maximum(x[0]/x_max,0)]  # then it's fraction of height
            ], dtype=np.float64)

    # Now we can use the model
    # Create the model
    thrown_object = ThrownObject()

    # Note that this model can be used with dictionaries like other models:
    # But this is slower since it has to convert back and forth from dictionaries to numpy arrays
    x = {'x': 1.83, 'v': 40}
    print('State at 0.1 seconds: ', thrown_object.next_state(x, {}, 0.1))

    # But it can also be used with numpy arrays:
    x = np.array([1.83, 40])
    print('State at 0.1 seconds: ', thrown_object.next_state_matrix(x, None, 0.1))

    # Now lets use it for simulation. Here the simulate_to_threshold function is using the matrix version, storing internal state information as a matrix, but the results returned are dictionaries
    # this way the same interface is maintained as PrognosticsModel
    thrown_object.simulate_to_threshold(lambda t, x = None: np.array([[]]), print = True, threshold_keys = 'impact', dt = 0.1)

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
