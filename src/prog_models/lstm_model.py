# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from warnings import warn

from . import PrognosticsModel
from .sim_result import SimResult


class LSTMStateTransitionModel(PrognosticsModel):
    """
    A State Transition Model with no events using an Keras LSTM Model.
    State transition models map form the inputs at time t and outputs at time t-1 plus historical data to the outputs at time t

    Args:
        model (keras.Model): Keras model to use for state transition

    Keyword Args:
        inputs (list[str]): List of input keys
        outputs (list[str]): List of output keys

    See Also:
        LSTMStateTransitionModel.from_data

    TODO: Examples
    """
    default_params = {
    }

    def __init__(self, model, **kwargs):
        input_shape = model.input.shape

        self.outputs = kwargs.get('outputs', [f'z{i}' for i in range(model.output.shape[1])])
        # Outputs from the last step are part of input
        self.inputs = [f'{z_key}_t-1' for z_key in self.outputs]
        input_keys = kwargs.get('inputs', [f'u{i}' for i in range(input_shape[2]-len(self.outputs))])
        self.inputs.extend(input_keys)
        
        # States are in format [u_t-n+1, z_t-n, ..., u_t, z_t-1]
        self.states = []
        for j in range(input_shape[1]-1, -1, -1):
            self.states.extend([f'{input_i}_t-{j}' for input_i in input_keys])
            self.states.extend([f'{output_i}_t-{j+1}' for output_i in self.outputs])

        kwargs['sequence_length'] = model.input.shape[1]

        super().__init__(**kwargs)

        self.model = model

    def initialize(self, u=None, z=None):
        return self.StateContainer(np.array([[None] for _ in self.states]))

    def next_state(self, x, u, dt):
        # Rotate new input into state
        input_data = u.matrix
        states = x.matrix[len(input_data):]
        return self.StateContainer(np.vstack((states, input_data)))

    def output(self, x):
        if x.matrix[0,0] is None:
            warn(f"Output estimation is not available until at least {1+self.parameters['sequence_length']} timesteps have passed.")
            return self.OutputContainer(np.array([[None] for _ in self.outputs]))
        m_input = x.matrix[:self.parameters['sequence_length']*len(self.inputs)].reshape(self.parameters['sequence_length'], len(self.inputs))
        m_output = self.model(m_input.T)
        return self.OutputContainer(m_output.numpy().T)

    @staticmethod
    def _pre_process_data(data, sequence_length, **kwargs):
        # Data is a List[Tuple] or Tuple (where Tuple is equivilant to [Tuple]))
        # Tuple is (input, output)

        u_all = []
        z_all = []
        for (u, z) in data:
            # Each item (u, z) is a 1-d array, a 2-d array, or a SimResult

            # Process Input
            if isinstance(u, SimResult):
                if len(u[0].keys()) == 0:
                    u = []
                else:
                    u = np.array([u_i.matrix[:][0] for u_i in u])
            if isinstance(u, (list, np.ndarray)):
                if len(u) == 0:
                    # No inputs
                    u_i = []
                elif np.isscalar(u[0]):
                    # Input is 1-d array (i.e., 1 input)
                    # Note: 1 is added to account for current time (current input used to predict output at time i)
                    u_i = [[[u[i+j]] for j in range(1, sequence_length+1)] for i in range(len(u)-sequence_length-1)]
                elif isinstance(u[0], (list, np.ndarray)):
                    # Input is d-d array
                    # Note: 1 is added to account for current time (current input used to predict output at time i)
                    n_inputs = len(u[0])
                    u_i = [[[u[i+j][k] for k in range(n_inputs)] for j in range(1,sequence_length+1)] for i in range(len(u)-sequence_length-1)]
                else:
                    raise Exception(f"Unsupported input type: {type(u)} for internal element (data[0][i]")  
            else:
                raise Exception(f"Unsupported data type: {type(u)}. input u must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")

            # Process Output
            if isinstance(z, SimResult):
                if len(z[0].keys()) == 0:
                    z = []
                else:
                    z = np.array([z_i.matrix[:][0] for z_i in z])
            if isinstance(z, (list, np.ndarray)):
                if len(z) == 0:
                    # No inputs
                    z_i = []
                elif np.isscalar(z[0]):
                    # Output is 1-d array (i.e., 1 output)
                    z_i = [[[z[i+j]] for j in range(sequence_length)] for i in range(len(z)-sequence_length-1)]
                elif isinstance(z[0], (list, np.ndarray)):
                    # Input is d-d array
                    n_outputs = len(z[0])
                    z_i = [[[z[i+j][k] for k in range(n_outputs)] for j in range(sequence_length)] for i in range(len(z)-sequence_length-1)]
                else:
                    raise Exception(f"Unsupported input type: {type(z)} for internal element (data[0][i]")  
                # Also add to input (past outputs are part of input)
                if len(u_i) == 0:
                    u_i = z_i.copy()
                else:
                    for i in range(len(z_i)):
                        for j in range(sequence_length):
                            u_i[i][j].extend(z_i[i][j])
            else:
                raise Exception(f"Unsupported data type: {type(u)}. input u must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")
            
            u_all.extend(u_i)
            z_all.extend(z_i)
        
        u_all = np.array(u_all)
        z_all = np.array(z_all)
        return (u_all, z_all)

    @staticmethod
    def from_data(data, **kwargs):
        """
        Generate a LSTMStateTransitionModel from data

        Args:
            data (List[Tuple[Array, Array]]): list of runs to use for training. Each element is a tuple (input, output) for a single run.

        Keyword Args:
            sequence_length (int): Length of the input sequence
            inputs (List[str]): List of input keys
            outputs (List[str]): List of outputs keys
            validation_percentage (float): Percentage of data to use for validation, between 0-1
            epochs (int): Number of epochs to use in training

        Returns:
            LSTMStateTransitionModel: Generated Model
        """
        params = { # default_params
            'sequence_length': 128,
            'validation_split': 0.25,
            'epochs': 2
        }
        # TODO(CT): Add shuffling
        # TODO(CT): Add layers
        params.update(LSTMStateTransitionModel.default_params)
        params.update(kwargs)

        # Input Validation
        if params['sequence_length'] <= 0:
            raise Exception(f"sequence_length must be greater than 0, got {params['sequence_length']}")
        if params['validation_split'] < 0 or params['validation_split'] > 1:
            raise Exception(f"validation_split must be between 0 and 1, got {params['validation_split']}")
        if params['epochs'] < 1:
            raise Exception(f"epochs must be greater than 0, got {params['epochs']}")
        if isinstance(data, tuple) and len(data) == 2:
            # Just one dataset, turn into array so loop works properly
            data = [data]

        # Prepare datasets
        (u_all, z_all) = LSTMStateTransitionModel._pre_process_data(data, **params)

        # Build model
        callbacks = [
            keras.callbacks.ModelCheckpoint("jena_sense.keras", save_best_only=True)
        ]

        inputs = keras.Input(shape=u_all.shape[1:])
        x = layers.LSTM(16)(inputs)
        x = layers.Dense(z_all.shape[1] if len(z_all) == 2 else 1)(x)
        model = keras.Model(inputs, x)
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        model.fit(u_all, z_all, epochs=params['epochs'], callbacks = callbacks, validation_split = params['validation_split'])

        return LSTMStateTransitionModel(keras.models.load_model("jena_sense.keras"), **params)
        
    def simulate_to_threshold(self, future_loading_eqn, first_output = None, threshold_keys = None, **kwargs):
        t = kwargs.get('t0', 0)
        dt = kwargs.get('dt', 0)
        x = kwargs.get('x', self.initialize(future_loading_eqn(t), first_output))

        # configuring next_time function to define prediction time step, default is constant dt
        if callable(dt):
            dt_mode = 'function'
        elif isinstance(dt, tuple):
            dt_mode = dt[0]
            dt = dt[1]               
        elif isinstance(dt, str):
            dt_mode = dt
            if dt_mode == 'constant':
                dt = 1.0  # Default
            else:
                dt = np.inf
        else:
            dt_mode = 'constant'

        if dt_mode in ('constant', 'auto'):
            def next_time(t, x):
                return dt
        elif dt_mode != 'function':
            raise Exception(f"'dt' mode {dt_mode} not supported. Must be 'constant', 'auto', or a function")

        # Simulate until passing minimum number of steps
        while x.matrix[0,0] is None:
            dt = next_time(t, x)
            t = t + dt/2
            # Use state at midpoint of step to best represent the load during the duration of the step
            u = future_loading_eqn(t, x)
            t = t + dt/2
            x = self.next_state(x, u, dt)

        # Now do actual simulate_to_threshold
        kwargs['t0'] = t
        x.matrix = np.array(x.matrix, dtype=np.float)
        kwargs['x'] = x
        if 'horizon' in kwargs:
            kwargs['horizon'] = kwargs['horizon'] - t
        return super().simulate_to_threshold(future_loading_eqn, first_output, threshold_keys, **kwargs)
        
    # TODO(CT): From Model
