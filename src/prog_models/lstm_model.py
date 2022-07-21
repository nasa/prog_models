# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections.abc import Iterable
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
        examples.lstm_model

    Examples:
        :
            >>> from prog_models import LSTMStateTransitionModel
            >>> # Generate model from data
            >>> m = LSTMStateTransitionModel.from_data(inputs, outputs)
    """
    default_params = {
    }

    def __init__(self, model, **kwargs):

        # Setup inputs, outputs, states 
        self.outputs = kwargs.get('outputs', [f'z{i}' for i in range(model.output.shape[1])])

        input_shape = model.input.shape
        input_keys = kwargs.get('inputs', [f'u{i}' for i in range(input_shape[2]-len(self.outputs))])
        self.inputs = input_keys.copy()
        # Outputs from the last step are part of input
        self.inputs.extend([f'{z_key}_t-1' for z_key in self.outputs])

        # States are in format [u_t-n+1, z_t-n, ..., u_t, z_t-1]
        self.states = []
        for j in range(input_shape[1]-1, -1, -1):
            self.states.extend([f'{input_i}_t-{j}' for input_i in input_keys])
            self.states.extend([f'{output_i}_t-{j+1}' for output_i in self.outputs])

        kwargs['sequence_length'] = input_shape[1]

        super().__init__(**kwargs)

        # Save Model
        self.model = model

    def __eq__(self, other):
        # Needed bacause we add .model, which is not present in the parent class
        if not isinstance(other, LSTMStateTransitionModel):
            return False
        return super().__eq__(self, other) and self.model == other.model

    def initialize(self, u=None, z=None):
        """
        Initialize the model with the given inputs and outputs. For LSTMStateTransitionModel, the initial state is set to None. As data is added (i.e., next_state is called), the initial state is updated.

        This is one of the rare models that cannot be initialized right away.

        Args:
            u (InputContainer, optional): first input
            z (OutputContainer, optional): first output

        Returns:
            StateContainer: First State
        """
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

        # Enough data has been received to calculate output
        # Format input into np array with shape (1, sequence_length, num_inputs)
        m_input = x.matrix[:self.parameters['sequence_length']*len(self.inputs)].reshape(1, self.parameters['sequence_length'], len(self.inputs))

        # Pass into model to calculate output
        m_output = self.model(m_input)

        return self.OutputContainer(m_output.numpy().T)

    @staticmethod
    def _pre_process_data(data, sequence_length, **kwargs):
        """
        Pre-process data for the LSTMStateTransitionModel. This is run inside from_data to convert the data into the desired format 

        Args:
            data (List[Tuple] or Tuple (where Tuple is equivilant to [Tuple]))): Data to be processed. each element is of format (input, output), where input and output can be ndarray or SimulationResult
            sequence_length (int): Length of a single sequence

        Returns:
            Tuple[ndarray, ndarray]: pre-processed data (input, output). Where input is of size (num_sequences, sequence_length, num_inputs) and output is of size (num_sequences, num_outputs)
        """
        # Data is a List[Tuple] or Tuple (where Tuple is equivilant to [Tuple]))
        # Tuple is (input, output)

        u_all = []
        z_all = []
        for (u, z) in data:
            # Each item (u, z) is a 1-d array, a 2-d array, or a SimResult

            # Process Input
            if isinstance(u, SimResult):
                if len(u[0].keys()) == 0:
                    # No inputs
                    u = []
                else:
                    u = np.array([u_i.matrix[:,0] for u_i in u])

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
                    raise TypeError(f"Unsupported input type: {type(u)} for internal element (data[0][i]")  
            else:
                raise TypeError(f"Unsupported data type: {type(u)}. input u must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")

            # Process Output
            if isinstance(z, SimResult):
                if len(z[0].keys()) == 0:
                    # No outputs
                    z = []
                else:
                    z = np.array([z_i.matrix[:,0] for z_i in z])

            if isinstance(z, (list, np.ndarray)):
                if len(z) != len(u) and len(u) != 0 and len(z) != 0:
                    # Checked here to avoid SimResults from accidentially triggering this check
                    raise IndexError(f"Number of outputs ({len(z)}) does not match number of inputs ({len(u)})")

                if len(z) == 0:
                    # No outputs
                    z_i = []
                elif np.isscalar(z[0]):
                    # Output is 1-d array (i.e., 1 output)
                    z_i = [[z[i]] for i in range(sequence_length+1, len(z))]
                elif isinstance(z[0], (list, np.ndarray)):
                    # Input is d-d array
                    n_outputs = len(z[0])
                    z_i = [[z[i][k] for k in range(n_outputs)] for i in range(sequence_length+1, len(z))]
                else:
                    raise TypeError(f"Unsupported input type: {type(z)} for internal element (data[0][i]")  

                # Also add to input (past outputs are part of input)
                if len(u_i) == 0:
                    u_i = [[z_ii for _ in range(sequence_length)] for z_ii in z_i]
                else:
                    for i in range(len(z_i)):
                        for j in range(sequence_length):
                            u_i[i][j].extend(z_i[i])
            else:
                raise TypeError(f"Unsupported data type: {type(u)}. input u must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")
            
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
            data (List[Tuple[Array, Array]]): list of runs to use for training. Each element is a tuple (input, output) for a single run. Input and Output are of size (n_times, n_inputs/outputs)

        Keyword Args:
            sequence_length (int): Length of the input sequence
            inputs (List[str]): List of input keys
            outputs (List[str]): List of outputs keys
            validation_percentage (float): Percentage of data to use for validation, between 0-1
            epochs (int): Number of epochs to use in training
            layers (int): Number of layers in the LSTM

        Returns:
            LSTMStateTransitionModel: Generated Model
        """
        params = { # default_params
            'sequence_length': 128,
            'validation_split': 0.25,
            'epochs': 2,
            'layers': 1
        }.copy()  # Copy is needed to avoid updating default

        params.update(LSTMStateTransitionModel.default_params)
        params.update(kwargs)

        # Input Validation
        if not np.isscalar(params['sequence_length']):
            raise TypeError(f"sequence_length must be an integer greater than 0, not {type(params['sequence_length'])}")
        if params['sequence_length'] <= 0:
            raise ValueError(f"sequence_length must be greater than 0, got {params['sequence_length']}")
        if not np.isscalar(params['layers']):
            raise TypeError(f"layers must be an integer greater than 0, not {type(params['layers'])}")
        if params['layers'] <= 0:
            raise ValueError(f"layers must be greater than 0, got {params['layers']}")
        if not np.isscalar(params['validation_split']):
            raise TypeError(f"validation_split must be an float between 0 and 1, not {type(params['validation_split'])}")
        if params['validation_split'] < 0 or params['validation_split'] > 1:
            raise ValueError(f"validation_split must be between 0 and 1, got {params['validation_split']}")
        if not np.isscalar(params['epochs']):
            raise TypeError(f"epochs must be an integer greater than 0, not {type(params['epochs'])}")
        if params['epochs'] < 1:
            raise ValueError(f"epochs must be greater than 0, got {params['epochs']}")
        if isinstance(data, tuple):
            # Just one dataset, turn into array so loop works properly
            data = [data]
        if not isinstance(data, Iterable):
            raise ValueError(f"data must be in format [(input, output), ...], got {type(data)}")
        if len(data) == 0:
            raise ValueError("No data provided. Data must be in format [(input, output), ...] and have at least one element")
        if not isinstance(data[0], tuple):
            raise ValueError(f"Each element of data must be a tuple, got {type(data[0])}")
        if len(data[0]) != 2:
            raise ValueError("Each element of data must be in format (input, output), where input and output are either np.array or SimulationResults and have at least one element")

        # Prepare datasets
        (u_all, z_all) = LSTMStateTransitionModel._pre_process_data(data, **params)

        # Build model
        callbacks = [
            keras.callbacks.ModelCheckpoint("jena_sense.keras", save_best_only=True)
        ]

        inputs = keras.Input(shape=u_all.shape[1:])
        x = inputs
        for i in range(params['layers']):
            if i == params['layers'] - 1:
                # Last layer
                x = layers.LSTM(16)(x)
            else:
                # Intermediate layer
                x = layers.LSTM(16, return_sequences=True)(x)
        x = layers.Dense(z_all.shape[1] if z_all.ndim == 2 else 1)(x)
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
            if 'horizon' in kwargs and t > kwargs['horizon']:
                raise Exception(f'Not enough timesteps to reach minimum number of steps for model simulation')
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
