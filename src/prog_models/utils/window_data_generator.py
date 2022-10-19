# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from itertools import chain
import numpy as np
from tensorflow import keras

from prog_models.sim_result import SimResult


class WindowDataGenerator(keras.utils.Sequence):
    """

    Args:
        inputs (List[ndarray or SimulationResult]): Data to be processed. Each element is of format, ndarray or SimulationResult
        outputs (List[ndarray or SimulationResult]): Data to be processed. Each element is of format, ndarray or SimulationResult
        event_states (List[ndarray or SimulationResult]): Data to be processed. Each element is of format, ndarray or SimulationResult
        t_met (List[ndarray or SimulationResult]): Data to be processed. Each element is of format, ndarray or SimulationResult
        window (int): Length of a single sequence
    """
    def __init__(self, inputs, outputs, event_states=None, t_met = None, window=10, _validation = False, **kwargs):
        if isinstance(inputs, WindowDataGenerator) and _validation:
            # Special case, creating validation data generator 
            # Only used internally
            self.window = inputs.window
            self.n_inputs = inputs.n_inputs
            self.n_outputs = inputs.n_outputs
            self.n_event_states = inputs.n_event_states
            self.n_thresholds = inputs.n_thresholds

            n_elements = round(len(inputs) * outputs)

            self.u_all = []
            self.z_all = []
            self.es_all = []
            self.t_all = []
            while n_elements > 0:
                n_batches = len(inputs.z_all[-1])
                if n_batches <= n_elements:
                    self.u_all.append(inputs.u_all[-1])
                    del inputs.u_all[-1]
                    self.z_all.append(inputs.z_all[-1])
                    del inputs.z_all[-1]
                    self.es_all.append(inputs.es_all[-1])
                    del inputs.es_all[-1]
                    self.t_all.append(inputs.t_all[-1])
                    del inputs.t_all[-1]
                    n_elements -= n_batches
                    continue

                self.u_all.append(inputs.u_all[-1][len(inputs.z_all[-1])-n_elements:])
                inputs.u_all[-1] = np.delete(inputs.u_all[-1], list(range(len(inputs.u_all[-1])-n_elements, len(inputs.u_all[-1]))), axis=0)

                self.z_all.append(inputs.z_all[-1][-n_elements:])
                inputs.z_all[-1] = np.delete(inputs.z_all[-1], list(range(len(inputs.z_all[-1])-n_elements, len(inputs.z_all[-1]))), axis=0)
                
                if self.n_event_states > 0:
                    self.es_all.append(inputs.es_all[-1][-n_elements:])
                    inputs.es_all[-1] = np.delete(inputs.es_all[-1], list(range(len(inputs.es_all[-1])-n_elements, len(inputs.es_all[-1]))), axis=0)
                else:
                    self.es_all.append([])

                if self.n_thresholds > 0:
                    self.t_all.append(inputs.t_all[-1][-n_elements:])
                    inputs.t_all[-1] = np.delete(inputs.t_all[-1], list(range(len(inputs.t_all[-1])-n_elements, len(inputs.t_all[-1]))), axis=0)
                else:
                    self.t_all.append([])

                n_elements -= n_batches
                return
        
        if isinstance(inputs, SimResult):
            inputs = [inputs]
        if len(inputs) == 0:
            raise ValueError("No data provided. inputs must be in format [run1_inputs, ...] and have at least one element")
        if isinstance(outputs, SimResult):
            outputs = [outputs]
        if len(inputs) != len(outputs):
            raise ValueError("Inputs must be same length as outputs")            
        if event_states is not None and isinstance(event_states, SimResult):
            event_states = [event_states]
        if event_states is not None and len(inputs) != len(event_states):
            raise ValueError("Event States must be same length as inputs")
        if t_met is not None and isinstance(t_met, SimResult):
            t_met = [t_met]
        if t_met is not None and len(inputs) != len(t_met):
            raise ValueError("Thresholds met must be same length as inputs")
        if event_states is not None and len(inputs) != len(event_states):
            raise ValueError("Inputs must be same length as event_states")

        self.window = window
        
        self.u_all = []
        self.z_all = []
        self.es_all = []
        self.t_all = []
        n_inputs = 0
        n_outputs = 0
        n_event_states = 0
        n_thresholds = 0

        for i in range(len(inputs)):
            # Each item (u, z) is a 1-d array, a 2-d array, or a SimResult
            # Turn u into n_inputs + n_outputs by n_steps. e.g., 
            # [[u_1, z_0], [u_2, z_1]]

            # Process Output
            z = outputs[i]
            if isinstance(z, SimResult):
                if len(z[0].keys()) == 0:
                    # No outputs
                    z = []
                else:
                    z = np.array([z_i.matrix[:,0] for z_i in z])

            if isinstance(z, (list, np.ndarray)):
                if len(z) == 0:
                    # No outputs
                    z_i = []
                elif np.isscalar(z[0]):
                    # Output is 1-d array (i.e., 1 output)
                    n_outputs = 1
                    z_i = np.reshape(z, (len(z), 1))
                elif isinstance(z[0], (list, np.ndarray)):
                    # Input is 2-d array
                    n_outputs = len(z[0])
                    z_i = z
                else:
                    raise TypeError(f"Unsupported input type: {type(z)} for internal element (output[i])")  
            else:
                raise TypeError(f"Unsupported data type: {type(z)}. output z must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")

            u = inputs[i]
            if isinstance(u, SimResult):
                if len(u[0].keys()) == 0:
                    # No inputs
                    u = []
                else:
                    u = np.array([u_i.matrix[:,0] for u_i in u])

                    if len(u) < window:
                        raise TypeError(f"Not enough data for window size {window}. Only {len(u)} elements present.")

            if isinstance(u, (list, np.ndarray)):
                if len(z) != len(u) and len(u) != 0 and len(z) != 0:
                    # Checked here to avoid SimResults from accidentially triggering this check
                    raise IndexError(f"Number of outputs ({len(z)}) does not match number of inputs ({len(u)})")
                if len(u) == 0:
                    # No inputs
                    u_i = z_i[:-1]
                elif np.isscalar(u[0]):
                    # Input is 1-d array (i.e., 1 input)
                    n_inputs = 1
                    u_i = np.reshape(u, (len(u), 1))[1:]
                    if len(z_i) != 0:
                        u_i = np.hstack((u_i, z_i[:-1]))
                elif isinstance(u[0], (list, np.ndarray)):
                    # Outputs is 2-d array
                    n_inputs = len(u[0])
                    u_i = u[1:]
                    if len(z_i) != 0:
                        u_i = np.hstack((u_i, z_i[:-1]))
                else:
                    raise TypeError(f"Unsupported input type: {type(u)} for internal element (data[0][i]")  
            else:
                raise TypeError(f"Unsupported data type: {type(u)}. input u must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")

            if event_states is not None:
                es = event_states[i]
                if isinstance(es, SimResult):
                    if len(es[0].keys()) == 0:
                        # No event_states
                        es = []
                    else:
                        es = np.array([[es_i[key] for key in es_i.keys()] for es_i in es])

                if isinstance(es, (list, np.ndarray)):
                    if len(es) != len(u) and len(u) != 0 and len(es) != 0:
                        # Checked here to avoid SimResults from accidentially triggering this check
                        raise IndexError(f"Number of event_states ({len(es)}) does not match number of inputs ({len(u)})")

                    if len(es) == 0:
                        # No event states
                        es_i = []
                    elif np.isscalar(es[0]):
                        # Event states is 1-d array (i.e., 1 output)
                        n_event_states = 1
                        es_i = [[es[i]] for i in range(window, len(es))]
                    elif isinstance(es[0], (list, np.ndarray)):
                        # event states is 2-d array
                        n_event_states = len(es[0])
                        es_i = [[es[i][k] for k in range(n_event_states)] for i in range(window, len(es))]
                    else:
                        raise TypeError(f"Unsupported input type: {type(es)} for internal element (es[i])")  

                else:
                    raise TypeError(f"Unsupported data type: {type(es)}. event state must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")
            else:
                es_i = []

            if t_met is not None:
                t = t_met[i]
                if isinstance(t, SimResult):
                    if len(t[0].keys()) == 0:
                        # No t_met
                        t = []
                    else:
                        t = np.array([[t_i[key] for key in t_i.keys()] for t_i in t])

                if isinstance(t, (list, np.ndarray)):
                    if len(t) != len(u) and len(u) != 0 and len(t) != 0:
                        # Checked here to avoid SimResults from accidentially triggering this check
                        raise IndexError(f"Number of t_met ({len(t)}) does not match number of inputs ({len(u)})")

                    if len(t) == 0:
                        # No t_met
                        t_i = []
                    elif np.isscalar(t[0]):
                        # t_met is 1-d array (i.e., 1 output)
                        n_thresholds = 1
                        t_i = [[1, 0] if t[i] else [0, 1] for i in range(window, len(t))]
                    elif isinstance(t[0], (list, np.ndarray)):
                        # t_met is 2-d array
                        n_thresholds = len(t[0])
                        # Convert for classification
                        # True = 1, 0; False = 0, 1
                        t_i = [list(chain.from_iterable((1, 0) if t[i][k] else (0, 1) for k in range(n_thresholds))) for i in range(window, len(t))]
                    else:
                        raise TypeError(f"Unsupported input type: {type(t[0])} for internal element (t[i])")  

                else:
                    raise TypeError(f"Unsupported data type: {type(t)}. t_met must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")
            else:
                t_i = []

            self.u_all.append(np.array(u_i, dtype=np.float64))
            self.z_all.append(np.array(z_i[window:], dtype=np.float64))
            self.es_all.append(np.array(es_i, dtype=np.float64))
            self.t_all.append(np.array(t_i, dtype=np.float64))  
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_event_states = n_event_states
        self.n_thresholds = n_thresholds

    def split_validation(self, validation_split = 0.2):
        """
        Splits data generator into training and validation data generators

        Args:
            validation_split (float, optional): Percentage of training data to split out. Defaults to 0.2.

        Returns:
            tuple[WindowDataGenerator, WindowDataGenerator]: training and validation data generators
        """
        return (self, WindowDataGenerator(self, validation_split, _validation=True))

    def normalize_outputs(self, z_mean, z_std):
        """
        Normalize the outputs

        Args:
            z_mean (np.ndarray): Mean outputs
            z_std (np.ndarray): Standard deviation of outputs
        """
        self.z_all = [(z-z_mean)/z_std for z in self.z_all]

    def calculate_normalization(self):
        """
        Calculate the mean and standard deviations for normalization

        Returns:
            tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray): (mean inputs, standard deviation of inputs, mean outputs, standard deviation of outputs)

        Example:
            (u_mean, u_std, z_mean, z_std) = gen.calculate_normalization()
        """
        u_in = [u_i[:self.n_inputs] for u in self.u_all for u_i in u]
        u_mean = np.mean(u_in, axis=0)
        u_std = np.std(u_in, axis=0)

        # If there's no variation- dont normalize 
        u_std[u_std == 0] = 1

        z_in = np.vstack(self.z_all)
        z_mean = np.mean(z_in, axis=0)
        z_std = np.std(z_in, axis=0)
        # If there's no variation- dont normalize 
        z_std[z_std == 0] = 1

        # Add output (since z_t-1 is last input)
        u_mean = np.hstack((u_mean, z_mean))
        u_std = np.hstack((u_std, z_std))

        return (u_mean, u_std, z_mean, z_std)

    def __len__(self):
        'Calculate the number of batches per epoch'
        return sum((len(z) for z in self.z_all))

    def __getitem__(self, index):
        'Generate one batch of data'
        if index >= len(self):
            raise IndexError('Index out of range')

        for i in range(len(self.u_all)):
            if index >= len(self.z_all[i]):
                index -= len(self.z_all[i])
                continue
            u = self.u_all[i][index:index+self.window]
            z = [self.z_all[i][index]]
            if len(self.es_all[i]) != 0:
                z.append(self.es_all[i][index])
            if len(self.t_all[i]) != 0:
                z.append(self.t_all[i][index])
            if len(z) > 1:
                z = tuple(z)
            else:
                z = z[0]
            break

        return ([u], z)