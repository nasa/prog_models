# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections import UserDict
from copy import deepcopy
from numbers import Number
import types
from typing import Callable

from .noise_functions import measurement_noise_functions, process_noise_functions
from ..exceptions import ProgModelTypeError

from typing import TYPE_CHECKING
if TYPE_CHECKING: # Fix circular import issue in PrognosticsModelParameters init
    from prog_models.prognostics_model import PrognosticsModel

class PrognosticsModelParameters(UserDict):
    """
    Prognostics Model Parameters - this class replaces a standard dictionary.
    It includes the extra logic to process the different supported manners of defining noise.

    Args:
        model: PrognosticsModel for which the params correspond
        dict_in: Initial parameters
        callbacks: Any callbacks for derived parameters f(parameters) : updates (dict)
    """
    def __init__(self, model : "PrognosticsModel", dict_in : dict = {}, callbacks : dict = {}):
        super().__init__()
        self.__m = model
        self.callbacks = {}
        # Note: Callbacks are set to empty to prevent calling callbacks with a partial or empty dict on line 32. 
        for (key, value) in dict_in.items():
            # Deepcopy is needed here to force copying when value is an object (e.g., dict)
            self[key] = deepcopy(value)

        # Add and run callbacks
        # Has to be done here so the base parameters are all set 
        self.callbacks = callbacks
        for key in callbacks:
            if key in self:
                for callback in callbacks[key]:
                    changes = callback(self)
                    self.update(changes)

    def __setitem__(self, key : str, value : float) -> None:
        """Set model configuration, overrides dict.__setitem__()

        Args:
            key (string): configuration key to set
            value: value to set that configuration value to

        Raises:
            ProgModelTypeError: Improper configuration for a model
        """
        super().__setitem__(key, value)

        if key in self.callbacks:
            for callback in self.callbacks[key]:
                changes = callback(self)
                self.update(changes) # Merge in changes
        
        if key == 'process_noise' or key == 'process_noise_dist':
            if callable(self['process_noise']):  # Provided a function
                self.__m.apply_process_noise = types.MethodType(self['process_noise'], self.__m)
            else:  # Not a function
                # Process noise is single number - convert to dict
                if isinstance(self['process_noise'], Number):
                    self['process_noise'] = {key: self['process_noise'] for key in self.__m.states}
                
                # Process distribution type
                if 'process_noise_dist' in self and self['process_noise_dist'].lower() not in process_noise_functions:
                    raise ProgModelTypeError("Unsupported process noise distribution")
                
                if all(value == 0 for value in self['process_noise'].values()):
                    # No noise, use none function
                    fcn = process_noise_functions['none']
                    self.__m.apply_process_noise = types.MethodType(fcn, self.__m)
                elif 'process_noise_dist' in self:
                    fcn = process_noise_functions[self['process_noise_dist'].lower()]
                    self.__m.apply_process_noise = types.MethodType(fcn, self.__m)
                else:
                    # Default to gaussian
                    fcn = process_noise_functions['gaussian']
                    self.__m.apply_process_noise = types.MethodType(fcn, self.__m)
                
                # Make sure every key is present (single value already handled above)
                if not all([key in self['process_noise'] for key in self.__m.states]):
                    raise ProgModelTypeError("Process noise must have every key in model.states")

        elif key == 'measurement_noise' or key == 'measurement_noise_dist':
            if callable(self['measurement_noise']):
                self.__m.apply_measurement_noise = types.MethodType(self['measurement_noise'], self.__m)
            else:
                # Process noise is single number - convert to dict
                if isinstance(self['measurement_noise'], Number):
                    self['measurement_noise'] = {key: self['measurement_noise'] for key in self.__m.outputs}
                
                # Process distribution type
                if 'measurement_noise_dist' in self and self['measurement_noise_dist'].lower() not in measurement_noise_functions:
                    raise ProgModelTypeError("Unsupported measurement noise distribution")

                if all(value == 0 for value in self['measurement_noise'].values()):
                    # No noise, use none function
                    fcn = measurement_noise_functions['none']
                    self.__m.apply_measurement_noise = types.MethodType(fcn, self.__m)
                elif 'measurement_noise_dist' in self:
                    fcn = measurement_noise_functions[self['measurement_noise_dist'].lower()]
                    self.__m.apply_measurement_noise = types.MethodType(fcn, self.__m)
                else:
                    # Default to gaussian
                    fcn = measurement_noise_functions['gaussian']
                    self.__m.apply_measurement_noise = types.MethodType(fcn, self.__m)
                    
                # Make sure every key is present (single value already handled above)
                if not all([key in self['measurement_noise'] for key in self.__m.outputs]):
                    raise ProgModelTypeError("Measurement noise must have ever key in model.outputs")

    def register_derived_callback(self, key : str, callback : Callable) -> None:
        """Register a new callback for derived parameters

        Args:
            key (string): key for which the callback is triggered
            callback (function): callback function f(parameters) -> updates (dict)
        """
        if key in self.callbacks:
            self.callbacks[key].append(callback)
        else:
            self.callbacks[key] = [callback]

        # Run new callback
        if key in self:
            updates = callback(self[key])
            self.update(updates)
