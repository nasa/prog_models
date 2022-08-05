# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from abc import ABC, abstractclassmethod

from .. import PrognosticsModel


class DataModel(PrognosticsModel, ABC):
    """
    Abstract Base Class for all Data Models (e.g., LSTM, DMD). Defines the interface and all common tools. To create a new Data-Driven model, first subclass this, then define the abstract methods from this class and PrognosticsModel

    See Also:
        PrognosticsModel
    """

    @abstractclassmethod
    def from_data(cls, data: list, **kwargs) -> "DataModel":
        """
        Create a Data Model from data. This class is overwritten by specific data-driven classes (e.g., LSTM)

        Args:
            data (List[Tuple[Array, Array]]): list of runs to use for training. Each element is a tuple (input, output) for a single run. Input and Output are of size (n_times, n_inputs/outputs)

        Keyword Arguments:
            See specific data class for more information

        Returns:
            DataModel: Trained PrognosticsModel

        Example:
            |
                # Replace DataModel with specific classname below
                m = DataModel.from_data(data)
        """
        pass
