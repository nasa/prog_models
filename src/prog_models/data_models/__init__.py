# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from .data_model import DataModel
from .dmd import DMDModel
from .nn_model import NNStateTransitionModel
from .fnn_model import FNNStateTransitionModel
from .lstm_model import LSTMStateTransitionModel

SURROAGATE_METHOD_LOOKUP = {
    'dmd': DMDModel.from_model,
    'lstm': LSTMStateTransitionModel.from_model
}
