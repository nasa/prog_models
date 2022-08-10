# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from .data_model import DataModel
from .lstm_model import LSTMStateTransitionModel
from .dmd import SurrogateDMDModel

SURROAGATE_METHOD_LOOKUP = {
    'dmd': SurrogateDMDModel,
    'lstm': LSTMStateTransitionModel
}


__all__ = ['DataModel', 'LSTMStateTransitionModel']
