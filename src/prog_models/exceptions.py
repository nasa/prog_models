# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.


class ProgModelException(Exception):
    """
    Base Prognostics Model Exception
    """


class ProgModelInputException(ProgModelException):
    """
    Prognostics Input Exception - indicates the method input parameters were incorrect
    """


class ProgModelTypeError(ProgModelException, TypeError):
    """
    Prognostics Type Error - indicates the model could not be constructed
    """


class ProgModelStateLimitWarning(Warning):
    """
    Prognostics State Limit Warning - indicates the model state was outside the limits, and was adjusted
    """
