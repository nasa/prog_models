# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

__all__ = ["prognostics_model", "models", "ProgModelException", "ProgModelInputException", "ProgModelTypeError"]

class ProgModelException(Exception):
    """
    Base Prognostics Model Exception
    """
    pass

class ProgModelInputException(ProgModelException):
    """
    Prognostics Input Exception - indicates the method input parameters were incorrect
    """
    pass

class ProgModelTypeError(ProgModelException, TypeError):
    """
    Prognostics Type Error - indicates the model could not be constructed
    """
    pass