# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

__all__ = ["deriv_prog_model", "prognostics_model", "models", "ProgModelException", "ProgModelInputException", "ProgModelTypeError"]

class ProgModelException(Exception):
    pass

class ProgModelInputException(ProgModelException):
    pass

class ProgModelTypeError(ProgModelException, TypeError):
    pass