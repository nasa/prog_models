# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

__all__ = ["deriv_prog_model", "model", "prognostics_model", "models"]

class ProgModelException(Exception):
    pass

class ProgModelInputException(ProgModelException):
    pass