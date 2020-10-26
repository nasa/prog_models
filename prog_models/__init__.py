__all__ = ["deriv_prog_model", "model", "prognostics_model", "models"]

class ProgModelException(Exception):
    pass

class ProgModelInputException(ProgModelException):
    pass