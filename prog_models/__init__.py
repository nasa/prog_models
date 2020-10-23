__all__ = ["model", "prognostics_model", "models"]

class ProgModelException(Exception):
    pass

class ProgModelInputException(ProgModelException):
    pass