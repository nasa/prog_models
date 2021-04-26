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