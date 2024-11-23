class BaseContrastSenseException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseContrastSenseException):
    """Raised when the choice of backbone Convnet is invalid."""


class InvalidDatasetSelection(BaseContrastSenseException):
    """Raised when the choice of datasets is invalid."""
