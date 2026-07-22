class QEMError(Exception):
    """Base exception for all quantum error mitigation errors."""
    pass

class DimensionError(QEMError, ValueError):
    """Raised when dimensions of counts and response matrices do not match."""
    pass

class ShapeError(QEMError, IndexError):
    """Raised when the shape of matrices do not match."""
    pass

class InvalidResponseMatrixError(QEMError, ValueError):
    """Raised when the response matrix violates physical or probabilistic constraints."""
    pass