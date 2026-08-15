"""Project-specific exceptions with safe, non-participant-level messages."""


class NeuroSaarthiError(Exception):
    """Base exception for expected platform failures."""


class ConfigurationError(NeuroSaarthiError):
    """Raised when configuration is invalid or internally inconsistent."""


class DataGovernanceError(NeuroSaarthiError):
    """Raised when an operation violates the active data-governance policy."""


class DataValidationError(NeuroSaarthiError):
    """Raised when data fail a declared schema or scientific invariant."""
