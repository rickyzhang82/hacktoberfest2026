"""Typed errors so the UI can show actionable messages instead of tracebacks."""


class RoboUIError(Exception):
    """Base class for all expected RoboUI failures."""


class ModelFileMissingError(RoboUIError):
    """A configured GGUF/mmproj file does not exist on disk."""


class OutOfMemoryError(RoboUIError):
    """The model could not be loaded or run within available (V)RAM."""


class BackendUnavailableError(RoboUIError):
    """Optional dependency or configuration for a backend is missing."""


class NetworkBlockedError(RoboUIError, OSError):
    """Raised when code inside the network sandbox attempts a connection."""


class CodeExtractionError(RoboUIError):
    """The model output did not contain usable HTML."""
