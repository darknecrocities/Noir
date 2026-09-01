"""Custom exception hierarchy for Project NOIR."""


class NoirException(Exception):
    """Base exception for all Project NOIR exceptions."""
    pass


class ConfigurationError(NoirException):
    """Raised when configuration is invalid, missing, or cannot be parsed."""
    pass


class EngineStateError(NoirException):
    """Raised when an invalid state transition is requested in the engine."""
    pass


class TrainingError(NoirException):
    """Raised when a numerical, optimization, or execution error occurs during training."""
    pass


class CheckpointError(NoirException):
    """Raised when checkpoint saving, loading, or validation fails."""
    pass


class StorageError(NoirException):
    """Raised when database or file storage operations fail."""
    pass


class RecoveryError(NoirException):
    """Raised when recovering from a crash or previous run fails."""
    pass


class MCPError(NoirException):
    """Raised when MCP server or tool execution fails."""
    pass


class StrategyError(NoirException):
    """Raised when LLM strategist interaction or hypothesis generation fails."""
    pass
