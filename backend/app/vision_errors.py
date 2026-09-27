"""Provider-neutral errors for visual product analysis. All are safe to show and retryable."""


class VisionNotConfiguredError(RuntimeError):
    pass


class VisionUnavailableError(RuntimeError):
    """The vision backend is unreachable, the model is missing, or it returned an error."""


class VisionTimeoutError(TimeoutError):
    pass


class VisionMalformedResponseError(RuntimeError):
    """Output was not valid JSON, did not match the schema, or was incomplete."""
