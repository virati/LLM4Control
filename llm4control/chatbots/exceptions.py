class GPT3APIError(Exception):
    """Exception raised for errors in the GPT-3.5 API."""

    def __init__(
        self,
        message: str = "An error occurred with the GPT-3.5 API",
        *args: object,
        **kwargs: object
    ) -> None:
        super().__init__(message, *args, **kwargs)


class MistralAPIError(Exception):
    """Exception raised for errors in the Mistral API."""

    def __init__(
        self,
        message: str = "An error occurred with the Mistral API",
        *args: object,
        **kwargs: object
    ) -> None:
        super().__init__(message, *args, **kwargs)


class GeminiAPIError(Exception):
    """Exception raised for errors in the Gemini API."""

    def __init__(
        self,
        message: str = "An error occurred with the Gemini API",
        *args: object,
        **kwargs: object
    ) -> None:
        super().__init__(message, *args, **kwargs)


class GPT4APIError(Exception):
    """Exception raised for errors in the GPT-4 API."""

    def __init__(
        self, message: str = "An error occurred with the GPT-4 API", *args: object, **kwargs: object
    ) -> None:
        super().__init__(message, *args, **kwargs)
