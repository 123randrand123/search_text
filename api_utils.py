def is_rate_limit_error(exc: BaseException) -> bool:
    """Checks if an exception is a rate limit error (429)."""
    is_429 = ("429" in str(exc)) or (
        hasattr(exc, "status_code") and getattr(exc, "status_code") == 429
    )
    is_mistral_limit_msg = "Rate limit exceeded" in str(
        exc
    ) or "Too Many Requests" in str(exc)
    return is_429 or is_mistral_limit_msg
