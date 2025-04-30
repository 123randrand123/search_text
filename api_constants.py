from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
)

from api_utils import is_rate_limit_error
from utils import logger


class RateLimitHeaderNotFoundError(Exception):
    """Raised when the Retry-After header is missing from the rate limit response."""

    pass


def get_retry_after(exception):
    """
    Extract the wait time from the exception's raw_response headers.
    First, try "Retry-After", then fallback to "ratelimitbysize-reset".
    Returns the header value as a float if present, or None otherwise.
    """
    try:
        headers = exception.raw_response.headers

        retry_after = headers.get("Retry-After") or headers.get("ratelimitbysize-reset")
        if retry_after is not None:
            return float(retry_after)
    except Exception as e:
        logger.error("Error extracting retry wait time: %s", e)
    return None


def custom_wait(retry_state) -> float:
    """
    Custom wait function that uses the Retry-After header from the exception.
    If the header is not found, it raises an error.
    """
    exception = retry_state.outcome.exception()
    retry_after = get_retry_after(exception)
    if retry_after is None:
        message = "Retry-After header not found in rate limit error response."
        logger.error(message)
        raise RateLimitHeaderNotFoundError(message)
    return retry_after


RETRY_DECORATOR = retry(
    retry=retry_if_exception(is_rate_limit_error),
    wait=custom_wait,
    stop=stop_after_attempt(5),
    reraise=True,
)
