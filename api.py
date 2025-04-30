from mistralai import Mistral

from api_constants import RETRY_DECORATOR
from models import SuggestionResponse, ValidationResponse
from utils import logger


@RETRY_DECORATOR
async def call_suggestion_api(
    client: Mistral,
    model: str,
    messages: list[dict],
    blocked_terms_str: str,
) -> SuggestionResponse:
    """Calls Mistral API for structured suggestion."""
    logger.debug(f"Calling Suggestion API with {len(messages)} messages.")
    system_message = {
        "role": "system",
        "content": (
            "You are a JSON-only suggestion engine. Replace the blocked term in"
            " the excerpt, preserving original meaning but avoiding EXACT"
            f" blocked terms.\nBlocked: {blocked_terms_str}\nGoal: Find clever,"
            " alternative phrasing for the blocked term, not topic avoidance."
            " Analyze context & history. Output ONLY JSON matching schema."
        ),
    }
    full_messages = [system_message] + messages
    try:
        response = await client.chat.parse_async(
            model=model,
            messages=full_messages,
            response_format=SuggestionResponse,
            max_tokens=100,
            temperature=0.6,
        )
        parsed_suggestion = response.choices[0].message.parsed
        if not isinstance(parsed_suggestion, SuggestionResponse):
            logger.error(
                "Suggestion parse_async result type mismatch: %s",
                type(parsed_suggestion),
            )
            raise ValueError("API response failed SuggestionResponse schema")
        logger.debug("Suggestion API parsed response: %s", parsed_suggestion)
        return parsed_suggestion
    except Exception as e:
        logger.exception("Error in call_suggestion_api: %s", e)
        raise


@RETRY_DECORATOR
async def call_validation_api(
    client: Mistral,
    model: str,
    original_excerpt: str,
    candidate: str,
    full_text_context: str,
    blocked_terms_str: str,
) -> ValidationResponse:
    """Calls Mistral API for structured validation."""
    logger.debug(
        "Calling Validation API for candidate: '%s' replacing '%s'",
        candidate,
        original_excerpt,
    )
    system_message = {
        "role": "system",
        "content": (
            "You are a JSON-only validation engine. Evaluate if 'candidate' is"
            " acceptable for 'original_excerpt' in 'full_text_context'.\nCriteria:\n1."
            " **No Blocked Terms:** Candidate must NOT contain these exact terms:"
            f" {blocked_terms_str}\n2. **Meaning/Circumvention:** Conveys original"
            " meaning without blocked term? (Valid circumvention, not evasion).\n3."
            " **Context:** Suitable and respectful?\n4. **No New Harm:** Doesn't"
            " introduce new offensive content?\nSet `should_block: true` if ANY check"
            " fails (esp. #1 or #4). Provide `reason`. Output ONLY JSON."
        ),
    }
    user_message = {
        "role": "user",
        "content": (
            f"Full Text Context:\n---\n{full_text_context}\n---\nOriginal"
            f' Excerpt: "{original_excerpt}"\nCandidate Suggestion:'
            f' "{candidate}"\n\nEvaluate based on system prompt criteria.'
            " Output JSON."
        ),
    }
    messages = [system_message, user_message]
    try:
        response = await client.chat.parse_async(
            model=model,
            messages=messages,
            response_format=ValidationResponse,
            max_tokens=150,
            temperature=0.1,
        )
        parsed_validation = response.choices[0].message.parsed
        if not isinstance(parsed_validation, ValidationResponse):
            logger.error(
                "Validation parse_async result type mismatch: %s",
                type(parsed_validation),
            )
            raise ValueError("API response failed ValidationResponse schema")
        logger.debug("Validation API parsed response: %s", parsed_validation)
        return parsed_validation
    except Exception as e:
        logger.exception("Error in call_validation_api: %s", e)
        raise
