import asyncio
import json
import os
import re

from mistralai import Mistral

from api import call_suggestion_api, call_validation_api
from api_utils import is_rate_limit_error
from utils import logger


async def _run_replacement_loop(
    job_id: str,
    full_text: str,
    excerpt: str,
    blocked_keywords_list: list[str],
    api_key_override: str | None = None,
) -> str:
    """Main async logic for finding replacement."""

    MKEY = api_key_override
    key_source = "passed argument"
    if not MKEY:
        logger.critical(
            f"Job {job_id}: Missing required API key. No api_key_override was provided."
        )
        raise RuntimeError(
            "Missing required configuration: MISTRAL_API_KEY was not provided for the job."
        )

    masked_key = f"{'*' * (len(MKEY) - 4)}{MKEY[-4:]}" if len(MKEY) > 4 else "Short key"
    logger.info(
        f"Job {job_id}: Using Mistral Key. Source: {key_source}, Value (masked): {masked_key}"
    )

    SUGGESTION_MODEL = os.getenv("SUGGESTION_MODEL", "mistral-large-latest")
    VALIDATION_MODEL = os.getenv("VALIDATION_MODEL", "mistral-large-latest")
    TURNS = int(os.getenv("MAX_TURNS", "10"))
    blocked_found = None

    try:
        try:
            client = Mistral(api_key=MKEY)

            logger.info(f"Mistral client initialized for job {job_id}.")
        except (Exception,) as e:
            logger.error(f"Failed to initialize Mistral client for job {job_id}: {e}")
            raise RuntimeError(
                "Failed to initialize Mistral client: Invalid API Key or other issue."
            ) from e

        blocked_terms_str = ""
        pattern = None
        if blocked_keywords_list:
            blocked_unique = sorted(
                list(set(filter(None, [k.strip() for k in blocked_keywords_list]))),
                key=len,
                reverse=True,
            )
            if blocked_unique:
                blocked_terms_str = ", ".join(blocked_unique)
                try:
                    pattern_str = (
                        r"\b(" + "|".join(map(re.escape, blocked_unique)) + r")\b"
                    )
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    logger.info(
                        f"Using {len(blocked_unique)} provided blocked words for job {job_id}."
                    )
                except re.error as re_err:
                    logger.error(f"Regex compilation failed for job {job_id}: {re_err}")
                    blocked_terms_str = ""
                    pattern = None
            else:
                logger.warning(f"Provided keyword list was empty for job {job_id}.")
        else:
            logger.warning(f"Received empty keyword list for job {job_id}.")

        suggestion_cache = {}
        validation_cache = {}
        suggestion_messages: list[dict] = []
        initial_prompt = (
            f"Full document context:\n---\n{full_text}\n---\n\nExcerpt to"
            f' replace: "{excerpt}"\n\nPlease suggest a replacement for the'
            " excerpt that maintains the original meaning as much as possible"
            " within the context, following the rules provided in the system"
            " prompt."
        )
        suggestion_messages.append({"role": "user", "content": initial_prompt})
        last_suggestion = ""

        for turn in range(1, TURNS + 1):
            logger.info(f"--- Job {job_id} Turn {turn}/{TURNS} ---")

            sug_resp = None
            sug_key = (
                SUGGESTION_MODEL,
                json.dumps(suggestion_messages, sort_keys=True),
                blocked_terms_str,
            )

            if sug_key in suggestion_cache:
                logger.info(f"Cache hit for suggestion in job {job_id} turn {turn}")
                sug_resp = suggestion_cache[sug_key]
            else:
                logger.info(
                    f"Cache miss for suggestion job {job_id} turn {turn}, calling API..."
                )
                try:
                    sug_resp = await call_suggestion_api(
                        client, SUGGESTION_MODEL, suggestion_messages, blocked_terms_str
                    )
                    suggestion_cache[sug_key] = sug_resp
                except Exception as e:
                    logger.exception(
                        f"Error during suggestion API call job {job_id} turn {turn}: {e}"
                    )

                    if not is_rate_limit_error(e):
                        raise Exception(
                            f"Unretryable API error during suggestion: {e}"
                        ) from e

                    if turn == TURNS:
                        raise Exception(
                            f"Suggestion failed after multiple retries on turn {turn}."
                        ) from e
                    feedback = "API temporarily unavailable, retrying suggestion..."
                    suggestion_messages.append({"role": "user", "content": feedback})
                    continue

            sug = sug_resp.replacement.strip()
            logger.info(f"Suggestion received (job {job_id} turn {turn}): '{sug}'")

            assistant_message_content = sug_resp.model_dump_json()
            if (
                not suggestion_messages
                or suggestion_messages[-1].get("role") != "assistant"
                or suggestion_messages[-1].get("content") != assistant_message_content
            ):
                suggestion_messages.append(
                    {"role": "assistant", "content": assistant_message_content}
                )

            if sug == last_suggestion and turn > 1:
                logger.warning(f"Suggestion '{sug}' repeated (job {job_id}). Retrying.")
                feedback = (
                    f"Suggestion '{sug}' was already rejected. New alternative needed."
                )
                suggestion_messages.append({"role": "user", "content": feedback})
                if (
                    len(suggestion_messages) > 1
                    and suggestion_messages[-2]["role"] == "assistant"
                ):
                    suggestion_messages.pop(-2)
                continue

            last_suggestion = sug

            if pattern:
                match = pattern.search(sug)
                if match:
                    blocked_found = match.group(0)
                    logger.warning(
                        f"Suggestion '{sug}' blocked locally: '{blocked_found}' (job {job_id})."
                    )
                    feedback = f"Suggestion '{sug}' contains blocked term '{blocked_found}'. Avoid all blocked terms."
                    suggestion_messages.append({"role": "user", "content": feedback})
                    if (
                        len(suggestion_messages) > 1
                        and suggestion_messages[-2]["role"] == "assistant"
                    ):
                        suggestion_messages.pop(-2)
                    continue

            val_resp = None
            val_key = (VALIDATION_MODEL, excerpt, sug, full_text, blocked_terms_str)

            if val_key in validation_cache:
                logger.info(f"Cache hit for validation in job {job_id} turn {turn}")
                val_resp = validation_cache[val_key]
            else:
                logger.info(
                    f"Cache miss for validation job {job_id} turn {turn}, calling API..."
                )
                try:
                    logger.info(
                        f"Validating suggestion '{sug} for {excerpt} for blocked term {blocked_found if blocked_found else ''} for job {job_id}"
                    )
                    val_resp = await call_validation_api(
                        client,
                        VALIDATION_MODEL,
                        excerpt,
                        sug,
                        full_text,
                        blocked_terms_str,
                    )
                    validation_cache[val_key] = val_resp
                except Exception as e:
                    logger.exception(
                        f"Error during validation API call job {job_id} turn {turn}: {e}"
                    )
                    if not is_rate_limit_error(e):
                        raise Exception(
                            f"Unretryable API error during validation: {e}"
                        ) from e
                    if turn == TURNS:
                        raise Exception(
                            f"Validation failed after multiple retries on turn {turn}."
                        ) from e
                    feedback = "API temporarily unavailable, retrying validation..."
                    suggestion_messages.append({"role": "user", "content": feedback})

                    continue

            if val_resp.should_block:
                logger.warning(
                    f"Validator rejected '{sug}' for job {job_id}: {val_resp.reason}."
                )
                feedback = f"Validator rejected '{sug}': {val_resp.reason}. Try again."
                suggestion_messages.append({"role": "user", "content": feedback})
                if (
                    len(suggestion_messages) > 1
                    and suggestion_messages[-2]["role"] == "assistant"
                ):
                    suggestion_messages.pop(-2)
                continue
            else:
                logger.info(
                    f"Suggestion '{sug}' validated successfully for job {job_id}."
                )
                return sug

        logger.error(f"No valid suggestion found after {TURNS} turns for job {job_id}.")
        raise Exception(f"Failed to find valid suggestion after {TURNS} turns")

    finally:
        pass


def get_valid_replacement(
    job_id: str,
    full_text: str,
    excerpt: str,
    blocked_keywords_list: list[str],
    api_key_override: str | None = None,
) -> str:
    """Synchronous wrapper, ensures loop runs and returns result or raises error."""
    try:
        result = asyncio.run(
            _run_replacement_loop(
                job_id,
                full_text,
                excerpt,
                blocked_keywords_list,
                api_key_override,
            )
        )

        return result
    except Exception as e:
        logger.error(f"Job {job_id} failed execution: {e}", exc_info=True)
        raise
