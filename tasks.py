import asyncio
import json
import os
import re
import uuid
from collections import Counter

from mistralai import Mistral
from pinecone import Pinecone

from api import call_suggestion_api, call_validation_api
from api_utils import is_rate_limit_error
from utils import logger

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")
KNN_K = 3
KNN_SIMILARITY_THRESHOLD = 0.9


async def _run_replacement_loop(
    job_id: str,
    full_text: str,
    excerpt: str,
    blocked_keywords_list: list[str],
    api_key_override: str | None = None,
    pinecone_api_key: str | None = None,
    pinecone_index_name: str | None = None,
) -> str | None:
    """Main async logic for finding replacement."""

    if not api_key_override:
        logger.critical(f"Job {job_id}: Missing required Mistral API key.")
        return None
    if not pinecone_api_key or not pinecone_index_name:
        logger.error(f"Job {job_id}: Missing required Pinecone configuration.")
        # Decide: Fail hard or proceed without KNN? Let's proceed without KNN.
        pinecone_available = False
    else:
        pinecone_available = True

    mistral_client = None
    pinecone_client = None
    pinecone_index = None

    try:
        mistral_client = Mistral(api_key=api_key_override)
        logger.info(f"Mistral client initialized for job {job_id}.")
    except Exception as e:
        logger.error(f"Failed to initialize Mistral client for job {job_id}: {e}")
        return None  # Cannot proceed without Mistral

    if pinecone_available:
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            # Simple check if index exists. Add creation logic here if desired.
            if pinecone_index_name in pc.list_indexes().names():
                pinecone_index = pc.Index(pinecone_index_name)
                logger.info(
                    f"Pinecone client initialized and connected to index '{pinecone_index_name}' for job {job_id}."
                )
            else:
                logger.error(
                    f"Pinecone index '{pinecone_index_name}' not found. Cannot use KNN for job {job_id}."
                )
                pinecone_available = False  # Disable Pinecone use if index missing
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client for job {job_id}: {e}")
            pinecone_available = False  # Disable Pinecone use on init error
    input_embedding_list: list[float] | None = None
    try:
        logger.info(f"Job {job_id}: Requesting embedding for excerpt...")
        response = await asyncio.to_thread(
            mistral_client.embeddings.create, model=EMBEDDING_MODEL, inputs=[excerpt]
        )
        if response.data and len(response.data) > 0:
            input_embedding_list = response.data[0].embedding
            logger.info(f"Job {job_id}: Embedding received.")
        else:
            logger.error(f"Job {job_id}: Embedding API call returned no data.")
    except Exception as e:
        logger.error(f"Job {job_id}: Failed to get embedding for excerpt: {e}")
        # Proceed without embedding? Or fail? Let's try to proceed without KNN.

    # --- KNN Check (Pinecone Query) ---
    if (
        pinecone_available
        and pinecone_index is not None
        and input_embedding_list is not None
    ):
        try:
            logger.info(f"Job {job_id}: Querying Pinecone for similar excerpts...")
            query_response = await asyncio.to_thread(
                pinecone_index.query,
                vector=input_embedding_list,
                top_k=KNN_K,
                include_metadata=True,
            )

            candidate_replacements = []
            if query_response.matches:
                for match in query_response.matches:
                    if match.score >= KNN_SIMILARITY_THRESHOLD:
                        logger.info(
                            f"Job {job_id}: Found KNN match with score {match.score}."
                        )
                        metadata = match.metadata
                        if metadata and "replacement" in metadata:
                            candidate_replacements.append(metadata["replacement"])

            if candidate_replacements:
                count = Counter(candidate_replacements)
                most_common = count.most_common(1)
                if most_common:
                    knn_suggestion = most_common[0][0]
                    # Basic check: don't suggest the same thing
                    if knn_suggestion.strip().lower() != excerpt.strip().lower():
                        logger.info(
                            f"Job {job_id}: KNN pre-suggestion found via Pinecone: '{knn_suggestion}'."
                        )
                        return knn_suggestion  # <<< Return KNN result
                    else:
                        logger.warning(
                            f"Job {job_id}: KNN suggestion same as original, ignoring."
                        )

        except Exception as e:
            logger.error(f"Job {job_id}: Error during Pinecone query: {e}")
            # Fall through to Mistral API

    # --- Fallback to Mistral Suggestion Loop ---
    logger.info(
        f"Job {job_id}: KNN did not provide suggestion, proceeding with Mistral API."
    )
    SUGGESTION_MODEL = os.getenv("SUGGESTION_MODEL", "mistral-large-latest")
    VALIDATION_MODEL = os.getenv("VALIDATION_MODEL", "mistral-large-latest")
    TURNS = int(os.getenv("MAX_TURNS", "10"))
    blocked_found = None

    try:
        try:
            client = Mistral(api_key=api_key_override)

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
                if (
                    pinecone_available
                    and pinecone_index is not None
                    and input_embedding_list is not None
                    and sug
                ):
                    try:
                        vector_id = str(uuid.uuid4())
                        metadata_payload = {
                            "replacement": sug,
                            "original_excerpt": excerpt,
                        }
                        logger.info(
                            f"Job {job_id}: Upserting successful suggestion to Pinecone..."
                        )
                        await asyncio.to_thread(
                            pinecone_index.upsert,
                            vectors=[
                                (vector_id, input_embedding_list, metadata_payload)
                            ],
                        )
                        logger.info(f"Job {job_id}: Suggestion upserted to Pinecone.")
                    except Exception as e:
                        logger.error(
                            f"Job {job_id}: Failed to upsert suggestion to Pinecone: {e}"
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
    pinecone_api_key: str | None = None,
    pinecone_index_name: str | None = None,
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
                pinecone_api_key,
                pinecone_index_name,
            )
        )

        return result
    except Exception as e:
        logger.error(f"Job {job_id} failed execution: {e}", exc_info=True)
        raise
