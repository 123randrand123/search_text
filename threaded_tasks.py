import threading
import queue

from tasks import get_valid_replacement
from utils import log


def suggestion_worker(
    match_index: int,
    result_q: queue.Queue,
    session_id: str,  # Pass session_id for context
    full_text: str,
    excerpt: str,
    blocked_keywords: list[str],
    api_key: str,
):
    """
    Target function executed in a separate thread to get suggestions.
    Communicates results back via the provided queue.
    IMPORTANT: This function should NOT use Streamlit 'st.*' commands directly.
    """
    thread_name = threading.current_thread().name
    job_id_for_log = (
        f"thread-{session_id}-{match_index}"  # Construct a unique ID for logging
    )

    log(
        f"{thread_name} ({job_id_for_log}): Starting suggestion for match {match_index} ('{excerpt}')",
        "info",
    )
    try:
        # Call the core logic function from tasks.py
        result = get_valid_replacement(
            job_id=job_id_for_log,  # Pass the ID for logging within the task
            full_text=full_text,
            excerpt=excerpt,
            blocked_keywords_list=blocked_keywords,
            api_key_override=api_key,
        )
        log(f"{thread_name} ({job_id_for_log}): Suggestion successful.", "info")
        # Put a tuple (index, result_data) into the queue
        result_q.put((match_index, result))  # Send back the string result

    except Exception as e:
        # Log the full exception if possible
        log(
            f"{thread_name} ({job_id_for_log}): Suggestion FAILED for match {match_index}. Error: {e}",
            "error",
            exc_info=True,
        )
        # Put an error indicator back into the queue
        # Send a dictionary indicating error, including type and message
        result_q.put((match_index, {"error": f"{type(e).__name__}: {str(e)}"}))

    log(f"{thread_name} ({job_id_for_log}): Exiting.", "info")


def start_suggestion_thread(
    match_index: int,
    result_q: queue.Queue,
    session_id: str,
    full_text: str,
    excerpt: str,
    blocked_keywords: list[str],
    api_key: str,
):
    """Creates and starts the background suggestion thread."""
    thread_name = f"SuggestThread-{match_index}"
    log(f"Main Thread: Launching {thread_name}", "info")

    thread = threading.Thread(
        target=suggestion_worker,
        args=(
            match_index,
            result_q,
            session_id,
            full_text,
            excerpt,
            blocked_keywords,
            api_key,
        ),
        daemon=True,
        name=thread_name,
    )
    thread.start()
