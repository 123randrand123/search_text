# test_task.py

import time
from pathlib import Path
import toml

try:
    from tasks import get_valid_replacement
    from utils import logger # Import logger if you want detailed logs
except ImportError:
    print("ERROR: Could not import 'get_valid_replacement' from tasks.py.")
    print("Ensure test_task.py is in the correct directory relative to tasks.py")
    exit(1)

# --- Configuration ---
SECRETS_FILE_PATH = Path(__file__).parent / ".streamlit/secrets.toml"

# --- Helper Function to Load Secrets (same as in setup script) ---
def load_secrets(file_path: Path) -> dict:
    """Loads secrets from a TOML file."""
    if not file_path.is_file():
        print(f"ERROR: Secrets file not found at {file_path.resolve()}")
        exit(1)
    try:
        secrets = toml.load(file_path)
        required_pinecone = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
        required_mistral = ["MISTRAL_API_KEY"]
        missing = [k for k in required_pinecone + required_mistral if k not in secrets]
        if missing:
            raise ValueError(f"Missing required secrets in {file_path.name}: {', '.join(missing)}")
        print("Secrets loaded successfully.")
        return secrets
    except Exception as e:
        print(f"Error loading secrets from {file_path.resolve()}: {e}")
        exit(1)

# --- Test Execution ---
if __name__ == "__main__":
    print("Starting Task Test Script...")
    secrets = load_secrets(SECRETS_FILE_PATH)

    # --- Test Data ---
    test_job_id_1 = "test_knn_miss"
    test_job_id_2 = "test_knn_hit"

    # Example text that might cause a KNN miss initially
    test_excerpt_miss = "a unique phrase needing replacement"
    test_full_text_miss = f"""
    This is a longer document containing context. We need to find a replacement
    for {test_excerpt_miss} which has specific meaning here. The goal is to
    preserve the intent while using different wording.
    """

    # Example text matching the one used in setup_pinecone.py's test
    # This is intended to check if the KNN lookup finds the pre-inserted vector
    test_excerpt_hit = "a unique phrase needing change"
    test_full_text_hit = f"""
    This is a longer document containing context. We need to find a replacement
    for {test_excerpt_hit} which has specific meaning here. The goal is to
    preserve the intent while using different wording.
    """

    test_blocked_keywords: list[str] = ["forbidden", "restricted"]

    # --- Run Test Case 1: KNN Miss -> Mistral -> Pinecone Upsert ---
    print(f"\n--- Test Case 1: KNN Miss (Excerpt: '{test_excerpt_miss}') ---")
    start_time = time.time()
    try:
        suggestion1 = get_valid_replacement(
            job_id=test_job_id_1,
            full_text=test_full_text_miss,
            excerpt=test_excerpt_miss,
            blocked_keywords_list=test_blocked_keywords,
            api_key_override=secrets["MISTRAL_API_KEY"],
            pinecone_api_key=secrets["PINECONE_API_KEY"],
            pinecone_index_name=secrets["PINECONE_INDEX_NAME"],
        )
        if suggestion1:
            print(f"Success! Suggestion received: '{suggestion1}'")
            print("Check logs above for 'proceeding with Mistral API' and 'Upserting'.")
        else:
            print("Failed: get_valid_replacement returned None.")
    except Exception as e:
        print(f"Failed: Exception occurred: {type(e).__name__}: {e}")
    end_time = time.time()
    print(f"Test Case 1 Duration: {end_time - start_time:.2f} seconds")


    # --- Run Test Case 2: KNN Hit ---
    print(f"\n--- Test Case 2: KNN Hit (Excerpt: '{test_excerpt_hit}') ---")
    # Requires setup_pinecone.py to have run successfully before
    start_time = time.time()
    try:
        suggestion2 = get_valid_replacement(
            job_id=test_job_id_2,
            full_text=test_full_text_hit,
            excerpt=test_excerpt_hit,
            blocked_keywords_list=test_blocked_keywords,
            api_key_override=secrets["MISTRAL_API_KEY"],
            pinecone_api_key=secrets["PINECONE_API_KEY"],
            pinecone_index_name=secrets["PINECONE_INDEX_NAME"],
        )
        if suggestion2:
            print(f"Success! Suggestion received: '{suggestion2}'")
            print("Check logs above for 'KNN pre-suggestion found'. Mistral API should NOT have been called.")
            # Verify if the suggestion matches the one upserted by setup_pinecone.py
            expected_knn_hit_suggestion = "This is a test replacement stored in Pinecone."
            if suggestion2 == expected_knn_hit_suggestion:
                 print("Suggestion matches expected KNN result.")
            else:
                 print(f"WARNING: Suggestion '{suggestion2}' received, but expected '{expected_knn_hit_suggestion}' from KNN.")
        else:
            print("Failed: get_valid_replacement returned None.")
    except Exception as e:
        print(f"Failed: Exception occurred: {type(e).__name__}: {e}")
    end_time = time.time()
    print(f"Test Case 2 Duration: {end_time - start_time:.2f} seconds")

    print("\n--- Test Script Finished ---")