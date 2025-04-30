# setup_pinecone.py

import time
import uuid
from pathlib import Path

import toml
from mistralai import Mistral
from pinecone import Pinecone, ServerlessSpec

SECRETS_FILE_PATH = Path(__file__).parent / ".streamlit/secrets.toml"
SPEC = ServerlessSpec(cloud="aws", region="us-east-1")


def load_secrets(file_path: Path) -> dict:
    print(f"Attempting to load secrets from: {file_path.resolve()}")
    if not file_path.is_file():
        print(f"ERROR: Secrets file not found at {file_path.resolve()}")
        exit(1)
    try:
        secrets = toml.load(file_path)
        required_pinecone = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
        required_mistral = ["MISTRAL_API_KEY"]
        missing = [k for k in required_pinecone + required_mistral if k not in secrets]
        if missing:
            raise ValueError(
                f"Missing required secrets in {file_path.name}: {', '.join(missing)}"
            )
        print("Secrets loaded successfully.")
        return secrets
    except Exception as e:
        print(f"Error loading secrets from {file_path.resolve()}: {e}")
        exit(1)


def setup_index(pc: Pinecone, index_name: str):
    print(f"\n--- Setting up Pinecone Index '{index_name}' ---")
    try:
        index_list_response = pc.list_indexes()
        existing_indexes = index_list_response.names()
        print(f"Found existing indexes: {existing_indexes}")

        if index_name not in existing_indexes:
            print(f"Index '{index_name}' not found. Creating...")
            pc.create_index(
                name=index_name, dimension=VECTOR_DIMENSION, metric=METRIC, spec=SPEC
            )
            print(
                f"Index '{index_name}' creation initiated. Waiting up to ~2 minutes for readiness..."
            )
            wait_time = 0
            max_wait = 120
            while True:
                if wait_time >= max_wait:
                    print(
                        f"ERROR: Index '{index_name}' did not become ready after {max_wait} seconds."
                    )
                    return False
                index_description = pc.describe_index(index_name)
                if index_description.status["ready"]:
                    print(f"Index '{index_name}' is ready.")
                    break
                print(f"Still waiting for index... ({wait_time}s)")
                time.sleep(5)
                wait_time += 5
            return True

        else:
            print(f"Index '{index_name}' already exists.")
            desc = pc.describe_index(index_name)
            correct_config = True
            if desc.dimension != VECTOR_DIMENSION:
                print(
                    f"WARNING: Index '{index_name}' has WRONG dimension! Expected: {VECTOR_DIMENSION}, Found: {desc.dimension}"
                )
                correct_config = False
            if desc.metric.lower() != METRIC.lower():
                print(
                    f"WARNING: Index '{index_name}' has WRONG metric! Expected: {METRIC}, Found: {desc.metric}"
                )
                correct_config = False

            if correct_config:
                print(f"Existing index '{index_name}' configuration is correct.")
                return True
            else:
                print(
                    f"ERROR: Existing index '{index_name}' has incorrect configuration. Please delete it manually via the Pinecone console and re-run this script."
                )
                return False

    except Exception as e:
        print(f"An error occurred during Pinecone index setup: {e}")
        return False


def test_operations(pc: Pinecone, index_name: str, mistral_api_key: str):
    print("\n--- Running Basic API Test ---")
    test_text = "This is a sample sentence to test API operations."
    test_replacement = "This is a test replacement stored in Pinecone."
    mistral_client = None
    pinecone_index = None
    embedding_list = None

    try:
        print("Initializing Mistral client for test...")
        mistral_client = Mistral(api_key=mistral_api_key)
        print("Initializing Pinecone index object for test...")
        pinecone_index = pc.Index(index_name)
        print("Clients initialized.")
    except Exception as e:
        print(f"ERROR: Failed to initialize clients during test: {e}")
        return

    try:
        print(f"Getting embedding for: '{test_text}'")
        response = mistral_client.embeddings.create(inputs=[test_text], model=EMBEDDING_MODEL)
        if response.data and len(response.data) > 0:
            embedding_list = response.data[0].embedding
            print(f"Embedding received (vector length: {len(embedding_list)}).")
        else:
            print("ERROR: Mistral embedding call returned no data.")
            return
    except Exception as e:
        print(f"ERROR: Failed to get embedding from Mistral: {e}")
        return

    try:
        print(f"Querying Pinecone index '{index_name}' with embedding...")
        query_response = pinecone_index.query(
            vector=embedding_list, top_k=2, include_metadata=True
        )
        print(
            f"Pinecone query successful. Found {len(query_response.matches)} matches (showing top 2)."
        )
        if query_response.matches:
            for match in query_response.matches:
                meta_str = str(match.metadata)
                if len(meta_str) > 100:
                    meta_str = meta_str[:100] + "..."
                print(f"  - ID: {match.id}, Score: {match.score:.4f}, Meta: {meta_str}")
        else:
            print("  - No similar vectors found (expected if index is new/empty).")

    except Exception as e:
        print(f"ERROR: Failed to query Pinecone index: {e}")

    try:
        test_vector_id = f"test-{uuid.uuid4()}"
        print(
            f"Upserting test vector '{test_vector_id}' to Pinecone index '{index_name}'..."
        )
        metadata_payload = {
            "replacement": test_replacement,
            "original_excerpt": test_text,
            "source": "setup_script_test",
        }
        upsert_response = pinecone_index.upsert(
            vectors=[(test_vector_id, embedding_list, metadata_payload)]
        )
        print(f"Pinecone upsert successful. Count: {upsert_response.upserted_count}")
        print(
            f"NOTE: You can manually delete vector '{test_vector_id}' from the Pinecone console if desired."
        )
    except Exception as e:
        print(f"ERROR: Failed to upsert vector to Pinecone index: {e}")

    print("--- Basic API Test Finished ---")


if __name__ == "__main__":
    print("Starting Pinecone Setup and Test Script...")
    secrets = load_secrets(SECRETS_FILE_PATH)

    PINECONE_API_KEY = secrets["PINECONE_API_KEY"]
    PINECONE_INDEX_NAME = secrets["PINECONE_INDEX_NAME"]
    MISTRAL_API_KEY = secrets["MISTRAL_API_KEY"]
    EMBEDDING_MODEL = secrets.get("EMBEDDING_MODEL", "mistral-7b-v0.1")
    VECTOR_DIMENSION = secrets.get("VECTOR_DIMENSION", 4096)
    METRIC = secrets.get("METRIC", "cosine")

    try:
        print("Initializing Pinecone client for setup...")
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        print(f"FATAL: Could not initialize Pinecone client: {e}")
        exit(1)

    setup_successful = setup_index(pinecone_client, PINECONE_INDEX_NAME)

    if setup_successful:
        test_operations(pinecone_client, PINECONE_INDEX_NAME, MISTRAL_API_KEY)
    else:
        print("\nSkipping tests due to index setup issues.")

    print("\nScript finished.")