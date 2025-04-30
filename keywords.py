import streamlit as st
from cryptography.fernet import Fernet, InvalidToken

from constants import DEFAULT_KEYWORDS_FILENAME
from utils import logger


@st.cache_data()
def load_default_keywords(key):
    """Loads/decrypts default keywords. Returns list or error string."""
    logger.debug("Attempting to load/decrypt default keywords...")
    if not key:
        logger.warning(
            f"FERNET_KEY missing, cannot load from {DEFAULT_KEYWORDS_FILENAME}."
        )
        return "ERROR:Encrypted File - No Key"
    try:
        with DEFAULT_KEYWORDS_FILENAME.open("rb") as f:
            blob = f.read()
        fernet = Fernet(key.encode())
        decrypted_bytes = fernet.decrypt(blob)
        raw = decrypted_bytes.decode("utf-8").splitlines()
        blocked = sorted([w.strip() for w in raw if w.strip()], key=len, reverse=True)
        logger.info(f"Decrypted {len(blocked)} default keywords")
        return blocked
    except FileNotFoundError:
        logger.error(f"Default keyword file '{DEFAULT_KEYWORDS_FILENAME}' not found.")
        return "ERROR:FileNotFound"
    except InvalidToken:
        logger.error("Failed decrypting default keywords: Invalid Key or File")
        return "ERROR:Invalid Key or Corrupted File"
    except Exception as e:
        logger.error(f"Error loading/decrypting default keywords: {e}", exc_info=True)
        return f"ERROR:Unexpected - {e}"


def parse_custom_keywords(text_input):
    """Parses multiline text into a sorted list of unique keywords."""
    if not text_input:
        return []
    lines = text_input.strip().splitlines()
    keywords_set = {line.strip() for line in lines if line.strip()}
    return sorted(list(keywords_set), key=len, reverse=True)
