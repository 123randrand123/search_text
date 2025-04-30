import os
from pathlib import Path

DEFAULT_KEYWORDS_FILENAME = Path("keywords.enc")
FERNET_KEY = os.getenv("FERNET_KEY")
MISTRAL_API_KEY_ENV = os.getenv("MISTRAL_API_KEY")
MODEL = os.getenv("MODEL", "mistral-small-latest")
