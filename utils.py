import logging
import sys
import time
import traceback

import streamlit as st

MAX_LOG_LINES = 5000

logger = logging.getLogger("keyword_replacer_app")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s (%(name)s): %(message)s")
    )
    logger.addHandler(handler)

if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []


def log(msg: str, level: str = "info", exc_info: bool = False):
    """Logs message and updates session state buffer."""
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {level.upper()}: {msg}"
    if exc_info:
        exception_details = traceback.format_exc()
        log_entry += f"\n{exception_details}"
    getattr(logger, level)(msg, exc_info=exc_info)
    if "log_buffer" in st.session_state:
        st.session_state.log_buffer.append(log_entry)
        if len(st.session_state.log_buffer) > MAX_LOG_LINES:
            del st.session_state.log_buffer[
                : len(st.session_state.log_buffer) - MAX_LOG_LINES
            ]
    else:
        print(f"Log (Pre-Session): {log_entry}")
