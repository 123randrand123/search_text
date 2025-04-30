import queue
import re
import sys
import time
import uuid

import streamlit as st
from streamlit import runtime
from streamlit.web import cli


from constants import (
    FERNET_KEY,
    MISTRAL_API_KEY_ENV,
)
from doc_utils import pdf_to_text, docx_to_text
from keywords import load_default_keywords, parse_custom_keywords
from threaded_tasks import start_suggestion_thread
from utils import log, logger


POLL_INTERVAL = 2.0  # Seconds for polling background tasks
st.set_page_config(layout="wide", page_title="Keyword Replacer")


# --- Initialize Session State ---
def init_session_state():
    """Initializes session state variables."""
    log("Initializing session state.", "debug")
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("mistral_api_key", MISTRAL_API_KEY_ENV)
    st.session_state.setdefault("use_custom_keywords", False)
    st.session_state.setdefault("custom_keywords_input", "")
    st.session_state.setdefault("default_keywords_cache", None)
    st.session_state.setdefault("active_keywords_list", None)
    st.session_state.setdefault("pattern_compiled", None)
    st.session_state.setdefault("last_processed_keyword_mode", None)
    st.session_state.setdefault("last_processed_custom_keywords", None)
    st.session_state.setdefault(
        "replacements", {}
    )  # Stores {match_index: replacement_str | error_dict}
    st.session_state.setdefault("apply_selection", {})  # Stores {match_index: bool}
    st.session_state.setdefault("current_text", "")
    st.session_state.setdefault("last_file_id", None)
    st.session_state.setdefault("last_text_input", None)
    st.session_state.setdefault("last_mode", "Upload PDF/DOCX")
    st.session_state.setdefault("session_logged", False)
    st.session_state.setdefault("show_key_error_on_suggest", False)
    st.session_state.setdefault("result_queue", queue.Queue())
    st.session_state.setdefault(
        "tasks_pending", set()
    )  # Set of match indices being processed by AI
    st.session_state.setdefault(
        "corrected_text_output", None
    )  # Stores the final output
    # Removed 'manual_inputs' - will read directly from widget state


# --- State Clearing Function ---
def clear_job_related_state():
    """Clears state related to a specific text processing job."""
    log("Clearing job-related state (replacements, selection, tasks, output).", "info")
    st.session_state.replacements = {}
    st.session_state.apply_selection = {}
    st.session_state.tasks_pending = set()
    st.session_state.corrected_text_output = None
    st.session_state.pop("matches_for_current_text", None)
    keys_to_clear = [k for k in st.session_state if k.startswith("match_")]
    for k in keys_to_clear:
        del st.session_state[k]
    log(f"Cleared {len(keys_to_clear)} specific match widget states.", "debug")


# --- Callback Functions ---
# Defined outside app() for clarity, they operate on st.session_state


def handle_manual_input_change(match_index):
    """Callback when a manual text input changes."""
    widget_key = f"match_{match_index}_manual_input"
    current_manual_value = st.session_state.get(widget_key, "").strip()
    log(
        f"Callback: Manual input changed for match {match_index} to: '{current_manual_value}'",
        "debug",
    )

    if current_manual_value:
        # Store the manual input as the definitive replacement
        st.session_state.replacements[match_index] = current_manual_value
        # Automatically select it for application
        st.session_state.apply_selection[match_index] = True
        # Manual input overrides any pending/failed AI task for this match
        st.session_state.tasks_pending.discard(match_index)
        log(
            f"Callback: Set replacement and selection=True for match {match_index} due to manual input.",
            "info",
        )
    else:
        # If manual input is cleared, remove the replacement *if* it was the manual one
        # and deselect. Don't remove if it was a successful AI suggestion.
        current_replacement = st.session_state.replacements.get(match_index)
        # Check if the current replacement is NOT an error dict (meaning it's AI or previous manual)
        # This logic might need refinement depending on desired behavior when clearing manual input
        # For now, let's just deselect if the box is cleared.
        # We might need to know if the current replacement was manual or AI.
        # Let's simplify: if manual is cleared, deselect. User can re-select AI if needed.
        st.session_state.apply_selection[match_index] = False
        # Maybe remove the replacement entirely? Or keep AI suggestion if it exists?
        # Let's remove replacement only if it wasn't an error dict (don't remove error messages)
        if not isinstance(current_replacement, dict):
            st.session_state.replacements.pop(match_index, None)
        log(
            f"Callback: Cleared manual input for match {match_index}, setting selection=False.",
            "info",
        )


def handle_apply_ai_change(match_index):
    """Callback when the 'Use AI' checkbox changes."""
    widget_key = f"match_{match_index}_apply_ai"
    is_checked = st.session_state.get(widget_key, False)
    log(
        f"Callback: Apply AI checkbox changed for match {match_index} to: {is_checked}",
        "debug",
    )
    st.session_state.apply_selection[match_index] = is_checked

    # Optional: Clear manual input when AI is selected?
    # if is_checked:
    #     manual_input_key = f"match_{match_index}_manual_input"
    #     st.session_state[manual_input_key] = "" # Clear the widget state
    #     log(f"Callback: Cleared manual input for match {match_index} because AI was selected.", "debug")


# --- Main Application Logic ---
def app():
    # Initialize state on first run
    init_session_state()

    if not st.session_state.session_logged:
        log(f"Session started: {st.session_state.session_id}", "info")
        st.session_state.session_logged = True

    # --- UI Sections ---
    st.markdown("## Keyword Replacer")
    # (Keep instructions, API key, keyword config sections as they were - they seemed okay)
    st.markdown("## How to Use This App")
    st.markdown(
        """
        1.  **(Optional) Enter API Key:** If you want AI suggestions, get a Mistral API key (see below) and paste it in the 'Configure API Key' section. You can skip this initially.
        2.  **Input Text:** Either upload a PDF/DOCX file or paste your text directly into the text area.
        3.  **Configure Keywords:** Choose whether to use the default list of sensitive keywords or enter your own custom list (one per line). Keywords are used to find text segments to replace.
        4.  **Process Matches:** For each keyword found:
            *   Click 'Suggest AI' to get an AI alternative (Requires API Key).
            *   OR, type your replacement directly into the 'Manual Replacement' box. Typing here automatically selects it for replacement.
            *   If an AI suggestion is available, check 'Use this AI Suggestion' to select it.
        5.  **Apply & Download:** Click 'Apply Selected Replacements'. A preview of the corrected text will appear. You can then download this corrected text as a `.txt` file.
        """
    )
    st.divider()

    # --- API Key Configuration ---
    st.markdown("### Configure API Key")
    active_mistral_key = st.session_state.mistral_api_key
    key_present = bool(active_mistral_key)
    key_expander_expanded = not key_present
    with st.expander(
        "Enter Mistral API Key (Needed for AI Suggestions)",
        expanded=key_expander_expanded,
    ):
        if not key_present:
            st.warning(
                "No Mistral API Key found. AI suggestions are disabled until a key is entered.",
                icon="🔑",
            )

        st.markdown(
            """
            **How to Get a Free* Mistral API Key:**
            1.  **Go to Mistral AI:** Open [console.mistral.ai](https://console.mistral.ai/) in your web browser.
            2.  **Sign Up / Log In:** Create a free account or log in if you already have one.
            3.  **Navigate to API Keys:** Look for a section called "API Keys" or similar in your account dashboard (usually on the left sidebar).
            4.  **Create a Key:** Click the button to "Create new key". Give it a name (e.g., "KeywordReplacerApp").
            5.  **Copy Key Immediately:** Mistral will show you the key **only once**. Copy it right away!
            6.  **Paste Below:** Paste the copied key into the text box below.
            7.  **Keep it Secret:** Treat this key like a password. Don't share it publicly.

            \\*Mistral usually offers a free tier or trial credits sufficient for moderate use.*
            """,
            unsafe_allow_html=True,
        )
        entered_key = st.text_input(
            "Paste Mistral API Key here:",
            key="api_key_input",
            type="password",
            value=st.session_state.mistral_api_key or "",
            label_visibility="collapsed",
        )
        if entered_key and entered_key != st.session_state.mistral_api_key:
            st.session_state.mistral_api_key = entered_key
            active_mistral_key = entered_key
            log("Mistral API Key entered/updated via UI.", "info")
            st.session_state.show_key_error_on_suggest = False
            st.success("API Key stored for this session.", icon="✅")
            time.sleep(0.5)
            if not key_present:
                st.rerun()  # Rerun only if key was initially missing

    if (
        st.session_state.get("show_key_error_on_suggest", False)
        and not active_mistral_key
    ):
        st.error(
            "Please enter your Mistral API Key above before requesting suggestions.",
            icon="🔑",
        )
    st.divider()

    # --- Keyword Configuration & Pattern Compilation ---
    keywords_error_msg = None
    with st.expander("Configure Keywords"):
        st.checkbox("Use Custom Keyword List", key="use_custom_keywords")
        st.text_area(
            "Custom Keywords (one per line):",
            key="custom_keywords_input",
            height=150,
            disabled=not st.session_state.use_custom_keywords,
            help="Enter keywords/phrases. Changing keywords clears existing suggestions.",
        )

    current_keyword_mode = st.session_state.use_custom_keywords
    current_custom_keywords_text = st.session_state.custom_keywords_input
    reprocess_keywords = False
    if current_keyword_mode != st.session_state.get("last_processed_keyword_mode") or (
        current_keyword_mode
        and current_custom_keywords_text
        != st.session_state.get("last_processed_custom_keywords")
    ):
        reprocess_keywords = True
        log("Keyword configuration changed, will reprocess.", "info")

    if reprocess_keywords or st.session_state.active_keywords_list is None:
        log(f"Processing keywords (Custom: {current_keyword_mode}).", "debug")
        # ... (Keyword loading/parsing logic - seems okay) ...
        active_keywords = []
        keywords_error_msg = None
        if st.session_state.use_custom_keywords:
            active_keywords = parse_custom_keywords(current_custom_keywords_text)
            # ... (logging/warnings) ...
        else:
            # ... (load default keywords) ...
            if st.session_state.default_keywords_cache is None:
                st.session_state.default_keywords_cache = load_default_keywords(
                    FERNET_KEY
                )
            default_result = st.session_state.default_keywords_cache
            # ... (handle errors/results) ...
            if isinstance(default_result, list):
                active_keywords = default_result
            elif isinstance(default_result, str):
                keywords_error_msg = default_result

        st.session_state.active_keywords_list = active_keywords
        st.session_state.last_processed_keyword_mode = current_keyword_mode
        st.session_state.last_processed_custom_keywords = current_custom_keywords_text

        # --- Compile Pattern ---
        pattern_global = None
        if active_keywords:
            try:
                escaped = [re.escape(str(k)) for k in active_keywords if k]
                if escaped:
                    pattern_str = r"\b(" + "|".join(escaped) + r")\b"
                    pattern_global = re.compile(pattern_str, re.IGNORECASE)
                    log(f"Keyword pattern compiled with {len(escaped)} terms.", "debug")
                else:
                    keywords_error_msg = (
                        keywords_error_msg or "Warning: No valid keywords."
                    )
            except re.error as re_err:
                keywords_error_msg = f"ERROR: Regex Compile Error - {re_err}"
                active_keywords = []  # Invalidate keywords on regex error
        st.session_state.pattern_compiled = pattern_global
        st.session_state.active_keywords_list = (
            active_keywords  # Update list if regex failed
        )

        if reprocess_keywords:
            log("Keywords reprocessed, clearing job state.", "info")
            clear_job_related_state()  # Clear old results if keywords change

    # Display keyword errors
    if keywords_error_msg:
        if keywords_error_msg.startswith("ERROR:"):
            st.error(f"Keyword Error: {keywords_error_msg.split(':', 1)[1].strip()}")
        else:
            st.warning(keywords_error_msg)
    st.divider()

    # --- Input & Suggestions ---
    st.markdown("### Input & Suggestions")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📄 1. Input Document")
        # --- Input Mode Selection ---
        mode_options = ["Upload PDF/DOCX", "Paste text"]
        try:
            current_mode_index = mode_options.index(st.session_state.last_mode)
        except ValueError:
            current_mode_index = 0
        selected_mode = st.radio(
            "Choose input method:",
            options=mode_options,
            index=current_mode_index,
            horizontal=True,
            key="input_mode_radio",
            label_visibility="collapsed",
        )

        # Handle mode change
        if selected_mode != st.session_state.last_mode:
            log(f"Mode switched to '{selected_mode}'. Clearing text and jobs.", "info")
            st.session_state.last_mode = selected_mode
            st.session_state.current_text = ""
            st.session_state.last_file_id = None
            st.session_state.last_text_input = None
            clear_job_related_state()
            st.rerun()  # Rerun needed immediately on mode switch

        # --- Text Input Handling ---
        text_changed = False
        text = ""  # Default to empty text

        if st.session_state.last_mode == "Upload PDF/DOCX":
            uploaded_file = st.file_uploader(
                "Select PDF or DOCX file",
                type=["pdf", "docx"],
                key="file_uploader_widget",
            )
            if uploaded_file:
                file_id = (
                    f"{uploaded_file.name}-{uploaded_file.size}"
                )
                if file_id != st.session_state.get("last_file_id"):
                    log(f"Processing uploaded file: {uploaded_file.name}", "info")
                    st.session_state.last_file_id = file_id
                    st.session_state.last_text_input = (
                        None  # Clear other input source tracker
                    )
                    clear_job_related_state()  # Clear old results
                    with st.spinner(f"Reading {uploaded_file.name}..."):
                        try:
                            extracted_text = (
                                pdf_to_text(uploaded_file.getvalue())
                                if uploaded_file.type == "application/pdf"
                                else docx_to_text(uploaded_file.getvalue())
                            )
                        except Exception as e:
                            logger.error(
                                f"Error extracting text from {uploaded_file.name}: {e}",
                                exc_info=True,
                            )
                            st.error(f"Error reading file {uploaded_file.name}.")
                            extracted_text = None
                    st.session_state.current_text = (
                        extracted_text if extracted_text is not None else ""
                    )
                    text = st.session_state.current_text  # Assign to local 'text' variable
                    log("New file detected, finding matches immediately.", "debug")
                    matches = []
                    active_pattern = st.session_state.pattern_compiled  # Get compiled pattern
                    if text and active_pattern:  # Check if we have text and a pattern
                        try:
                            matches = [
                                {
                                    "span": m.span(),
                                    "match_text": m.group(0),
                                    "excerpt": text[
                                               max(0, m.start() - 50): min(
                                                   len(text), m.end() + 50
                                               )
                                               ],
                                    "start": m.start(),
                                    "end": m.end(),
                                }
                                for m in active_pattern.finditer(text)
                            ]
                            log(f"Found {len(matches)} matches in new file.", "info")
                        except Exception as e:
                            logger.error(f"Error during regex finditer for new file: {e}", exc_info=True)
                            st.error("An error occurred while searching keywords in the new file.")
                    else:
                        log(f"Skipping match finding for new file (Text empty: {not text}, Pattern invalid: {not active_pattern})",
                            "debug")

                    st.session_state.matches_for_current_text = matches  # Store matches *before* rerun
                    # ----------------------------------------------------------

                    st.rerun()  # Rerun to update UI with the new text and matches
                else:
                    # File hasn't changed, use current text
                    text = st.session_state.current_text
                    # Retrieve previously found matches for this text
                    matches = st.session_state.get("matches_for_current_text", [])  # Default to empty list if not found
            elif st.session_state.get("last_file_id") is not None:
                # File was removed
                log("Uploaded file removed.", "info")
                st.session_state.current_text = ""
                st.session_state.last_file_id = None
                clear_job_related_state()
                st.rerun()  # Rerun to clear display
            # else: no file uploaded or present

        else:  # Paste text mode
            # Use a callback for the text_area to detect changes
            def text_area_on_change():
                log("Pasted text area changed.", "debug")
                # Check if content actually changed before clearing state
                if st.session_state.text_area_input != st.session_state.get(
                    "last_text_input"
                ):
                    log("Pasted text content differs, updating state.", "info")
                    st.session_state.current_text = st.session_state.text_area_input
                    st.session_state.last_text_input = st.session_state.current_text
                    st.session_state.last_file_id = (
                        None  # Clear other input source tracker
                    )
                    clear_job_related_state()  # Clear results for new text
                    text = st.session_state.current_text  # Use the new text
                    matches = []
                    active_pattern = st.session_state.pattern_compiled
                    if text and active_pattern:
                        try:
                            matches = [
                                {
                                    "span": m.span(),
                                    "match_text": m.group(0),
                                    "excerpt": text[max(0, m.start() - 50): min(len(text), m.end() + 50)],
                                    "start": m.start(),
                                    "end": m.end(),
                                }
                                for m in active_pattern.finditer(text)
                            ]
                            log(f"Found {len(matches)} matches in pasted text.", "info")
                        except Exception as e:
                            logger.error(f"Error during regex finditer for pasted text: {e}", exc_info=True)
                            st.error("An error occurred while searching keywords in pasted text.")
                    else:
                        log(f"Skipping match finding for pasted text (Text empty: {not text}, Pattern invalid: {not active_pattern})",
                            "debug")

                    st.session_state.matches_for_current_text = matches
                else:
                    log(
                        "Pasted text area changed callback fired, but content is the same.",
                        "debug",
                    )

            st.text_area(
                "Paste text here",
                height=300,
                key="text_area_input",
                on_change=text_area_on_change,
                value=st.session_state.current_text,  # Use current_text to prefill
            )
            text = st.session_state.current_text  # Use the centrally stored text

        # --- Match Processing and Display ---
        if text:
            active_pattern = st.session_state.pattern_compiled

            if not st.session_state.active_keywords_list or not active_pattern:
                if (
                    not keywords_error_msg
                ):  # Don't show this if there's already a keyword error
                    st.warning(
                        "No active keywords configured or pattern failed. Cannot find matches.",
                        icon="🧩",
                    )
            else:
                st.subheader("💡 2. Keyword Matches & Replacements")
                st.markdown(
                    """
                <style>
                .highlight-word-box {
                    background-color: #e0e0e0; /* Light grey background */
                    border: 1px solid #cccccc; /* Slightly darker grey border */
                    border-radius: 5px;       /* Rounded corners */
                    padding: 0.1em 0.3em;     /* Small padding (vertical, horizontal) */
                    display: inline-block;    /* Allows padding and border */
                    font-weight: bold;        /* Keep the bold emphasis */
                    color: #333;              /* Darker text color for contrast */
                    line-height: 1.2;         /* Adjust line height slightly if needed */
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )
                if not matches:
                    st.info("✅ No keywords found in the provided text.")
                else:
                    st.caption(
                        f"Found {len(matches)} keyword instances. Manage replacements below."
                    )

                    # --- Display Headers ---
                    cols_h = st.columns((6, 1, 3))
                    headers = ["Match Context", "AI Action", "Replacement"]
                    for idx, h in enumerate(headers):
                        cols_h[idx].markdown(f"**{h}**")

                    # --- Loop Through Matches ---
                    for i, m in enumerate(matches):
                        orig = m["match_text"]
                        match_key_prefix = (
                            f"match_{i}"  # Unique prefix for widgets of this match
                        )
                        manual_input_key = f"{match_key_prefix}_manual_input"
                        apply_ai_key = f"{match_key_prefix}_apply_ai"

                        # --- Determine Current State for this Match ---
                        current_replacement_info = st.session_state.replacements.get(i)
                        is_pending = i in st.session_state.tasks_pending
                        is_error = (
                            isinstance(current_replacement_info, dict)
                            and "error" in current_replacement_info
                        )
                        # Successful AI suggestion is a string stored in replacements
                        ai_suggestion = (
                            current_replacement_info
                            if isinstance(current_replacement_info, str)
                            and not st.session_state.apply_selection.get(i, False)
                            else None
                        )
                        # Manual replacement is also a string, check if selection is True and it's not an error
                        manual_replacement_applied = (
                            st.session_state.apply_selection.get(i, False)
                            and isinstance(current_replacement_info, str)
                        )

                        # --- Display Columns for Match ---
                        match_col, button_col, status_col = st.columns((6, 1, 3))

                        with match_col:  # Display context
                            st.markdown(f"**{i + 1}.** `{orig}`")
                            match_start_in_excerpt = m["excerpt"].find(orig)
                            if match_start_in_excerpt != -1:
                                highlighted_word_html = (
                                    f'<span class="highlight-word-box">{orig}</span>'
                                )
                                snippet = (
                                    m["excerpt"][:match_start_in_excerpt]
                                    + highlighted_word_html
                                    + m["excerpt"][match_start_in_excerpt + len(orig) :]
                                )
                            else:
                                snippet = m["excerpt"]  # Fallback
                            snippet_display = snippet.replace("\n", " ")
                            st.markdown(
                                f"> 💬 ...{snippet_display}...", unsafe_allow_html=True
                            )

                        with button_col:  # AI Suggest/Retry Button
                            ai_button_active = not is_pending
                            ai_button_label = "Retry AI" if is_error else "Suggest AI"
                            ai_button_key = f"{match_key_prefix}_suggest_ai"
                            if st.button(
                                ai_button_label,
                                key=ai_button_key,
                                help="Request AI suggestion",
                                disabled=not ai_button_active,
                            ):
                                if not active_mistral_key:
                                    st.session_state.show_key_error_on_suggest = True
                                    log(
                                        "Suggest AI clicked but API Key is missing.",
                                        "warning",
                                    )
                                    # Rerun needed to show the error message at the top
                                    st.rerun()
                                else:
                                    st.session_state.show_key_error_on_suggest = False
                                    log(
                                        f"Requesting AI suggestion thread for match {i} ('{orig}')",
                                        "info",
                                    )
                                    start_suggestion_thread(
                                        match_index=i,
                                        result_q=st.session_state.result_queue,
                                        session_id=st.session_state.session_id,
                                        full_text=text,
                                        excerpt=orig,
                                        blocked_keywords=st.session_state.active_keywords_list
                                        or [],
                                        api_key=active_mistral_key,
                                    )
                                    # Update state immediately
                                    st.session_state.tasks_pending.add(i)
                                    st.session_state.replacements.pop(
                                        i, None
                                    )  # Clear previous result/error
                                    st.session_state.apply_selection[i] = (
                                        False  # Deselect previous
                                    )
                                    # Clear manual input widget state for this match if AI is requested
                                    st.session_state[manual_input_key] = ""
                                    log(
                                        f"Cleared manual input widget {manual_input_key}",
                                        "debug",
                                    )
                                    # No explicit rerun here - polling loop will handle updates

                        with status_col:  # Replacement Status / Input Area
                            manual_widget_disabled = (
                                is_pending  # Disable manual input while AI runs
                            )
                            checkbox_widget_disabled = (
                                is_pending or is_error
                            )  # Disable checkbox if pending or error

                            if is_pending:
                                st.info("⏳ Processing AI...", icon="⏳")
                            elif is_error:
                                st.error(
                                    f"🔥 AI Failed: {current_replacement_info.get('error', 'Unknown')}",
                                    icon="🔥",
                                )
                                # Ensure selection is False on error
                                st.session_state.apply_selection[i] = False
                            elif ai_suggestion:  # Successful AI suggestion exists
                                st.success(f"AI: → **{ai_suggestion}**")
                                # Render checkbox to apply AI suggestion
                                st.checkbox(
                                    "Use this AI Suggestion",
                                    key=apply_ai_key,
                                    value=st.session_state.apply_selection.get(
                                        i, False
                                    ),  # Reflect current selection
                                    on_change=handle_apply_ai_change,
                                    args=(i,),
                                    disabled=checkbox_widget_disabled,
                                )
                            # If manual replacement is currently selected, indicate it
                            elif manual_replacement_applied:
                                st.info(
                                    f"Using Manual: → **{current_replacement_info}**"
                                )

                            # Always show manual input, disable if AI pending
                            st.text_input(
                                "Manual Replacement:",
                                key=manual_input_key,
                                value=st.session_state.get(
                                    manual_input_key, ""
                                ),  # Use widget state
                                on_change=handle_manual_input_change,
                                args=(i,),
                                placeholder="Type replacement here...",
                                label_visibility="collapsed",
                                disabled=manual_widget_disabled,
                                help="Type here to set a manual replacement. It will be selected automatically.",
                            )

                    st.divider()

                    # --- Apply Replacements Section ---
                    st.subheader("✨ 3. Apply Replacements & Output")

                    # Calculate selected count based on the reliable apply_selection state
                    selected_indices = {
                        idx
                        for idx, selected in st.session_state.apply_selection.items()
                        if selected
                    }
                    valid_replacements_exist = {
                        idx
                        for idx, rep in st.session_state.replacements.items()
                        if isinstance(rep, str)
                    }  # Only count successful string replacements
                    applicable_selection_count = len(
                        selected_indices.intersection(valid_replacements_exist)
                    )

                    log(f"Selected indices: {selected_indices}", "debug")
                    log(
                        f"Valid replacement indices: {valid_replacements_exist}",
                        "debug",
                    )
                    log(
                        f"Applicable selection count: {applicable_selection_count}",
                        "debug",
                    )

                    apply_button_disabled = applicable_selection_count == 0
                    if st.button(
                        "Apply Selected Replacements", disabled=apply_button_disabled
                    ):
                        log(
                            f"Applying {applicable_selection_count} selected replacements.",
                            "info",
                        )
                        final_text_parts = []
                        last_processed_index = len(text)

                        # Use the persistent matches list
                        if "matches_for_current_text" not in st.session_state:
                            st.error("Internal error: Matches not found when applying.")
                        else:
                            current_matches = st.session_state.matches_for_current_text
                            sorted_match_indices = sorted(
                                range(len(current_matches)),
                                key=lambda k: current_matches[k]["end"],
                                reverse=True,
                            )

                            for idx in sorted_match_indices:
                                should_apply = st.session_state.apply_selection.get(
                                    idx, False
                                )
                                replacement_value = st.session_state.replacements.get(
                                    idx
                                )
                                m_data = current_matches[idx]
                                start, end = m_data["start"], m_data["end"]

                                if end <= last_processed_index:
                                    # Append text after the current match
                                    final_text_parts.append(
                                        text[end:last_processed_index]
                                    )
                                    # Check if we should apply and have a valid string replacement
                                    if should_apply and isinstance(
                                        replacement_value, str
                                    ):
                                        final_text_parts.append(replacement_value)
                                        log(
                                            f"Applied replacement for match {idx}: '{m_data['match_text']}' -> '{replacement_value}'",
                                            "debug",
                                        )
                                    else:
                                        # Append original text
                                        final_text_parts.append(text[start:end])
                                    last_processed_index = start
                                else:
                                    log(
                                        f"Skipping match {idx} due to potential overlap.",
                                        "warning",
                                    )

                            # Append text before the first match
                            final_text_parts.append(text[:last_processed_index])
                            corrected_text = "".join(reversed(final_text_parts))
                            st.session_state.corrected_text_output = corrected_text
                            log(
                                "Corrected text generated and stored in session state.",
                                "info",
                            )
                            # No rerun needed here, natural rerun will display it

                    # --- Display Corrected Text (if available) ---
                    if st.session_state.corrected_text_output is not None:
                        st.markdown("**Corrected Document Preview:**")
                        st.text_area(
                            "Corrected Text",
                            value=st.session_state.corrected_text_output,
                            height=300,
                            key="corrected_text_area_display",
                            disabled=True,
                        )
                        st.download_button(
                            "📥 Download Corrected Text (.txt)",
                            st.session_state.corrected_text_output,
                            "corrected_document.txt",
                            "text/plain",
                            key="download_corrected",
                        )
                    elif not apply_button_disabled:
                        st.caption(
                            "Select replacements using checkboxes or by typing in manual fields, then click 'Apply'."
                        )
                    else:
                        st.caption(
                            "Get/enter replacements and select them to enable the 'Apply' button."
                        )

        else:  # No text input yet
            st.markdown("*(Upload a file or paste text in the area above to begin)*")

    # --- Original Text Display (Right Column) ---
    with col2:
        # --- Log Display ---
        st.subheader("📊 Logs")
        log_buffer = st.session_state.get("log_buffer", [])
        st.text_area(
            "Log Output",
            value="\n".join(log_buffer),
            height=300,
            key="log_display_area",
            disabled=True,
            help="Shows application events and potential errors.",
        )

    # --- Background Task Polling ---
    # This section remains crucial for handling async AI results
    if st.session_state.tasks_pending:
        results_processed_this_run = False
        polling_placeholder = st.empty()
        try:
            while not st.session_state.result_queue.empty():
                match_index, result_data = st.session_state.result_queue.get_nowait()
                if match_index in st.session_state.tasks_pending:
                    log(
                        f"Polling: Received result for pending match {match_index}.",
                        "debug",
                    )
                    st.session_state.replacements[match_index] = (
                        result_data  # Store result string or error dict
                    )
                    st.session_state.tasks_pending.remove(match_index)
                    # Default selection for successful AI result to False (user must check the box)
                    if not isinstance(result_data, dict):
                        st.session_state.apply_selection.setdefault(match_index, False)
                    results_processed_this_run = True
                else:
                    log(
                        f"Polling: Received STALE result for match {match_index}. Discarding.",
                        "warning",
                    )
                st.session_state.result_queue.task_done()
        except queue.Empty:
            pass  # Normal case, no results waiting
        except Exception as e:
            log(f"Polling: Error processing result queue: {e}", "error")

        # --- Rerun Logic Based on Polling ---
        if results_processed_this_run:
            log("Polling: Results processed, triggering immediate rerun.", "debug")
            polling_placeholder.empty()
            st.rerun()
        elif st.session_state.tasks_pending:
            polling_placeholder.info(
                f"⏳ Checking {len(st.session_state.tasks_pending)} AI suggestion(s)...",
                icon="⏳",
            )
            log(
                f"Polling: Tasks still pending ({len(st.session_state.tasks_pending)}), sleeping {POLL_INTERVAL}s.",
                "debug",
            )
            time.sleep(POLL_INTERVAL)
            polling_placeholder.empty()
            st.rerun()
        else:  # Should not happen if tasks_pending is checked first, but good practice
            polling_placeholder.empty()


# --- Script Entry Point ---
if __name__ == "__main__":
    # Ensure state is initialized before running app logic
    init_session_state()
    if runtime.exists():
        app()
    else:
        # Allow running directly with `python your_script.py` for local dev
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(cli.main())
