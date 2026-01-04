# chat_ui.py
"""
Gradio Chat UI for the RAG project.

Features:
 - Integrates ChatWithMemory pipeline (retrieval + memory + summarization + English enforcement)
 - Optional training feedback hooks via training_logger (mark for training, thumbs up/down, corrections)
 - Clear UI history, clear persistent memory
 - Memory inspector to view recent memory items
 - Avoids usage of Gradio component methods that cause AttributeError (no .update() calls)

Requirements (project files present in same folder):
 - chat_with_memory.py  -> provides ChatWithMemory
 - training_logger.py   -> optional, provides log_example / get_default_logger
 - llm_client.py, memory_manager.py, vectorstore.py, prompts.py etc.

Run:
    python chat_ui.py
"""
from __future__ import annotations

import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

import gradio as gr

# Ensure project root is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# Import ChatWithMemory pipeline
try:
    from chat_with_memory import ChatWithMemory
except Exception as e:
    raise ImportError(f"Could not import ChatWithMemory from chat_with_memory.py: {e}")

# Optional training logger
try:
    import training_logger  # exposes log_example(...) and get_default_logger()
    _HAS_TRAINING_LOGGER = True
except Exception:
    training_logger = None
    _HAS_TRAINING_LOGGER = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat_ui")

# Singleton pipeline (heavy init only once). Enable training logging only if logger available
try:
    _CHAT_PIPELINE: ChatWithMemory = ChatWithMemory()
    logger.info("ChatWithMemory pipeline initialized.")
except Exception as e:
    logger.exception("Failed to initialize ChatWithMemory pipeline: %s", e)
    raise

# UI parameters
DEFAULT_TOP_K = 5
UI_HISTORY_MAX_TURNS = 60  # max messages kept in the UI


# -----------------------
# Utility helpers
# -----------------------
def _ensure_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    return history if isinstance(history, list) else []


def _tail_history(history: List[Dict[str, str]], max_turns: int) -> List[Dict[str, str]]:
    return history[-max_turns:]


def _build_training_example_from_history(history: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """
    Convert recent history into an example for training_logger.
    Heuristic: include the full conversation and the last assistant reply as 'answer'.
    """
    if not history or len(history) < 2:
        return None
    # Ensure last message is assistant
    last = history[-1]
    if last.get("role") != "assistant":
        return None
    conv = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        if role and content is not None:
            conv.append({"role": role, "content": content})
    example = {
        "conversation": conv,
        "question": None,
        "answer": conv[-1]["content"],
        "language": "en",
        "sources_used": False,
        "timestamp": int(time.time()),
    }
    return example


# -----------------------
# Core handlers
# -----------------------
def ask_fn(history: Optional[List[Dict[str, str]]], user_input: str, top_k: Optional[int]) -> Tuple[List[Dict[str, str]], str]:
    """
    Main handler for sending a message from the UI.
    Returns (updated_history, empty_input_string)
    """
    history = _ensure_history(history)
    user_input = (user_input or "").strip()
    if not user_input:
        return history, ""

    # Append user turn for immediate UI feedback
    history.append({"role": "user", "content": user_input})
    history_for_pipeline = _tail_history(history, UI_HISTORY_MAX_TURNS)

    # ensure top_k int
    try:
        top_k_int = int(top_k) if top_k is not None else DEFAULT_TOP_K
    except Exception:
        top_k_int = DEFAULT_TOP_K

    # Call pipeline
    try:
        answer = _CHAT_PIPELINE.ask(question=user_input, ui_history=history_for_pipeline, top_k=top_k_int)
    except Exception as e:
        logger.exception("Pipeline ask failed: %s", e)
        answer = f"❌ Internal error generating response: {e}"

    # Append assistant
    history.append({"role": "assistant", "content": answer})

    # Truncate UI history
    if len(history) > UI_HISTORY_MAX_TURNS:
        history = history[-UI_HISTORY_MAX_TURNS:]

    return history, ""


def clear_history() -> Tuple[List, str]:
    return [], ""


def clear_memory_and_history() -> Tuple[List[Dict[str, str]], str, str]:
    """
    Clear UI history and persistent ConversationMemory, return updated UI and status message.
    Returns (history, input_reset, status_message)
    """
    status_msg = "Memory cleared."
    try:
        _CHAT_PIPELINE.memory.clear()
        logger.info("ConversationMemory cleared via UI.")
        status_msg = "Memory cleared successfully."
    except Exception:
        logger.exception("Failed to clear ConversationMemory.")
        status_msg = "Failed to clear memory (see logs)."
    return [], "", status_msg


# -----------------------
# Feedback & training helpers
# -----------------------
def mark_use_for_training(history: Optional[List[Dict[str, str]]]) -> str:
    """
    Mark the last assistant response with its preceding conversation as a training example.
    """
    history = _ensure_history(history)
    if len(history) < 2:
        return "No conversation available to log."

    if history[-1].get("role") != "assistant":
        return "Last message is not an assistant response."

    example = _build_training_example_from_history(history)
    if not example:
        return "Could not build training example."

    if not _HAS_TRAINING_LOGGER:
        return "Training logger not available on the server."

    try:
        fingerprint = training_logger.log_example(example, user_confirmed=True, recorded_by="ui_mark")
        if fingerprint:
            return f"Example saved for training (id={fingerprint[:8]})."
        else:
            return "Example appears duplicate or was skipped."
    except Exception as e:
        logger.exception("Failed to log training example: %s", e)
        return f"Failed to log example: {e}"


def thumbs_feedback(history: Optional[List[Dict[str, str]]], positive: bool = True) -> str:
    """
    Quick thumbs up / thumbs down feedback. Stored as lightweight training record.
    """
    history = _ensure_history(history)
    if len(history) < 2:
        return "No conversation to provide feedback for."
    if not _HAS_TRAINING_LOGGER:
        return "Training logger not available on the server."

    example = _build_training_example_from_history(history)
    if not example:
        return "Could not extract example."

    try:
        fingerprint = training_logger.log_example(example, user_confirmed=positive, recorded_by="ui_feedback")
        if fingerprint:
            return "Feedback recorded. Thank you."
        else:
            return "Feedback was duplicate or not recorded."
    except Exception as e:
        logger.exception("Failed to record feedback: %s", e)
        return f"Failed to record feedback: {e}"


def correct_last_answer(history: Optional[List[Dict[str, str]]], correction: str) -> Tuple[List[Dict[str, str]], str]:
    """
    Save user's correction of the last assistant response and update the UI-visible assistant message.
    Returns (updated_history, status_message)
    """
    history = _ensure_history(history)
    correction = (correction or "").strip()
    if not correction:
        return history, "Correction text is empty."
    if len(history) < 2:
        return history, "No conversation to correct."
    if history[-1].get("role") != "assistant":
        return history, "Last message is not an assistant response."

    corrected_history = history.copy()
    corrected_history[-1] = {"role": "assistant", "content": correction}

    example = _build_training_example_from_history(corrected_history)
    if not example:
        return history, "Could not build corrected training example."

    if not _HAS_TRAINING_LOGGER:
        return history, "Training logger not available on the server."

    try:
        fingerprint = training_logger.log_example(example, user_confirmed=True, recorded_by="ui_correction")
        msg = f"Correction saved for training (id={fingerprint[:8]})" if fingerprint else "Correction was duplicate or not saved."
    except Exception as e:
        logger.exception("Failed to save correction: %s", e)
        msg = f"Failed to save correction: {e}"

    # Update UI history to show corrected assistant message
    history[-1] = {"role": "assistant", "content": correction}
    return history, msg


def show_memory(n: int = 10) -> str:
    """
    Return a readable string with recent memory items (most recent last).
    """
    try:
        info = _CHAT_PIPELINE.memory.get_recent(n)
        if not info:
            return "Memory is empty."
        lines = []
        for item in info:
            ts = item.get("ts")
            ts_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "N/A"
            role = item.get("role", "assistant")
            text = item.get("text", "")
            lines.append(f"[{item.get('id')}] {role} @ {ts_readable}:\n{text}\n")
        return "\n".join(lines)
    except Exception as e:
        logger.exception("Failed to read memory: %s", e)
        return f"Failed to read memory: {e}"


# -----------------------
# Build Gradio UI
# -----------------------
def build_ui() -> gr.Blocks:
    initial_status = "Status: Ready — training logger available" if _HAS_TRAINING_LOGGER else "Status: Ready — training logger NOT available"
    with gr.Blocks(title="RAG Chat — Memory & Feedback") as demo:
        gr.Markdown(
            """
            # RAG Chat — Memory & Feedback
            - Retrieval + local LLM + persistent conversational memory (FAISS + embeddings).
            - Provide feedback (👍/👎), mark assistant answers for training, or submit corrections.
            """
        )

        # Chatbot component (allow_tags explicit to match future Gradio versions)
        chatbot = gr.Chatbot(elem_id="chatbot", label="Chat", type="messages", height=560, allow_tags=False)

        with gr.Row():
            txt = gr.Textbox(label="Your message", placeholder="Ask something...", lines=2)
            top_k = gr.Slider(minimum=1, maximum=20, value=DEFAULT_TOP_K, step=1, label="top_k")

        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear conversation")
            clear_mem_btn = gr.Button("Clear memory (persistent)")

        with gr.Row():
            thumbs_up = gr.Button("👍 Use for training")
            thumbs_down = gr.Button("👎 Feedback (down)")
            mark_train = gr.Button("Mark answer for training")
            show_mem_btn = gr.Button("Show memory")

        correction_box = gr.Textbox(label="Correction (edit assistant's last reply)", lines=2, placeholder="Paste corrected assistant reply here...")
        correct_btn = gr.Button("Submit correction")

        status_md = gr.Markdown(initial_status)
        memory_output = gr.Textbox(label="Memory (recent items)", lines=10)
        feedback_output = gr.Textbox(label="Feedback status", lines=2)

        # Bind events - note outputs must match return types
        send_btn.click(fn=ask_fn, inputs=[chatbot, txt, top_k], outputs=[chatbot, txt])
        clear_btn.click(fn=clear_history, inputs=None, outputs=[chatbot, txt])
        # clear memory returns (history, input_reset, status_message) -> map to chatbot, txt, status_md
        clear_mem_btn.click(fn=clear_memory_and_history, inputs=None, outputs=[chatbot, txt, status_md])

        # feedback buttons
        thumbs_up.click(fn=lambda h: mark_use_for_training(h), inputs=[chatbot], outputs=[feedback_output])
        thumbs_down.click(fn=lambda h: thumbs_feedback(h, positive=False), inputs=[chatbot], outputs=[feedback_output])
        mark_train.click(fn=lambda h: thumbs_feedback(h, positive=True), inputs=[chatbot], outputs=[feedback_output])

        # correction flow: returns (updated_history, message)
        correct_btn.click(fn=correct_last_answer, inputs=[chatbot, correction_box], outputs=[chatbot, feedback_output])

        # show memory -> returns string
        show_mem_btn.click(fn=lambda: show_memory(10), inputs=None, outputs=[memory_output])

    return demo


# -----------------------
# Run UI
# -----------------------
if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    # In Colab you may want share=True to get a public link; use according to your security needs
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
