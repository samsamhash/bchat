# chat_with_memory.py
"""
Chat pipeline with memory and compression for the RAG project.


This module:
 - retrieves document contexts from the global VectorStore
 - retrieves conversational memory from ConversationMemory
 - compresses (summarizes) long contexts using the LLM
 - builds a structured prompt that forces English output
 - calls the local LLM and stores user + assistant turns into memory
"""

import time
from typing import List, Dict, Optional, Any

# Import the local LLM client and the global vector store instance
import llm_client  # provides call_llm_local(...) and vs
from memory_manager import ConversationMemory

# ---------- CONFIGURATION ----------
# adjustable limits and parameters
DEFAULT_TOP_K = 5
MEMORY_TOP_K = 6           # how many memory items to retrieve for summarization/selection
RECENT_TURNS_KEEP = 6      # how many recent turns (user+assistant) to inject directly
MAX_CONTEXT_WORDS = 2000   # approximate maximum words to include in CONTEXT (before summarizing)
MAX_HISTORY_WORDS = 400    # approximate maximum words for history (before summarizing)
SUMMARY_MAX_WORDS = 120    # target words when summarizing with the LLM
SUMMARIZER_PROMPT = (
    "Summarize the following text concisely in English (maximum {max_words} words), "
    "preserving relevant facts and proper names. Respond only with the summary.\n\n### TEXT:\n{text}"
)

# Language guard to force English output at system-level
LANGUAGE_GUARD = (
    "SYSTEM INSTRUCTION:\n"
    "You MUST ALWAYS answer in English, regardless of the language of the question, documents, "
    "or conversation memory. Do not use any other language.\n\n"
)

# ---------- UTILITIES ----------
def _count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.strip().split())

def _truncate_by_words(text: str, max_words: int) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words])

# Summarizer: use the local LLM (call_llm_local) to compress text
def summarize_with_llm(text: str, max_words: int = SUMMARY_MAX_WORDS) -> str:
    """
    Call the local LLM to obtain a concise summary.
    Returns the summary or a fallback (truncated original text) on error.
    """
    if not text or not text.strip():
        return ""

    # Pre-truncate to a reasonable size to avoid giant prompts
    text_for_model = _truncate_by_words(text, max_words * 10)
    prompt = SUMMARIZER_PROMPT.format(max_words=max_words, text=text_for_model)

    try:
        summary = llm_client.call_llm_local(prompt)
        summary = summary.strip()
        # Safety: if summary is longer than requested, truncate it
        if _count_words(summary) > max_words:
            summary = _truncate_by_words(summary, max_words)
        return summary
    except Exception as e:
        # fallback: return a truncated original
        print(f"[summarize_with_llm] Error summarizing with LLM: {e}")
        return _truncate_by_words(text, max_words)

# Build the final RAG prompt that will be sent to the generator LLM
def build_rag_prompt(
    question: str,
    doc_contexts: List[Dict[str, Any]],
    memory_contexts: List[Dict[str, Any]],
    recent_turns: List[Dict[str, str]],
    instructions: Optional[str] = None
) -> str:
    """
    Create a complete prompt that includes:
     - instructions (optional)
     - CONTEXT (retrieved documents with headers)
     - MEMORY (summaries / memory items)
     - DIALOGUE (recent turns, for conversational coherence)
     - QUESTION (user question)
    Each block is clearly separated and labeled to help the LLM cite sources.
    """
    

    parts: List[str] = []
    

    # 1) DOCUMENT CONTEXTS
    parts.append("CONTEXT (retrieved documents):")
    if not doc_contexts:
        parts.append("NONE")
    else:
        for i, dc in enumerate(doc_contexts, start=1):
            # doc_contexts is a list of dicts with keys id/doc_id/chunk_id/text/score
            doc_id = dc.get("doc_id", "unknown")
            chunk_id = dc.get("chunk_id", i-1)
            score = dc.get("score", 0.0)
            text = dc.get("text", "") or ""
            header = f"[{doc_id}#{chunk_id}] (score={score:.3f})"
            parts.append(f"{header}\n{text}\n")

    parts.append("\n---\n")

    # 2) MEMORY (summaries or items)
    parts.append("MEMORY (relevant conversational memory):")
    if not memory_contexts:
        parts.append("NONE")
    else:
        for m in memory_contexts:
            rid = m.get("id")
            role = m.get("role", "assistant")
            ts = m.get("ts")
            score = m.get("score", 0.0)
            text = m.get("text", "")
            header = f"[mem_id:{rid} role:{role} score:{score:.3f}]"
            parts.append(f"{header}\n{text}\n")

    parts.append("\n---\n")

    # 3) RECENT DIALOGUE (for turn coherence)
    parts.append("DIALOGUE (recent turns, most recent last):")
    if not recent_turns:
        parts.append("NONE")
    else:
        for t in recent_turns:
            r = t.get("role", "user")
            c = t.get("content", "")
            parts.append(f"{r.upper()}: {c}")

    parts.append("\n---\n")

    # 4) QUESTION
    parts.append("QUESTION:")
    parts.append(question.strip())
    parts.append("\n\nFINAL ANSWER (in English):")

    return "\n".join(parts)

# Main orchestration: retrieval, memory, compression, prompt building, generation
class ChatWithMemory:
    def __init__(
        self,
        memory: Optional[ConversationMemory] = None,
        doc_top_k: int = DEFAULT_TOP_K,
        memory_top_k: int = MEMORY_TOP_K,
        recent_turns_keep: int = RECENT_TURNS_KEEP
    ):
        # If memory not provided, create a default one (uses separate index paths)
        self.memory = memory or ConversationMemory(index_path="faiss_memory.faiss", meta_path="memory_metadata.json")
        self.doc_top_k = int(doc_top_k)
        self.memory_top_k = int(memory_top_k)
        self.recent_turns_keep = int(recent_turns_keep)

    def _retrieve_document_contexts(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents from the global vector store in llm_client.
        Returns a list of dicts with keys id/doc_id/chunk_id/text/score (compatible with suggested VectorStore).
        """
        try:
            vs = getattr(llm_client, "vs", None)
            if vs is None:
                # Fallback: attempt to create one (not recommended)
                print("[ChatWithMemory] Warning: llm_client.vs not found.")
                return []
            res = vs.retrieve(query, top_k=top_k)
            return res or []
        except Exception as e:
            print(f"[ChatWithMemory] Error retrieving docs: {e}")
            return []

    def _get_recent_turns(self, ui_history: List[Dict[str, str]], max_turns: int) -> List[Dict[str, str]]:
        """
        ui_history: list like [{'role':'user'/'assistant','content': '...'}, ...]
        Keep the last max_turns turns (chronological order).
        """
        if not ui_history:
            return []
        return ui_history[-max_turns:]

    def ask(self, question: str, ui_history: Optional[List[Dict[str, str]]] = None, top_k: Optional[int] = None) -> str:
        """
        Flow:
         - Retrieve document contexts (doc_top_k)
         - Retrieve relevant conversation memory (memory_top_k)
         - Take recent turns from ui_history (if provided)
         - Compress (summarize) if CONTEXT or HISTORY exceed limits
         - Build the final prompt and call the local LLM
         - Add user + assistant turns to memory
        """
        top_k = int(top_k) if top_k is not None else self.doc_top_k
        ui_history = ui_history or []

        # 1) Retrieve document contexts
        doc_contexts = self._retrieve_document_contexts(question, top_k=top_k)

        # 2) Retrieve memory contexts
        memory_contexts = self.memory.retrieve_relevant(question, top_k=self.memory_top_k)

        # 3) Recent dialogue
        recent_turns = self._get_recent_turns(ui_history, self.recent_turns_keep)

        # 4) If contexts too long, compress:
        docs_text = " ".join([d.get("text","") for d in doc_contexts])
        mem_text = " ".join([m.get("text","") for m in memory_contexts])
        recent_text = " ".join([t.get("content","") for t in recent_turns])

        total_words = _count_words(docs_text) + _count_words(mem_text) + _count_words(recent_text)
        # If docs exceed MAX_CONTEXT_WORDS, summarize the least relevant half
        if _count_words(docs_text) > MAX_CONTEXT_WORDS:
            try:
                sorted_docs = sorted(doc_contexts, key=lambda x: float(x.get("score", 0.0)))
            except Exception:
                sorted_docs = doc_contexts
            n = max(1, len(sorted_docs)//2)
            to_summarize = sorted_docs[:n]
            rest = sorted_docs[n:]
            combined = "\n\n".join([f"[{d.get('doc_id')}#{d.get('chunk_id')}] {d.get('text','')}" for d in to_summarize])
            summary = summarize_with_llm(combined, max_words=SUMMARY_MAX_WORDS)
            doc_contexts = rest + [{"id": -1, "doc_id": "DOC_SUMMARY", "chunk_id": 0, "text": f"[SUMMARY of {n} docs]: {summary}", "score": max(d.get("score",0.0) for d in to_summarize)}]

        # If memory text is long, summarize into a single memory summary
        if _count_words(mem_text) > MAX_HISTORY_WORDS:
            combined = "\n\n".join([f"{m.get('role')}: {m.get('text','')}" for m in memory_contexts])
            summary = summarize_with_llm(combined, max_words=SUMMARY_MAX_WORDS)
            memory_contexts = [{"id": -1, "role": "assistant", "text": f"[MEMORY SUMMARY] {summary}", "ts": int(time.time()), "doc_id":"MEM_SUM", "score": 1.0}]

        # If recent dialogue is long, compress it as well
        if _count_words(recent_text) > MAX_HISTORY_WORDS:
            combined = "\n\n".join([f"{t.get('role')}: {t.get('content')}" for t in recent_turns])
            summary = summarize_with_llm(combined, max_words=max(int(SUMMARY_MAX_WORDS/2), 60))
            recent_turns = [{"role":"assistant","content":f"[DIALOGUE SUMMARY] {summary}"}]

        # 5) Build final prompt and prepend the language guard
        rag_prompt = build_rag_prompt(question=question, doc_contexts=doc_contexts, memory_contexts=memory_contexts, recent_turns=recent_turns)
        prompt =  rag_prompt

        # 6) Add user turn to memory (optimistic; could be added after successful generation)
        try:
            self.memory.add_user_turn(question)
        except Exception:
            pass

        # 7) Call local LLM
        try:
            answer = llm_client.call_llm_local(prompt)
            answer = answer.strip()
        except Exception as e:
            answer = f"❌ Generation error: {e}"
            print(f"[ChatWithMemory] LLM error: {e}")

        # 8) Save assistant turn in memory
        try:
            self.memory.add_assistant_turn(answer)
        except Exception:
            pass

        # 9) Return
        return answer

# ---------- Helpers for integration with Gradio / chat UI ----------
def ask_fn_for_ui(history: List[Dict[str,str]], user_input: str, top_k: int = DEFAULT_TOP_K):
    """
    
    - history is the list of messages [{'role':'user'/'assistant','content':...}, ...]
    - returns: (modified_history, empty_input_string)
    """
    if not user_input or not user_input.strip():
        return history, ""

    # Append the user message to the UI history (so it's visible immediately)
    history = history or []
    history.append({"role":"user", "content": user_input})

    # Create or reuse a global pipeline instance
    global _GLOBAL_CHAT_PIPELINE
    try:
        _GLOBAL_CHAT_PIPELINE
    except NameError:
        _GLOBAL_CHAT_PIPELINE = ChatWithMemory()

    pipeline = _GLOBAL_CHAT_PIPELINE

    # Main call
    try:
        answer = pipeline.ask(question=user_input, ui_history=history, top_k=top_k)
    except Exception as e:
        answer = f"❌ Internal error: {e}"

    # Append assistant reply to UI history
    history.append({"role":"assistant", "content": answer})
    return history, ""

# If run for quick debugging
if __name__ == "__main__":
    print("Quick ChatWithMemory demo (exit with Ctrl+C).")
    chat = ChatWithMemory()
    ui_hist = []
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        resp = chat.ask(q, ui_history=ui_hist, top_k=5)
        print("Assistant:", resp)
        ui_hist.append({"role":"user","content":q})
        ui_hist.append({"role":"assistant","content":resp})

