# memory_manager.py
"""
ConversationMemory backend for the RAG project.

Features:

  - Persistent conversational memory backed by a VectorStore (separate index).
  - Stores each turn as an independent chunk in the VectorStore.
  - Maintains local metadata (role, text, timestamp, id) indexed by FAISS id.
  - Semantic retrieval: retrieve_relevant(query, top_k) returns an ordered list
  of dicts with role / text / ts / id / doc_id / score.
  - Safe compaction: summarizes multiple items into a single entry and removes the old ones
  from the index (if requested by the summarizer).
  - Configurable retention (max_items) with LIFO deletion (oldest item is removed).
  - Utility methods: add_user_turn, add_assistant_turn, get_recent, clear, info.
  
"""

import time
import uuid
from threading import Lock
from typing import Callable, List, Dict, Optional, Any

from vector_store import VectorStore  


class ConversationMemory:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_device: str = "cpu",
        index_path: str = "faiss_memory.faiss",
        meta_path: str = "memory_metadata.json",
        max_items: int = 1000,
        doc_id_prefix: str = "mem"
    ):
        """
        Initializes the conversational memory.

          - model_name / device: for the embedder used by the memory’s VectorStore.
          - index_path / meta_path: files for the memory index (separate from the document index).
          - max_items: maximum number of items to keep in memory (automatic pruning).
          - doc_id_prefix: prefix for memory doc_ids (e.g., "mem").
        """
        self.lock = Lock()
        self.vs = VectorStore(model_name=model_name, model_device=model_device,
                              index_path=index_path, meta_path=meta_path)
        # _meta_by_id: id(int) -> {role, text, ts, doc_id}
        self._meta_by_id: Dict[int, Dict[str, Any]] = {}
        # order of insertion (oldest first)
        self._order: List[int] = []
        self.max_items = int(max_items)
        self.doc_id_prefix = doc_id_prefix

        
        self._reconstruct_meta_from_vs()

    # -------------------------
    # Internals
    # -------------------------
    def _now_ts(self) -> float:
        return time.time()

    def _make_doc_id(self, role: str) -> str:
        ts_ms = int(self._now_ts() * 1000)
        short = uuid.uuid4().hex[:8]
        return f"{self.doc_id_prefix}-{role}-{ts_ms}-{short}"

    def _reconstruct_meta_from_vs(self):
        
        try:
            # vs.id2meta may be large; we only reconstruct entries whose doc_id matches prefix
            id2meta = getattr(self.vs, "id2meta", None)
            if not id2meta:
                return
            for id_, meta in sorted(id2meta.items(), key=lambda x: int(x[0]) if isinstance(x[0], int) else int(x[0])):
                try:
                    id_int = int(id_)
                except Exception:
                    continue
                doc_id = meta.get("doc_id", "")
                if isinstance(doc_id, str) and doc_id.startswith(self.doc_id_prefix + "-"):
                    # Try to parse role from doc_id: mem-role-...
                    parts = doc_id.split("-")
                    role = parts[1] if len(parts) > 1 else "assistant"
                    text = meta.get("text", "")
                    # Timestamp unknown: set to current time as best-effort
                    ts = self._now_ts()
                    self._meta_by_id[id_int] = {"role": role, "text": text, "ts": ts, "doc_id": doc_id}
                    self._order.append(id_int)
        except Exception:
            # nunca fallar la inicialización
            return

    # -------------------------
    # Add memory
    # -------------------------
    def add_memory(self, role: str, text: str, doc_id_prefix: Optional[str] = None) -> Optional[int]:
        """
        Add chunk de memoria with 'role' ('user'|'assistant') & text.
        Return id FAISS.
        """
        if not text or not text.strip():
            return None
        if role not in ("user", "assistant"):
            role = "assistant"

        doc_prefix = doc_id_prefix or self.doc_id_prefix
        doc_id = self._make_doc_id(role) if doc_prefix == self.doc_id_prefix else f"{doc_prefix}-{role}-{int(self._now_ts()*1000)}-{uuid.uuid4().hex[:6]}"

       
        try:
            
            added = self.vs.add(doc_id, [text])
            if not added or added <= 0:
                return None
            # after adding, the VectorStore.next_id reflects next free id
            # assigned id is next_id - added ... next_id - 1
            last_assigned_id = int(self.vs.next_id) - 1
            first_assigned_id = last_assigned_id - (added - 1)
            assigned_id = int(first_assigned_id)  # single chunk -> first == last
            ts = int(self._now_ts())
            with self.lock:
                self._meta_by_id[assigned_id] = {"role": role, "text": text, "ts": ts, "doc_id": doc_id}
                self._order.append(assigned_id)
                # enforce retention if necessary
                self._enforce_retention()
            # Persist memory index/metadata
            try:
                if hasattr(self.vs, "save"):
                    self.vs.save()
            except Exception:
                
                pass
            return assigned_id
        except Exception as e:
            
            return None

    def add_user_turn(self, text: str) -> Optional[int]:
        return self.add_memory("user", text)

    def add_assistant_turn(self, text: str) -> Optional[int]:
        return self.add_memory("assistant", text)

    # -------------------------
    # Retrieval
    # -------------------------
    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant memory chunks.
        Returns a list ordered by relevance, with dicts:
          {"id", "role", "text", "ts", "doc_id", "score"}
        """
        try:
            results = self.vs.retrieve(query, top_k=top_k)
        except Exception as e:
            return []

        out = []
        with self.lock:
            for r in results:
                rid = int(r.get("id"))
                meta_local = self._meta_by_id.get(rid)
                if meta_local:
                    out.append({
                        "id": rid,
                        "role": meta_local.get("role"),
                        "text": meta_local.get("text"),
                        "ts": meta_local.get("ts"),
                        "doc_id": meta_local.get("doc_id"),
                        "score": float(r.get("score", 0.0))
                    })
                else:
                    # Fallback: if not present in local meta, use vs metadata
                    out.append({
                        "id": rid,
                        "role": None,
                        "text": r.get("text"),
                        "ts": None,
                        "doc_id": r.get("doc_id"),
                        "score": float(r.get("score", 0.0))
                    })
        return out

    # -------------------------
    # Compact / summarize
    # -------------------------
    def compact(self, summarizer: Callable[[str], str], keep_last_n: int = 20) -> bool:
        """
        Summarizes (compacts) old memories into a single entry:

          - summarizer: callable (text) -> summary
          - keep_last_n: keeps the last N entries unsummarized (the most recent ones)
        Returns True if a summary was created; False if there were not enough items.
        """
        with self.lock:
            if len(self._order) <= keep_last_n:
                return False

           
            to_summarize_ids = self._order[:-keep_last_n]
            if not to_summarize_ids:
                return False

            
            texts = [self._meta_by_id[i]["text"] for i in to_summarize_ids if i in self._meta_by_id]
            if not texts:
                return False

            combined = "\n\n".join([f"{self._meta_by_id[i]['role']}: {self._meta_by_id[i]['text']}" for i in to_summarize_ids if i in self._meta_by_id])

        
        try:
            summary = summarizer(combined).strip()
        except Exception as e:
            return False

        if not summary:
            return False

        
        new_id = self.add_assistant_turn(f"[MEMORY SUMMARY]\n{summary}")
        if new_id is None:
            return False

       
        deleted_count = 0
        with self.lock:
            for old_id in to_summarize_ids:
                try:
                    
                    if hasattr(self.vs, "delete_by_id"):
                        try:
                            self.vs.delete_by_id(old_id)
                        except Exception:
                            pass
                    
                    if old_id in self._meta_by_id:
                        del self._meta_by_id[old_id]
                        deleted_count += 1
                except Exception:
                    continue
          
            self._order = [i for i in self._order if i not in set(to_summarize_ids)]
            
            if new_id not in self._order:
                self._order.append(new_id)
           
            try:
                if hasattr(self.vs, "save"):
                    self.vs.save()
            except Exception:
                pass

        return True

    # -------------------------
    # Retention
    # -------------------------
    def _enforce_retention(self):
        """
        If memory > max_items, delete until = max_items.
        """
        while len(self._order) > self.max_items:
            oldest_id = self._order.pop(0)
            try:
                if hasattr(self.vs, "delete_by_id"):
                    self.vs.delete_by_id(oldest_id)
            except Exception:
                pass
            if oldest_id in self._meta_by_id:
                del self._meta_by_id[oldest_id]
       
        try:
            if hasattr(self.vs, "save"):
                self.vs.save()
        except Exception:
            pass

    # -------------------------
    # Utilities
    # -------------------------
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Return n items, short by time.
         {id, role, text, ts, doc_id}
        """
        with self.lock:
            recent_ids = self._order[-n:]
            
            out = []
            for rid in reversed(recent_ids):
                m = self._meta_by_id.get(rid)
                if m:
                    out.append({"id": rid, "role": m["role"], "text": m["text"], "ts": m["ts"], "doc_id": m["doc_id"]})
            return out

    def clear(self) -> None:
        """
        Limpia por completo la memoria (índice + metadatos).
        """
        with self.lock:
            try:
                self.vs.clear()
            except Exception:
               
                try:
                    self.vs.rebuild_index()
                except Exception:
                    pass
            self._meta_by_id = {}
            self._order = []
            
            try:
                if hasattr(self.vs, "save"):
                    self.vs.save()
            except Exception:
                pass

    def info(self) -> Dict[str, Any]:
        
        with self.lock:
            return {
                "n_memory": len(self._order),
                "max_items": int(self.max_items),
                "vs_info": self.vs.info() if hasattr(self.vs, "info") else {}
            }

    # -------------------------
    # Migration helper
    # -------------------------
    def migrate_from_lists(self, texts: List[str], doc_ids: List[str]) -> None:
       
        if len(texts) != len(doc_ids):
            raise ValueError("texts and doc_ids must have same length for migration")
        with self.lock:
            self.clear()
            for i, (txt, did) in enumerate(zip(texts, doc_ids)):
               
                role = None
                if isinstance(did, str) and did.startswith(self.doc_id_prefix + "-"):
                    parts = did.split("-")
                    role = parts[1] if len(parts) > 1 else None
                role = role or "assistant"
                
                added = self.vs.add(did, [txt])
                if added and added > 0:
                    last_assigned = int(self.vs.next_id) - 1
                    self._meta_by_id[last_assigned] = {"role": role, "text": txt, "ts": int(self._now_ts()), "doc_id": did}
                    self._order.append(last_assigned)
           
            try:
                if hasattr(self.vs, "save"):
                    self.vs.save()
            except Exception:
                pass
           
            self._enforce_retention()

