# memory_manager.py
"""
ConversationMemory backend for the RAG project.

Características:
- Memoria conversacional persistente sobre VectorStore (índice separado).
- Guarda cada turno como un chunk independiente en el VectorStore.
- Mantiene metadatos locales (role, text, timestamp, id) indexados por id FAISS.
- Recuperación semántica: retrieve_relevant(query, top_k) devuelve lista ordenada
  de dicts con role/text/ts/id/doc_id/score.
- Compactación segura: resume varios items en una sola entrada y elimina los antiguos
  del índice (si summarizer lo solicita).
- Retención configurable (max_items) con borrado en LIFO (el más antiguo se elimina).
- Métodos utilitarios: add_user_turn, add_assistant_turn, get_recent, clear, info.
- Diseñado para ser compatible con la VectorStore provista (debe implementar add, retrieve,
  delete_by_id, next_id, get_by_id, save, load, info).
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
        Inicializa la memoria conversacional.
        - model_name/device: para el embedder usado por la VectorStore de la memoria.
        - index_path/meta_path: archivos para el índice de memoria (separados del índice de documentos).
        - max_items: número máximo de items en memoria (prune automático).
        - doc_id_prefix: prefijo para doc_id de memoria (ej: "mem").
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

        # Intentar reconstruir meta local desde vs.id2meta (si existen entradas mem-*)
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
        """
        Al inicializar, intenta poblar _meta_by_id y _order a partir de vs.id2meta
        para entradas whose doc_id begins with doc_id_prefix.
        Esto permite reusar índices de memoria persistidos.
        """
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
        Añade un chunk de memoria con 'role' ('user'|'assistant') y texto.
        Devuelve el id FAISS asignado si se añade correctamente, o None en caso de error.
        """
        if not text or not text.strip():
            return None
        if role not in ("user", "assistant"):
            role = "assistant"

        doc_prefix = doc_id_prefix or self.doc_id_prefix
        doc_id = self._make_doc_id(role) if doc_prefix == self.doc_id_prefix else f"{doc_prefix}-{role}-{int(self._now_ts()*1000)}-{uuid.uuid4().hex[:6]}"

        # Añadimos como un solo chunk
        try:
            # llame a vs.add; vs.add devuelve count (normalmente 1)
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
                # no fallar la operación de escritura de disco
                pass
            return assigned_id
        except Exception as e:
            # bubble up? mejor devolver None para el caller de UI
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
        Recupera los chunks de memoria más relevantes.
        Devuelve lista en orden de relevancia de dicts:
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
        Resume (compacta) las memorias antiguas en una sola entrada:
        - summarizer: callable(text)->summary
        - keep_last_n: conserva las últimas N entradas sin resumir (las más recientes)
        Retorna True si se creó un summary; False si no había suficientes items.
        """
        with self.lock:
            if len(self._order) <= keep_last_n:
                return False

            # seleccionar items a resumir = todos excepto keep_last_n más recientes
            to_summarize_ids = self._order[:-keep_last_n]
            if not to_summarize_ids:
                return False

            # concatenar textos (orden cronológico)
            texts = [self._meta_by_id[i]["text"] for i in to_summarize_ids if i in self._meta_by_id]
            if not texts:
                return False

            combined = "\n\n".join([f"{self._meta_by_id[i]['role']}: {self._meta_by_id[i]['text']}" for i in to_summarize_ids if i in self._meta_by_id])

        # fuera del lock: llamar al summarizer (potencialmente costoso / I/O)
        try:
            summary = summarizer(combined).strip()
        except Exception as e:
            return False

        if not summary:
            return False

        # Añadir summary como turno del assistant y borrar los antiguos
        new_id = self.add_assistant_turn(f"[MEMORY SUMMARY]\n{summary}")
        if new_id is None:
            return False

        # Ahora eliminar items resumidos del índice y metadata
        deleted_count = 0
        with self.lock:
            for old_id in to_summarize_ids:
                try:
                    # eliminar de VectorStore
                    if hasattr(self.vs, "delete_by_id"):
                        try:
                            self.vs.delete_by_id(old_id)
                        except Exception:
                            pass
                    # eliminar de meta local
                    if old_id in self._meta_by_id:
                        del self._meta_by_id[old_id]
                        deleted_count += 1
                except Exception:
                    continue
            # limpiar el order list
            self._order = [i for i in self._order if i not in set(to_summarize_ids)]
            # añadir el nuevo summary id al final (ya añadido en add_assistant_turn)
            # aseguramos que el new_id esté al final de la lista:
            if new_id not in self._order:
                self._order.append(new_id)
            # persistir cambios
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
        Si la memoria supera max_items, elimina los más antiguos hasta cumplir max_items.
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
        # persistir si hubo cambios
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
        Devuelve los últimos n items (más recientes), ordenados del más reciente al más antiguo.
        Cada item: {id, role, text, ts, doc_id}
        """
        with self.lock:
            recent_ids = self._order[-n:]
            # retornar en orden inverso (más reciente primero)
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
                # fallback: rebuild empty index
                try:
                    self.vs.rebuild_index()
                except Exception:
                    pass
            self._meta_by_id = {}
            self._order = []
            # persistir
            try:
                if hasattr(self.vs, "save"):
                    self.vs.save()
            except Exception:
                pass

    def info(self) -> Dict[str, Any]:
        """
        Información de estado sobre la memoria.
        """
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
        """
        Migra desde la antigua representación de listas paralelas (texts/doc_ids).
        Reconstruye metadata local y llena el índice recalculando embeddings.
        """
        if len(texts) != len(doc_ids):
            raise ValueError("texts and doc_ids must have same length for migration")
        with self.lock:
            self.clear()
            for i, (txt, did) in enumerate(zip(texts, doc_ids)):
                # intentar inferir role desde doc_id si coincide con prefijo
                role = None
                if isinstance(did, str) and did.startswith(self.doc_id_prefix + "-"):
                    parts = did.split("-")
                    role = parts[1] if len(parts) > 1 else None
                role = role or "assistant"
                # añadir directamente en vs; add devuelve count
                added = self.vs.add(did, [txt])
                if added and added > 0:
                    last_assigned = int(self.vs.next_id) - 1
                    self._meta_by_id[last_assigned] = {"role": role, "text": txt, "ts": int(self._now_ts()), "doc_id": did}
                    self._order.append(last_assigned)
            # persistir cambios
            try:
                if hasattr(self.vs, "save"):
                    self.vs.save()
            except Exception:
                pass
            # enforce retention
            self._enforce_retention()
