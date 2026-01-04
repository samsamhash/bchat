# vectorstore.py
"""
VectorStore robusto para RAG — FAISS + SentenceTransformers

Características principales:
- Contrato público consistente: retrieve(...) devuelve List[Dict] con id/doc_id/chunk_id/text/score
- IndexIDMap para control explícito de IDs FAISS
- Soporta IndexFlatIP (coseno mediante normalización) y opcional IndexIVFFlat
- Persistencia atómica (tmp file + os.replace)
- Thread-safe mediante threading.Lock
- Operaciones útiles: add, retrieve, retrieve_texts (compatibilidad), get_by_id, delete_by_id,
  delete_by_doc_id, clear, rebuild_index, save, load, info
- Backwards compatibility: propiedades .texts y .doc_ids que replican la vista previa antigua
- Opciones: model_name, model_device, index_type ('flat'|'ivf'), ivf_nlist

Diseñado para integrarse con el proyecto RAG que proporcionaste:
- Ajusta rutas index_path/meta_path según tu entorno (Colab, servidor).
- Requiere: faiss, sentence-transformers, numpy
"""

import os
import json
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from threading import Lock
from typing import List, Dict, Optional, Any


class VectorStore:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_device: str = "cpu",                # "cpu" or "cuda"
        index_path: str = "faiss_index.faiss",
        meta_path: str = "metadata.json",
        index_type: str = "flat",                 # "flat" (IndexFlatIP) or "ivf" (IndexIVFFlat)
        ivf_nlist: int = 100                      # solo para index_type="ivf"
    ):
        self.lock = Lock()
        self.index_path = index_path
        self.meta_path = meta_path
        self.index_type = index_type.lower()
        self.ivf_nlist = int(ivf_nlist)

        # Cargar modelo de embeddings
        print(f"[VectorStore] Loading embedding model {model_name} on device {model_device}")
        self.embedder = SentenceTransformer(model_name, device=model_device)
        self.dim = self.embedder.get_sentence_embedding_dimension()

        # Metadata: id -> {doc_id, chunk_idx, text}
        # Usamos int keys in memory, but JSON stores str keys.
        self.id2meta: Dict[int, Dict[str, Any]] = {}
        self.next_id: int = 1

        # Crear índice FAISS (IndexIDMap sobre un índice base)
        self._create_index_instance()

        # Intentar cargar estado persistido si existe
        self.load()

    # -------------------------
    # Index creation & helpers
    # -------------------------
    def _create_index_instance(self):
        """
        Crea la instancia FAISS base y la envuelve en IndexIDMap.
        Usamos IndexFlatIP + normalización de vectores para similitud coseno.
        """
        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
            # index needs to be trained before add()
        else:
            # flat IP (inner product) + normalización == coseno
            index = faiss.IndexFlatIP(self.dim)

        # Usar IDMap para poder controlar IDs explícitos
        if not isinstance(index, faiss.IndexIDMap):
            try:
                self.index = faiss.IndexIDMap(index)
            except Exception:
                # En algunas versiones, IndexIDMap requiere pasar un índice base
                self.index = faiss.IndexIDMap2(index)
        else:
            self.index = index

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normaliza filas a longitud 1 para usar IP como coseno."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    # -------------------------
    # Persistence (atomic)
    # -------------------------
    def save(self) -> None:
        """
        Guarda el índice FAISS y la metadata de forma atómica.
        """
        with self.lock:
            # Guardar índice FAISS a un .tmp y mover
            tmp_idx = f"{self.index_path}.tmp"
            try:
                faiss.write_index(self.index, tmp_idx)
                os.replace(tmp_idx, self.index_path)
            except Exception as e:
                # limpiar tmp si existe
                if os.path.exists(tmp_idx):
                    try:
                        os.remove(tmp_idx)
                    except Exception:
                        pass
                raise RuntimeError(f"[VectorStore] Error saving FAISS index: {e}")

            # Guardar metadata (id2meta y next_id) de forma atómica
            tmp_meta = f"{self.meta_path}.tmp"
            meta = {
                "id2meta": {str(k): v for k, v in self.id2meta.items()},
                "next_id": int(self.next_id),
                "index_type": self.index_type,
                "ivf_nlist": self.ivf_nlist,
                "dim": int(self.dim),
            }
            try:
                with open(tmp_meta, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False)
                os.replace(tmp_meta, self.meta_path)
            except Exception as e:
                if os.path.exists(tmp_meta):
                    try:
                        os.remove(tmp_meta)
                    except Exception:
                        pass
                raise RuntimeError(f"[VectorStore] Error saving metadata: {e}")

    def load(self) -> bool:
        """
        Carga índice y metadata si existen.
        Si hay inconsistencia entre index.ntotal y len(id2meta), intenta reconstruir el índice
        re-computando embeddings desde id2meta (si es posible).
        """
        with self.lock:
            meta_exists = os.path.exists(self.meta_path)
            idx_exists = os.path.exists(self.index_path)

            if not meta_exists and not idx_exists:
                # Nada que cargar
                return False

            if meta_exists:
                try:
                    with open(self.meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    id2meta = {int(k): v for k, v in meta.get("id2meta", {}).items()}
                    self.id2meta = id2meta
                    self.next_id = int(meta.get("next_id", max(self.id2meta.keys(), default=0) + 1))
                    self.index_type = meta.get("index_type", self.index_type)
                    self.ivf_nlist = int(meta.get("ivf_nlist", self.ivf_nlist))
                except Exception as e:
                    print("[VectorStore] Could not load metadata:", e)
                    # keep defaults
            if idx_exists:
                try:
                    loaded_index = faiss.read_index(self.index_path)
                    # If loaded index is not IDMap, wrap or rebuild
                    if isinstance(loaded_index, faiss.IndexIDMap) or isinstance(loaded_index, faiss.IndexIDMap2):
                        self.index = loaded_index
                    else:
                        # Wrap into IDMap — but index might not have ids; rebuild safer
                        try:
                            self.index = faiss.IndexIDMap(loaded_index)
                        except Exception:
                            self._create_index_instance()  # fallback
                except Exception as e:
                    print("[VectorStore] Could not read FAISS index:", e)
                    self._create_index_instance()
            else:
                # index file missing — create fresh
                self._create_index_instance()

            # Consistencia: si ntotal != len(id2meta) => intentar rebuild desde id2meta si tenemos textos
            try:
                ntotal = int(self.index.ntotal)
            except Exception:
                ntotal = 0
            if len(self.id2meta) == 0 and ntotal == 0:
                return True

            if ntotal != len(self.id2meta):
                print("[VectorStore] Index / metadata size mismatch. Attempting to rebuild index from metadata...")
                try:
                    self.rebuild_index()
                except Exception as e:
                    print("[VectorStore] Rebuild failed:", e)
                    # Keep existing index but warn
                    return False
            return True

    # -------------------------
    # Add / Delete / Rebuild
    # -------------------------
    def _maybe_train_ivf(self, vectors: np.ndarray):
        """
        Si el índice es IVF y no está entrenado, lo entrena usando `vectors`.
        """
        if self.index_type != "ivf":
            return
        # acceder al índice base (IndexIDMap -> .index)
        base = self.index.index if hasattr(self.index, "index") else self.index
        # IndexIVFFlat tiene is_trained en algunas versiones
        try:
            if hasattr(base, "is_trained") and not base.is_trained:
                print("[VectorStore] Training IVF index...")
                base.train(vectors)
        except Exception as e:
            print("[VectorStore] Could not train IVF index:", e)
            raise

    def add(self, doc_id: str, chunks: List[str]) -> int:
        """
        Añade una lista de chunks (strings) asociada a doc_id.
        Devuelve el número de chunks añadidos.
        """
        if not chunks:
            return 0

        with self.lock:
            vectors = self.embedder.encode(chunks, convert_to_numpy=True)
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            vectors = self._normalize(vectors)

            # Entrenar IVF si aplica y aún no entrenado
            self._maybe_train_ivf(vectors)

            ids = np.arange(self.next_id, self.next_id + len(chunks), dtype=np.int64)
            try:
                # add_with_ids espera shape (n, dim) y array ids dtype=int64
                self.index.add_with_ids(vectors, ids)
            except Exception as e:
                # si falla, intentar añadir sin ids (no recomendado)
                raise RuntimeError(f"[VectorStore] FAISS add failed: {e}")

            # Guardar metadata en memoria
            for offset, chunk in enumerate(chunks):
                cur_id = int(self.next_id + offset)
                self.id2meta[cur_id] = {
                    "doc_id": doc_id,
                    "chunk_idx": offset,
                    "text": chunk
                }

            self.next_id += len(chunks)
            return len(chunks)

    def delete_by_id(self, id_: int) -> bool:
        """
        Elimina un vector por su ID FAISS (si existe).
        Devuelve True si se eliminó, False si no existía.
        """
        with self.lock:
            if int(id_) not in self.id2meta:
                return False
            try:
                ids = np.array([int(id_)], dtype=np.int64)
                # IndexIDMap soporta remove_ids
                self.index.remove_ids(ids)
            except Exception as e:
                print("[VectorStore] Error removing id from FAISS:", e)
                # continuar y eliminar metadata igualmente
            try:
                del self.id2meta[int(id_)]
            except KeyError:
                pass
            return True

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Elimina todos los vectores asociados a un doc_id.
        Devuelve el número de items eliminados.
        """
        with self.lock:
            to_delete = [i for i, m in self.id2meta.items() if m.get("doc_id") == doc_id]
            if not to_delete:
                return 0
            ids = np.array(to_delete, dtype=np.int64)
            try:
                self.index.remove_ids(ids)
            except Exception as e:
                print("[VectorStore] Error removing ids from FAISS:", e)
            count = 0
            for i in to_delete:
                if i in self.id2meta:
                    del self.id2meta[i]
                    count += 1
            return count

    def clear(self) -> None:
        """
        Vacía el índice y las metadata (reinicia).
        """
        with self.lock:
            self._create_index_instance()
            self.id2meta = {}
            self.next_id = 1

    def rebuild_index(self) -> None:
        """
        Reconstruye el índice a partir de self.id2meta re-computando embeddings.
        Útil si index y metadata estaban fuera de sync.
        """
        with self.lock:
            if not self.id2meta:
                self._create_index_instance()
                return

            all_ids = sorted(self.id2meta.keys())
            all_texts = [self.id2meta[i]["text"] for i in all_ids]
            vectors = self.embedder.encode(all_texts, convert_to_numpy=True)
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            vectors = self._normalize(vectors)

            # crear nuevo índice base y añadir con ids
            self._create_index_instance()
            # si IVF -> train
            self._maybe_train_ivf(vectors)

            ids = np.array(all_ids, dtype=np.int64)
            try:
                self.index.add_with_ids(vectors, ids)
            except Exception as e:
                raise RuntimeError(f"[VectorStore] Rebuild failed while adding vectors: {e}")

    # -------------------------
    # Retrieve API
    # -------------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recupera top_k resultados relevantes para `query`.
        Devuelve lista de dicts:
            {"id": int, "doc_id": str, "chunk_id": int, "text": str, "score": float}
        Score es el inner product (aprox coseno si normalizamos).
        """
        with self.lock:
            if len(self.id2meta) == 0 or getattr(self.index, "ntotal", 0) == 0:
                return []

            q_vec = self.embedder.encode([query], convert_to_numpy=True)
            q_vec = np.asarray(q_vec, dtype=np.float32)
            if q_vec.ndim == 1:
                q_vec = q_vec.reshape(1, -1)
            q_vec = self._normalize(q_vec)

            k = min(int(top_k), max(1, int(self.index.ntotal)))
            try:
                distances, indices = self.index.search(q_vec, k)
            except Exception as e:
                raise RuntimeError(f"[VectorStore] FAISS search failed: {e}")

            results = []
            for score, idx in zip(distances[0], indices[0]):
                if int(idx) < 0:
                    continue
                idx = int(idx)
                meta = self.id2meta.get(idx)
                if not meta:
                    # metadata faltante: saltar
                    continue
                results.append({
                    "id": idx,
                    "doc_id": meta.get("doc_id"),
                    "chunk_id": meta.get("chunk_idx"),
                    "text": meta.get("text"),
                    "score": float(score)
                })
            return results

    def retrieve_texts(self, query: str, top_k: int = 5) -> List[str]:
        """
        Compatibilidad: devuelve lista de strings (solo textos).
        """
        res = self.retrieve(query, top_k=top_k)
        return [r["text"] for r in res]

    def get_by_id(self, id_: int) -> Optional[Dict[str, Any]]:
        """
        Devuelve metadata completa para un ID, o None.
        """
        with self.lock:
            return self.id2meta.get(int(id_))

    # -------------------------
    # Compatibility properties (antiguo código)
    # -------------------------
    @property
    def texts(self) -> List[str]:
        """
        Retorna lista de textos en orden de IDs (para compatibilidad con implementaciones anteriores).
        Nota: puede ser costoso si index grande.
        """
        with self.lock:
            return [meta["text"] for _, meta in sorted(self.id2meta.items(), key=lambda x: x[0])]

    @property
    def doc_ids(self) -> List[str]:
        with self.lock:
            return [meta["doc_id"] for _, meta in sorted(self.id2meta.items(), key=lambda x: x[0])]

    # -------------------------
    # Info / util
    # -------------------------
    def info(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "ntotal": int(self.index.ntotal) if hasattr(self.index, "ntotal") else 0,
                "n_items": len(self.id2meta),
                "index_type": self.index_type,
                "dim": int(self.dim),
                "index_path": self.index_path,
                "meta_path": self.meta_path
            }

    # -------------------------
    # Migration helper (converts old texts/doc_ids lists to id2meta)
    # -------------------------
    def migrate_from_lists(self, texts: List[str], doc_ids: List[str]) -> None:
        """
        Utilidad para migrar desde la antigua representación donde
        self.texts y self.doc_ids eran listas paralelas.
        Esto no añade vectores al índice; solo actualiza metadata en memoria.
        Luego llama a rebuild_index() para recalcular vectores y poblar el índice.
        """
        if len(texts) != len(doc_ids):
            raise ValueError("texts and doc_ids must have same length")
        with self.lock:
            self.id2meta = {}
            cur = 1
            for i, txt in enumerate(texts):
                self.id2meta[cur] = {
                    "doc_id": doc_ids[i],
                    "chunk_idx": i,   # no tenemos chunk idx real; asignamos i relativo
                    "text": txt
                }
                cur += 1
            self.next_id = cur
            # rebuild index to populate FAISS with new embeddings
            self.rebuild_index()

