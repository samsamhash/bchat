# vectorstore.py
"""
VectorStore RAG  FAISS  SentenceTransformers

Main features:

  -Consistent public contract: retrieve(...) returns List[Dict] with id / doc_id / chunk_id / text / score
  -IndexIDMap for explicit control of FAISS IDs
  -Supports IndexFlatIP (cosine similarity via normalization) and optional IndexIVFFlat
  -Atomic persistence (temp file + os.replace)
  -Thread-safe via threading.Lock
  -Useful operations: add, retrieve, retrieve_texts (compatibility), get_by_id, delete_by_id,
  delete_by_doc_id, clear, rebuild_index, save, load, info
  -Backwards compatibility: .texts and .doc_ids properties that replicate the old preview view
  -Options: model_name, model_device, index_type ('flat' | 'ivf'), ivf_nlist
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
        ivf_nlist: int = 100                      #
    ):
        self.lock = Lock()
        self.index_path = index_path
        self.meta_path = meta_path
        self.index_type = index_type.lower()
        self.ivf_nlist = int(ivf_nlist)

        # model embeddings
        print(f"[VectorStore] Loading embedding model {model_name} on device {model_device}")
        self.embedder = SentenceTransformer(model_name, device=model_device)
        self.dim = self.embedder.get_sentence_embedding_dimension()

        # Metadata: id -> {doc_id, chunk_idx, text}
        
        self.id2meta: Dict[int, Dict[str, Any]] = {}
        self.next_id: int = 1

       
        self._create_index_instance()

        
        self.load()

    
    # Index creation & helpers
    # -------------------------
    def _create_index_instance(self):
      
        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
            # index needs to be trained before add()
        else:
           
            index = faiss.IndexFlatIP(self.dim)

        
        if not isinstance(index, faiss.IndexIDMap):
            try:
                self.index = faiss.IndexIDMap(index)
            except Exception:
                
                self.index = faiss.IndexIDMap2(index)
        else:
            self.index = index

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    
    # Persistence
    # -------------------------
    def save(self) -> None:
        """
        Save index FAISS and metadata.
        """
        with self.lock:
            
            tmp_idx = f"{self.index_path}.tmp"
            try:
                faiss.write_index(self.index, tmp_idx)
                os.replace(tmp_idx, self.index_path)
            except Exception as e:
               
                if os.path.exists(tmp_idx):
                    try:
                        os.remove(tmp_idx)
                    except Exception:
                        pass
                raise RuntimeError(f"[VectorStore] Error saving FAISS index: {e}")

            
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
       
        with self.lock:
            meta_exists = os.path.exists(self.meta_path)
            idx_exists = os.path.exists(self.index_path)

            if not meta_exists and not idx_exists:
              
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
       
        if self.index_type != "ivf":
            return
        # (IndexIDMap -> .index)
        base = self.index.index if hasattr(self.index, "index") else self.index
        # IndexIVFFlat
        try:
            if hasattr(base, "is_trained") and not base.is_trained:
                print("[VectorStore] Training IVF index...")
                base.train(vectors)
        except Exception as e:
            print("[VectorStore] Could not train IVF index:", e)
            raise

    def add(self, doc_id: str, chunks: List[str]) -> int:
       
        if not chunks:
            return 0

        with self.lock:
            vectors = self.embedder.encode(chunks, convert_to_numpy=True)
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            vectors = self._normalize(vectors)

           
            self._maybe_train_ivf(vectors)

            ids = np.arange(self.next_id, self.next_id + len(chunks), dtype=np.int64)
            try:
                # add_with_ids 
                self.index.add_with_ids(vectors, ids)
            except Exception as e:
                
                raise RuntimeError(f"[VectorStore] FAISS add failed: {e}")

           
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
      
        with self.lock:
            if int(id_) not in self.id2meta:
                return False
            try:
                ids = np.array([int(id_)], dtype=np.int64)
                
                self.index.remove_ids(ids)
            except Exception as e:
                print("[VectorStore] Error removing id from FAISS:", e)
              
            try:
                del self.id2meta[int(id_)]
            except KeyError:
                pass
            return True

    def delete_by_doc_id(self, doc_id: str) -> int:
        
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
        
        with self.lock:
            self._create_index_instance()
            self.id2meta = {}
            self.next_id = 1

    def rebuild_index(self) -> None:
      
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

           
            self._create_index_instance()
            # if IVF -> train
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
        get top_k results for `query`.
        Return list dicts:
            {"id": int, "doc_id": str, "chunk_id": int, "text": str, "score": float}
        
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
        
        res = self.retrieve(query, top_k=top_k)
        return [r["text"] for r in res]

    def get_by_id(self, id_: int) -> Optional[Dict[str, Any]]:
       
        with self.lock:
            return self.id2meta.get(int(id_))

    # -------------------------
    # Compatibility properties (antiguo código)
    # -------------------------
    @property
    def texts(self) -> List[str]:
        
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
       
        if len(texts) != len(doc_ids):
            raise ValueError("texts and doc_ids must have same length")
        with self.lock:
            self.id2meta = {}
            cur = 1
            for i, txt in enumerate(texts):
                self.id2meta[cur] = {
                    "doc_id": doc_ids[i],
                    "chunk_idx": i,   
                    "text": txt
                }
                cur += 1
            self.next_id = cur
            # rebuild index to populate FAISS with new embeddings
            self.rebuild_index()


