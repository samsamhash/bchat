# training_logger.py
"""
training_logger.py

Lightweight, safe, production-minded logger to collect user interaction examples
for later dataset construction / fine-tuning.

Features:
- Append examples to a JSONL file with atomic, thread-safe writes.
- Basic validation and sanitization (PII redaction for emails, phones, credit cards, URLs).
- Duplicate detection via SHA256 fingerprinting (persisted).
- Optional user-feedback field and metadata (timestamp, recorded_by).
- Utilities to list/load/rotate/export examples.
- Configurable storage directory and file names.

Usage:
    from training_logger import TrainingLogger, DEFAULT_DATA_DIR

    logger = TrainingLogger()  # uses defaults: data/training_logs/raw_interactions.jsonl
    example = {
        "conversation": [
            {"role": "user", "content": "My name is Alice and I live in Madrid."},
            {"role": "assistant", "content": "Nice to meet you, Alice."}
        ],
        "question": "Where does the user live?",
        "answer": "The user lives in Madrid.",
        "language": "en",
        "sources_used": True
    }
    logger.log_example(example, user_confirmed=True)

Notes on privacy/compliance:
- This module performs lightweight redaction but **is not** a substitute for a legal
  privacy review. If user data is sensitive (PII, health, financial), you must
  implement stronger anonymization and secure storage, and obtain legal consent.
"""

from __future__ import annotations

import os
import io
import re
import json
import time
import hashlib
import tempfile
import datetime
import threading
from typing import Dict, Any, Optional, List

# -------------------------
# Default configuration
# -------------------------
DEFAULT_DATA_DIR = "data/training_logs"
DEFAULT_RAW_FILENAME = "raw_interactions.jsonl"
DEFAULT_SEEN_FILENAME = "seen_hashes.json"
DEFAULT_CURATED_FILENAME = "curated_interactions.jsonl"

# -------------------------
# Simple redaction regexes
# -------------------------
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?:(?:\+?\d{1,3}[\s\-\.])?(?:\(?\d{2,4}\)?[\s\-\.])?\d{3,4}[\s\-\.]\d{3,4})"
)
_CREDIT_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_URL_RE = re.compile(
    r"\b(?:https?://|www\.)[^\s/$.?#].[^\s]*\b", flags=re.IGNORECASE
)

# -------------------------
# Helper utilities
# -------------------------
def _now_ts() -> int:
    return int(time.time())

def _iso_ts(ts: Optional[int] = None) -> str:
    if ts is None:
        ts = _now_ts()
    return datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z"

def _sha256_of_json(obj: Any) -> str:
    # Produce a deterministic representation
    j = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

def _atomic_append_line(path: str, line: str) -> None:
    """
    Append a single line atomically (safe-ish) by writing to a temp file in the same dir
    and then using os.rename to append heavy-IO safe? The safe approach is open+write+fsync.
    We'll open in append mode and fsync for durability; use a lock around calls in-process.
    """
    # Open file descriptor and append then fsync
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except (AttributeError, OSError):
            # os.fsync may not be available on some platforms, ignore then
            pass

def _redact_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    s = text
    s = _EMAIL_RE.sub("[REDACTED_EMAIL]", s)
    s = _PHONE_RE.sub("[REDACTED_PHONE]", s)
    s = _CREDIT_RE.sub("[REDACTED_TOKEN]", s)
    s = _URL_RE.sub("[REDACTED_URL]", s)
    return s

def _redact_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Walk the example structure and redact likely PII in string fields.
    This is a best-effort sanitizer and must be augmented for strict privacy needs.
    """
    def walk(v):
        if isinstance(v, str):
            return _redact_text(v)
        if isinstance(v, dict):
            return {k: walk(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [walk(i) for i in v]
        return v

    return walk(example)

# -------------------------
# TrainingLogger class
# -------------------------
class TrainingLogger:
    """
    Thread-safe logger for collecting training examples.

    Parameters:
        data_dir: base directory where logs and index files are stored
        raw_filename: name of the raw jsonl file where examples are appended
        seen_filename: file storing set/list of known fingerprints (to avoid duplicates)
        curated_filename: optional file for curated examples (not used by default)
        redact: whether to apply PII redaction by default
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        raw_filename: str = DEFAULT_RAW_FILENAME,
        seen_filename: str = DEFAULT_SEEN_FILENAME,
        curated_filename: str = DEFAULT_CURATED_FILENAME,
        redact: bool = True,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.raw_path = os.path.join(self.data_dir, raw_filename)
        self.seen_path = os.path.join(self.data_dir, seen_filename)
        self.curated_path = os.path.join(self.data_dir, curated_filename)
        self.redact = bool(redact)

        # internal synchronization
        self._lock = threading.Lock()

        # ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)

        # load seen hashes
        self._seen_hashes = self._load_seen_hashes()

    # -------------------------
    # Internal persistence helpers
    # -------------------------
    def _load_seen_hashes(self) -> Dict[str, int]:
        if not os.path.exists(self.seen_path):
            return {}
        try:
            with open(self.seen_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Expect dict hash -> timestamp
            if isinstance(data, dict):
                return {str(k): int(v) for k, v in data.items()}
        except Exception:
            # If file corrupted, ignore and start fresh
            return {}
        return {}

    def _save_seen_hashes(self) -> None:
        tmp = self.seen_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._seen_hashes, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, self.seen_path)

    # -------------------------
    # Validation & normalization
    # -------------------------
    def _validate_example(self, example: Dict[str, Any]) -> bool:
        """
        Basic checks for example structure. Return True if acceptable.
        Required keys: 'conversation' (list) OR ('question' and 'answer').
        """
        if not isinstance(example, dict):
            return False
        if "conversation" in example and isinstance(example["conversation"], list) and len(example["conversation"]) >= 1:
            return True
        if "question" in example and "answer" in example:
            return True
        return False

    # -------------------------
    # Public API
    # -------------------------
    def log_example(
        self,
        example: Dict[str, Any],
        *,
        user_confirmed: Optional[bool] = None,
        recorded_by: Optional[str] = None,
        redact: Optional[bool] = None,
        overwrite_if_duplicate: bool = False,
    ) -> Optional[str]:
        """
        Append an example to the raw log if valid and not a duplicate.

        Returns:
            fingerprint (hex str) if appended, None if skipped (invalid or duplicate).
        """
        if redact is None:
            redact = self.redact

        if not self._validate_example(example):
            raise ValueError("Invalid example format. Require 'conversation' list or 'question'+'answer'.")

        # copy and sanitize if requested
        ex_copy = json.loads(json.dumps(example))  # deep copy
        if redact:
            ex_copy = _redact_example(ex_copy)

        # metadata
        metadata = {
            "recorded_at": _iso_ts(),
            "recorded_by": recorded_by or "system",
            "user_confirmed": bool(user_confirmed) if user_confirmed is not None else None,
        }

        # compute fingerprint on sanitized content + metadata keys that describe the example
        fingerprint_obj = {
            "example": ex_copy,
        }
        fingerprint = _sha256_of_json(fingerprint_obj)

        with self._lock:
            # duplicate detection
            if fingerprint in self._seen_hashes and not overwrite_if_duplicate:
                return None

            # compose record
            record = {
                "id": fingerprint,
                "metadata": metadata,
                "example": ex_copy,
            }

            # append to raw jsonl
            line = json.dumps(record, ensure_ascii=False)
            try:
                _atomic_append_line(self.raw_path, line)
            except Exception as e:
                # on write failure, do not mark as seen
                raise RuntimeError(f"Failed to write raw log: {e}")

            # update seen and persist seen file
            self._seen_hashes[fingerprint] = _now_ts()
            try:
                self._save_seen_hashes()
            except Exception:
                # best-effort; not fatal for logging
                pass

        return fingerprint

    def count_raw(self) -> int:
        """
        Count number of lines (examples) in raw log file.
        """
        if not os.path.exists(self.raw_path):
            return 0
        try:
            with open(self.raw_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def load_raw_examples(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load raw examples as list of records. Optionally limit to last N (most recent).
        """
        if not os.path.exists(self.raw_path):
            return []
        records: List[Dict[str, Any]] = []
        try:
            with open(self.raw_path, "r", encoding="utf-8") as f:
                if limit is None:
                    for line in f:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            continue
                else:
                    # read last `limit` lines efficiently
                    # fallback simple approach: load all and tail
                    for line in f:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            continue
                    if len(records) > limit:
                        records = records[-limit:]
        except Exception:
            return []
        return records

    def rotate_raw(self, keep_backup: bool = True) -> str:
        """
        Rotate the raw log file. Returns path to rotated file.
        The current raw file is moved to a timestamped backup and a new empty raw file is created.
        """
        if not os.path.exists(self.raw_path):
            return ""

        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        rotated = os.path.join(self.data_dir, f"raw_interactions.{ts}.jsonl")
        os.replace(self.raw_path, rotated)
        # create an empty new raw file
        open(self.raw_path, "a", encoding="utf-8").close()
        return rotated

    def export_curated(self, output_path: Optional[str] = None, transform_fn=None) -> str:
        """
        Export a curated dataset derived from raw entries.
        transform_fn(record) -> yield one or more output dicts (instruction/response pairs).
        If transform_fn is None, a default transform will extract:
            {"input": concatenated conversation before last assistant reply, "output": last assistant reply}
        Returns the path to the exported file.
        """
        out_path = output_path or os.path.join(self.data_dir, "exported_dataset.jsonl")
        records = self.load_raw_examples()
        with open(out_path, "w", encoding="utf-8") as out_f:
            for rec in records:
                try:
                    if transform_fn:
                        items = transform_fn(rec)
                        for it in items:
                            out_f.write(json.dumps(it, ensure_ascii=False) + "\n")
                    else:
                        # default transform
                        ex = rec.get("example", {})
                        conv = ex.get("conversation")
                        if isinstance(conv, list) and len(conv) >= 2:
                            # find last assistant reply
                            last_assistant = None
                            before = []
                            for turn in conv:
                                if turn.get("role") == "assistant":
                                    last_assistant = turn.get("content", "")
                                else:
                                    before.append(turn.get("content", ""))
                            if last_assistant:
                                out = {"input": " ".join(before).strip(), "output": last_assistant}
                                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        return out_path

    def debug_stats(self) -> Dict[str, Any]:
        """
        Return simple diagnostics about storage and seen index.
        """
        return {
            "data_dir": self.data_dir,
            "raw_path_exists": os.path.exists(self.raw_path),
            "raw_count": self.count_raw(),
            "seen_count": len(self._seen_hashes),
            "seen_path": self.seen_path,
        }

# -------------------------
# Module-level singleton & convenience functions
# -------------------------
_default_logger: Optional[TrainingLogger] = None
_default_lock = threading.Lock()

def get_default_logger() -> TrainingLogger:
    global _default_logger
    with _default_lock:
        if _default_logger is None:
            _default_logger = TrainingLogger()
        return _default_logger

def log_example(
    example: Dict[str, Any],
    *,
    user_confirmed: Optional[bool] = None,
    recorded_by: Optional[str] = None,
    redact: Optional[bool] = None,
) -> Optional[str]:
    """
    Convenience wrapper to log using the module's default logger.
    """
    logger = get_default_logger()
    return logger.log_example(example, user_confirmed=user_confirmed, recorded_by=recorded_by, redact=redact)

# -------------------------
# CLI utilities (basic)
# -------------------------
def _cli_print_help():
    print("training_logger.py - simple CLI")
    print("Commands:")
    print("  inspect                 - print debug stats")
    print("  tail N                  - show last N raw examples (json)")
    print("  export [out.jsonl]      - export curated dataset to path (default data/exported_dataset.jsonl)")
    print("  rotate                  - rotate raw file to timestamped backup")

def _cli_main(argv: List[str]):
    logger_obj = get_default_logger()
    if not argv or argv[0] in ("-h", "--help"):
        _cli_print_help()
        return
    cmd = argv[0]
    if cmd == "inspect":
        print(json.dumps(logger_obj.debug_stats(), indent=2, ensure_ascii=False))
    elif cmd == "tail":
        n = int(argv[1]) if len(argv) > 1 else 10
        recs = logger_obj.load_raw_examples(limit=n)
        for r in recs:
            print(json.dumps(r, ensure_ascii=False))
    elif cmd == "export":
        out = argv[1] if len(argv) > 1 else None
        path = logger_obj.export_curated(out)
        print("Exported to", path)
    elif cmd == "rotate":
        path = logger_obj.rotate_raw()
        print("Rotated raw log to", path)
    else:
        print("Unknown command:", cmd)
        _cli_print_help()

if __name__ == "__main__":
    import sys
    _cli_main(sys.argv[1:])
