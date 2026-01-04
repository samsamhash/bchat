# dataset_builder.py
"""
Dataset builder for fine-tuning from raw interaction logs.

This script reads raw interaction records produced by `training_logger.py`
(or a compatible JSONL file), applies filtering, normalization and optional
language filtering, and writes out a cleaned dataset suitable for instruction
or chat-style fine-tuning (JSONL).

Features:
- Loads raw examples via training_logger.get_default_logger().load_raw_examples()
  or from an explicit raw jsonl file.
- Optional PII-light redaction (basic).
- Duplicate removal (by example fingerprint).
- Filtering by minimum/maximum length, presence of assistant reply, and user-confirmation.
- Optional language filtering (requires `langdetect` package; if absent, skip filter).
- Produce dataset in two common formats:
    * "instruct": {"input": "<conversation text>", "output": "<assistant reply>"}
    * "chat": {"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}, ...]}
- Save train/validation splits and produce basic dataset stats.

Usage (CLI):
    python dataset_builder.py --out data/datasets/chat_finetune.jsonl --format chat --val_split 0.05

Programmatic usage:
    from dataset_builder import build_dataset_from_logger
    build_dataset_from_logger(...)

Author: Generated for user's RAG project
"""

from __future__ import annotations

import os
import json
import argparse
import hashlib
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Try to import training_logger (expected to exist in the project)
try:
    import training_logger
    from training_logger import get_default_logger
except Exception:
    training_logger = None
    get_default_logger = None

# Optional language detection
try:
    from langdetect import detect as lang_detect
except Exception:
    lang_detect = None

# Simple PII redaction regexes (similar to training_logger)
import re
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s\-\.])?(?:\(?\d{2,4}\)?[\s\-\.])?\d{3,4}[\s\-\.]\d{3,4})")
_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s/$.?#].[^\s]*\b", flags=re.IGNORECASE)

logger = logging.getLogger("dataset_builder")
logging.basicConfig(level=logging.INFO)


# --------------------------
# Utilities
# --------------------------
def _sha256_of_obj(obj: Any) -> str:
    j = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()


def _redact_text(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    s = _EMAIL_RE.sub("[REDACTED_EMAIL]", s)
    s = _PHONE_RE.sub("[REDACTED_PHONE]", s)
    s = _URL_RE.sub("[REDACTED_URL]", s)
    return s


def _redact_example(example: Dict[str, Any]) -> Dict[str, Any]:
    # Walk the common fields and redact strings in conversation text fields
    def walk(v):
        if isinstance(v, str):
            return _redact_text(v)
        if isinstance(v, dict):
            return {k: walk(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [walk(i) for i in v]
        return v

    return walk(example)


# --------------------------
# Raw input loader
# --------------------------
def load_raw_from_logger(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads raw records using training_logger.get_default_logger().load_raw_examples(limit).
    Returns a list of records as returned by training_logger (dicts containing keys id, metadata, example).
    """
    if get_default_logger is None:
        raise RuntimeError("training_logger module not found. Please ensure training_logger.py is available.")
    logger_obj = get_default_logger()
    records = logger_obj.load_raw_examples(limit=limit)
    return records


def load_raw_from_file(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load raw JSONL file where each line is a JSON record (same schema as training_logger records).
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out.append(rec)
            except Exception:
                continue
    return out


# --------------------------
# Example normalization
# --------------------------
def _extract_conversation_and_assistant(example_rec: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    """
    Given a raw record (training_logger style), extract the conversation list and the last assistant reply.
    Returns (conversation_list, last_assistant_text) or (None, None) if not available.
    """
    ex = example_rec.get("example") if isinstance(example_rec, dict) and "example" in example_rec else example_rec
    # ex may be just {'conversation': [...] } or {'question':..., 'answer':...}
    if not isinstance(ex, dict):
        return None, None

    if "conversation" in ex and isinstance(ex["conversation"], list):
        conv = ex["conversation"]
        # Ensure each turn has role/content
        normalized = []
        last_assistant = None
        for turn in conv:
            role = turn.get("role") if isinstance(turn, dict) else None
            content = turn.get("content") if isinstance(turn, dict) else None
            if role and content:
                normalized.append({"role": role, "content": content})
                if role == "assistant":
                    last_assistant = content
        return normalized, last_assistant

    # fallback: question+answer form
    if "question" in ex and "answer" in ex:
        conv = [{"role": "user", "content": ex.get("question", "")}, {"role": "assistant", "content": ex.get("answer", "")}]
        return conv, ex.get("answer", "")

    return None, None


def build_instruct_record(conv: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    Convert a conversation into an instruction-style pair:
        input: concatenation of user turns and assistant previous replies as context (or a cleaned prompt)
        output: last assistant reply
    Heuristic: use all turns except the last assistant reply as input, last assistant reply as output.
    """
    if not conv or len(conv) < 2:
        return None
    # find last assistant reply (last element with role assistant)
    last_assistant_idx = None
    for i in range(len(conv)-1, -1, -1):
        if conv[i].get("role") == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None:
        return None
    # build input as everything before that assistant reply
    input_turns = conv[:last_assistant_idx]
    output_text = conv[last_assistant_idx].get("content", "").strip()
    # skip if no output
    if not output_text:
        return None
    # Input string: format turns into a short dialogue
    parts = []
    for t in input_turns:
        role = t.get("role", "user")
        content = t.get("content", "")
        if role == "user":
            parts.append(f"User: {content}")
        else:
            parts.append(f"Assistant: {content}")
    input_text = "\n".join(parts).strip()
    if not input_text:
        # as fallback, include the previous user message if exists
        if last_assistant_idx >= 1:
            prev_user = conv[last_assistant_idx - 1].get("content", "")
            input_text = f"User: {prev_user}"
        else:
            return None
    return {"input": input_text, "output": output_text}


def build_chat_record(conv: List[Dict[str, str]], system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert conversation to chat-style messages. Optionally prepend a system message.
    """
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for t in conv:
        role = t.get("role")
        content = t.get("content", "")
        # Normalize roles: map 'user'|'assistant' as-is, other roles default to 'user'
        if role not in ("user", "assistant"):
            role = "user"
        msgs.append({"role": role, "content": content})
    return {"messages": msgs}


# --------------------------
# Filtering and building
# --------------------------
def filter_and_transform_records(
    raw_records: Iterable[Dict[str, Any]],
    *,
    redact: bool = True,
    min_input_words: int = 1,
    max_input_words: Optional[int] = None,
    require_last_assistant: bool = True,
    require_user_confirmed: Optional[bool] = None,
    language: Optional[str] = None,
    output_format: str = "chat",
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Process raw_records and produce a list of output records in requested format.
    Parameters:
      - raw_records: iterable of raw records (as returned by training_logger)
      - redact: whether to apply lightweight redaction to text
      - min_input_words, max_input_words: filters on input length (in words)
      - require_last_assistant: only keep examples with an assistant reply
      - require_user_confirmed: if True, only keep records with metadata.user_confirmed True (if available).
      - language: ISO code (e.g., 'en') - requires langdetect package; if absent, language filter is skipped.
      - output_format: 'chat' or 'instruct'
      - system_prompt: optional system message to include in chat records
    Returns: list of normalized records (dicts) ready to be written as JSONL.
    """
    out: List[Dict[str, Any]] = []
    seen_fingerprints = set()

    for rec in raw_records:
        try:
            # optional metadata-based filter
            if require_user_confirmed is not None:
                md = rec.get("metadata", {})
                if md is None:
                    md = {}
                user_conf = md.get("recorded_by")  # not ideal, training_logger stores user_confirmed in metadata
                # training_logger stores metadata.user_confirmed maybe; try to access it
                uc = md.get("user_confirmed") if isinstance(md, dict) else None
                if uc is None:
                    # try top-level example metadata
                    ex = rec.get("example", {})
                    uc = ex.get("user_confirmed") if isinstance(ex, dict) else None
                if require_user_confirmed and not uc:
                    continue

            conv, last_assistant = _extract_conversation_and_assistant(rec)
            if conv is None:
                continue
            if require_last_assistant and not last_assistant:
                continue

            # optional language detection: try to detect language from last assistant or combined user text
            if language is not None and lang_detect is not None:
                probe_text = last_assistant or " ".join([t.get("content", "") for t in conv])
                try:
                    detected = lang_detect(probe_text)
                except Exception:
                    detected = None
                if detected is None or detected != language:
                    continue

            # Redact
            if redact:
                conv_redacted = []
                for t in conv:
                    conv_redacted.append({"role": t["role"], "content": _redact_text(t["content"])})
                conv = conv_redacted

            # Build final record depending on format
            if output_format == "instruct":
                instr = build_instruct_record(conv)
                if instr is None:
                    continue
                # length filters
                input_words = len(instr["input"].split())
                if input_words < min_input_words:
                    continue
                if max_input_words is not None and input_words > max_input_words:
                    continue

                fingerprint = _sha256_of_obj(instr)
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)
                out.append(instr)

            else:  # chat format
                chat_rec = build_chat_record(conv, system_prompt=system_prompt)
                # compute length from concatenated user content
                user_text = " ".join([t["content"] for t in conv if t["role"] == "user"])
                uw = len(user_text.split())
                if uw < min_input_words:
                    continue
                if max_input_words is not None and uw > max_input_words:
                    continue
                fingerprint = _sha256_of_obj(chat_rec)
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)
                out.append(chat_rec)
        except Exception:
            # skip problematic record but keep processing
            continue

    return out


# --------------------------
# Save helpers
# --------------------------
def write_jsonl(records: Iterable[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def split_and_save(records: List[Dict[str, Any]], out_path: str, val_split: float = 0.05, seed: int = 42) -> Tuple[str, Optional[str]]:
    """
    Split records into train / val and write to disk. Returns (train_path, val_path or None)
    """
    import random

    n = len(records)
    if n == 0:
        raise ValueError("No records to save.")
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)
    val_n = int(round(n * val_split)) if val_split and val_split > 0 else 0
    val_idx = set(indices[:val_n])
    train_path = out_path
    base, ext = os.path.splitext(out_path)
    val_path = f"{base}.val{ext}" if val_n > 0 else None

    # write train
    with open(train_path, "w", encoding="utf-8") as tf:
        for i in range(n):
            if i in val_idx:
                continue
            tf.write(json.dumps(records[i], ensure_ascii=False) + "\n")

    if val_n > 0:
        with open(val_path, "w", encoding="utf-8") as vf:
            for i in range(n):
                if i in val_idx:
                    vf.write(json.dumps(records[i], ensure_ascii=False) + "\n")

    return train_path, val_path


# --------------------------
# CLI / high-level entrypoint
# --------------------------
def build_dataset_from_logger(
    out_path: str,
    *,
    raw_limit: Optional[int] = None,
    redact: bool = True,
    min_input_words: int = 1,
    max_input_words: Optional[int] = None,
    require_last_assistant: bool = True,
    require_user_confirmed: Optional[bool] = None,
    language: Optional[str] = None,
    output_format: str = "chat",
    system_prompt: Optional[str] = None,
    val_split: float = 0.05,
) -> Tuple[str, Optional[str]]:
    """
    High-level helper to build dataset using training_logger default storage.
    Returns paths (train_path, val_path)
    """
    # load raw
    if get_default_logger is None:
        raise RuntimeError("training_logger not available; cannot load logs automatically.")
    logger.info("Loading raw examples from training_logger...")
    raw = load_raw_from_logger(limit=raw_limit)
    logger.info("Loaded %d raw records", len(raw))

    out = filter_and_transform_records(
        raw,
        redact=redact,
        min_input_words=min_input_words,
        max_input_words=max_input_words,
        require_last_assistant=require_last_assistant,
        require_user_confirmed=require_user_confirmed,
        language=language,
        output_format=output_format,
        system_prompt=system_prompt,
    )
    logger.info("After filtering and transforming: %d records", len(out))
    train_path, val_path = split_and_save(out, out_path, val_split=val_split)
    logger.info("Saved train dataset to: %s", train_path)
    if val_path:
        logger.info("Saved val dataset to: %s", val_path)
    return train_path, val_path


def build_dataset_from_file(
    raw_jsonl_path: str,
    out_path: str,
    *,
    raw_limit: Optional[int] = None,
    redact: bool = True,
    min_input_words: int = 1,
    max_input_words: Optional[int] = None,
    require_last_assistant: bool = True,
    require_user_confirmed: Optional[bool] = None,
    language: Optional[str] = None,
    output_format: str = "chat",
    system_prompt: Optional[str] = None,
    val_split: float = 0.05,
) -> Tuple[str, Optional[str]]:
    raw = load_raw_from_file(raw_jsonl_path, limit=raw_limit)
    logger.info("Loaded %d raw records from file %s", len(raw), raw_jsonl_path)
    out = filter_and_transform_records(
        raw,
        redact=redact,
        min_input_words=min_input_words,
        max_input_words=max_input_words,
        require_last_assistant=require_last_assistant,
        require_user_confirmed=require_user_confirmed,
        language=language,
        output_format=output_format,
        system_prompt=system_prompt,
    )
    logger.info("After filtering and transforming: %d records", len(out))
    train_path, val_path = split_and_save(out, out_path, val_split=val_split)
    logger.info("Saved train dataset to: %s", train_path)
    if val_path:
        logger.info("Saved val dataset to: %s", val_path)
    return train_path, val_path


def _cli():
    p = argparse.ArgumentParser(description="Build fine-tuning dataset from raw interaction logs.")
    p.add_argument("--raw-file", type=str, help="Path to raw JSONL file (if omitted, uses training_logger storage).")
    p.add_argument("--out", type=str, required=True, help="Output JSONL path for train dataset (e.g. data/datasets/chat_finetune.jsonl)")
    p.add_argument("--format", choices=["chat", "instruct"], default="chat", help="Output dataset format.")
    p.add_argument("--val-split", type=float, default=0.05, help="Validation split fraction (0-1).")
    p.add_argument("--limit", type=int, default=None, help="Limit number of raw records to process (for quick tests).")
    p.add_argument("--min-words", type=int, default=1, help="Minimum words in input to keep an example.")
    p.add_argument("--max-words", type=int, default=None, help="Maximum words in input to keep an example.")
    p.add_argument("--require-confirmed", action="store_true", help="Keep only user-confirmed examples if metadata present.")
    p.add_argument("--language", type=str, default=None, help="Filter by language code (requires langdetect).")
    p.add_argument("--no-redact", dest="redact", action="store_false", help="Disable PII redaction.")
    p.add_argument("--system-prompt", type=str, default=None, help="Optional system prompt to include in chat records.")
    args = p.parse_args()

    if args.raw_file:
        build_dataset_from_file(
            raw_jsonl_path=args.raw_file,
            out_path=args.out,
            raw_limit=args.limit,
            redact=args.redact,
            min_input_words=args.min_words,
            max_input_words=args.max_words,
            require_last_assistant=True,
            require_user_confirmed=(True if args.require_confirmed else None),
            language=args.language,
            output_format=args.format,
            system_prompt=args.system_prompt,
            val_split=args.val_split,
        )
    else:
        build_dataset_from_logger(
            out_path=args.out,
            raw_limit=args.limit,
            redact=args.redact,
            min_input_words=args.min_words,
            max_input_words=args.max_words,
            require_last_assistant=True,
            require_user_confirmed=(True if args.require_confirmed else None),
            language=args.language,
            output_format=args.format,
            system_prompt=args.system_prompt,
            val_split=args.val_split,
        )


if __name__ == "__main__":
    _cli()
