# llm_client.py
"""
LLM client for local inference with optional LoRA/PEFT adapter loading.


- Load tokenizer and model
- If a fine-tuned adapter exists in the configured OUTPUT_DIR, apply it 
- Expose:
    - call_llm_local(prompt: str, max_new_tokens: Optional[int]=None, temperature: Optional[float]=None) -> str
    - answer_query(query: str, top_k: int = 5) -> str   (compat API used by other modules)
    - reload_model(path: Optional[str]=None)           (reload model / switch adapter)
- Provide a global VectorStore instance `vs` for retrieval (keeps project compatibility)
"""

from __future__ import annotations

import os
import sys
import time
import logging
from typing import Optional, List, Dict, Any
import platform
import shutil
import subprocess
import re


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# PEFT
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None  # type: ignore
    _HAS_PEFT = False

# Try to import config helper 
try:
    import config as config_module
    _HAS_CONFIG_MODULE = True
except Exception:
    config_module = None
    _HAS_CONFIG_MODULE = False

# Project imports (assumed to exist in repo)
try:
    from vector_store import VectorStore
except Exception:
    # if running from different cwd, try to append repo root
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.append(CURRENT_DIR)
    from vector_store import VectorStore  # re-try

# Optional prompt builder 
try:
    from prompts import build_prompt
except Exception:
    def build_prompt(q, contexts):  # fallback simple prompt
        ctx = "\n\n".join(contexts) if contexts else ""
        return f"CONTEXT:\n{ctx}\n\nQUESTION:\n{q}\n\nFINAL ANSWER:"

# Optional configuration source: finetune_config.get_config()
try:
    from finetune_config import get_config
    _HAS_FINETUNE_CFG = True
    _FINETUNE_CFG = get_config()
except Exception:
    _HAS_FINETUNE_CFG = False
    _FINETUNE_CFG = None

# Backwards-compatible environment defaults
LLM_MODEL_ENV = os.getenv("LLM_MODEL", None)
LLM_TEMPERATURE_ENV = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS_ENV = int(os.getenv("LLM_MAX_TOKENS", "256"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------
# Globals
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer: Optional[AutoTokenizer] = None
model: Optional[torch.nn.Module] = None
_model_loaded_from: Optional[str] = None  # path or model id used to load
_is_finetuned: bool = False
_finetune_type: Optional[str] = None  # "lora" | "full"
_vs: Optional[VectorStore] = None

# Expose a global VectorStore instance for the project (legacy compatibility: llm_client.vs)
vs = None  # will be initialized later



# Internal helpers
# ------------------------
def _resolve_model_source() -> Dict[str, Any]:
    """
    Determine which base model and adapter path to use.
    Returns dict with keys:
      - base_model (str)
      - adapter_path (Optional[str])
      - load_in_8bit (bool)
    """
    base_model = None
    adapter_path = None
    load_in_8bit = False

    # Priority: finetune_config if available
    if _HAS_FINETUNE_CFG and _FINETUNE_CFG is not None:
        base_model = _FINETUNE_CFG.BASE_MODEL
        adapter_candidate = _FINETUNE_CFG.OUTPUT_DIR
        load_in_8bit = bool(_FINETUNE_CFG.LOAD_IN_8BIT)
        # Consider adapter present if directory exists and contains files
        if adapter_candidate and os.path.isdir(adapter_candidate) and any(os.listdir(adapter_candidate)):
            adapter_path = adapter_candidate

    # fallback: environment variable LLM_MODEL
    if base_model is None:
        base_model = LLM_MODEL_ENV or "gpt2"

    # allow explicit override by env variable ADAPTER_PATH
    env_adapter = os.getenv("LLM_ADAPTER_PATH")
    if env_adapter:
        adapter_path = env_adapter

    return {"base_model": base_model, "adapter_path": adapter_path, "load_in_8bit": load_in_8bit}


def _load_tokenizer_and_model():
    """
    Loads tokenizer and model into globals `tokenizer` and `model`.
    Applies PEFT adapter if available. Uses config.apply_adapter_to_model when present
    to handle embedding resizing when tokenizer/vocab mismatches exist.
    """
    global tokenizer, model, _model_loaded_from, device, _is_finetuned, _finetune_type

    info = _resolve_model_source()
    base_model = info["base_model"]
    adapter_path = info["adapter_path"]
    load_in_8bit = info["load_in_8bit"]

    logger.info("[LLM] Device: %s", device)
    logger.info("[LLM] Loading tokenizer from: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    # ensure pad token exists
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is None:
            tokenizer.add_special_tokens({"eos_token": ""})
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare load kwargs
    load_kwargs: Dict[str, Any] = {}
    # prefer dtype argument instead of deprecated torch_dtype
    if device == "cuda":
        # If load_in_8bit is set, rely on bitsandbytes + device_map
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            # use automatic device map to distribute layers if possible
            load_kwargs["device_map"] = "auto"
            # choose dtype: use fp16 on CUDA when possible
            load_kwargs["torch_dtype"] = torch.float16
    else:
        # CPU
        load_kwargs["device_map"] = None
        load_kwargs["torch_dtype"] = torch.float32

    logger.info("[LLM] Loading base model %s (kwargs: %s)", base_model, {k: v for k, v in load_kwargs.items()})
    # Load base model
    model_obj = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    # If adapter exists and PEFT available, attempt to apply using robust helper from config if available
    _is_finetuned = False
    _finetune_type = None
    if adapter_path and _HAS_PEFT:
        # Prefer config.apply_adapter_to_model if available, it handles tokenizer-check, checkpoint-inspection and resizing.
        if _HAS_CONFIG_MODULE and hasattr(config_module, "apply_adapter_to_model"):
            try:
                logger.info("[LLM] Attempting to apply adapter via config.apply_adapter_to_model from %s", adapter_path)

                # If adapter contains a tokenizer, prefer to use it for tokenizer 
                try:
                    # Try loading tokenizer from adapter path; if succeeds, replace global tokenizer
                    adapter_tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True, use_fast=True)
                    if adapter_tok is not None:
                        tokenizer = adapter_tok
                        logger.info("[LLM] Replaced tokenizer with adapter tokenizer from %s (vocab_size=%s)", adapter_path, getattr(tokenizer, "vocab_size", None))
                except Exception as e:
                    logger.debug("[LLM] Adapter tokenizer not loaded or not present at %s: %s", adapter_path, e)

                # apply_adapter_to_model will attempt to resize embeddings if necessary and wrap with PeftModel
                model_obj = config_module.apply_adapter_to_model(model_obj, adapter_path)
                _is_finetuned = True
                _finetune_type = "lora"
                logger.info("[LLM] PEFT adapter applied (via config helper).")
            except Exception as e:
                logger.exception("[LLM] config.apply_adapter_to_model failed: %s", e)
                # fallback to direct PEFT load below
        else:
            # fallback: try direct PeftModel.from_pretrained, but catch shape mismatch errors and log clear message
            try:
                logger.info("[LLM] Applying PEFT adapter from %s using PeftModel.from_pretrained()", adapter_path)
                model_obj = PeftModel.from_pretrained(model_obj, adapter_path, device_map="auto" if device == "cuda" else None)
                _is_finetuned = True
                _finetune_type = "lora"
            except Exception as e:
                # If PEFT load fails with shape mismatch, provide a clear error and re-raise so caller can debug.
                logger.exception("[LLM] Failed to load PEFT adapter: %s", e)
                # Re-raise to allow top-level to handle it (or to show the message)
                raise

    # Move model to correct device if not already (some device_map settings handle this)
    try:
        # If model is not already in desired dtype/device, ensure lowest common denominator
        if device == "cuda" and not next(model_obj.parameters()).is_cuda:
            model_obj.to(torch.device("cuda"))
        elif device == "cpu":
            model_obj.to(torch.device("cpu"))
    except Exception:
        pass

    # set globals
    model = model_obj
    _model_loaded_from = adapter_path or base_model

    # Final logging about finetuned vs base
    if _is_finetuned:
        logger.info("[LLM]  Fine-tuned adapter active (type=%s). loaded_from=%s", _finetune_type, _model_loaded_from)
    else:
        logger.info("[LLM]  Base model active (no adapter applied). loaded_from=%s", _model_loaded_from)


def reload_model(adapter_or_model_path: Optional[str] = None) -> None:
    """
    Public helper: reloads model. If adapter_or_model_path points to a directory with adapter,
    prefer to load base model from config and then apply adapter; if it points to a model id, load that.
    """
    global _FINETUNE_CFG, _model_loaded_from
    if adapter_or_model_path:
        # try to treat arg as adapter path; temporarily override env
        os.environ["LLM_ADAPTER_PATH"] = adapter_or_model_path
    else:
        # remove override
        if "LLM_ADAPTER_PATH" in os.environ:
            del os.environ["LLM_ADAPTER_PATH"]

    logger.info("[LLM] Reloading model (adapter override: %s)", adapter_or_model_path)
    _load_tokenizer_and_model()
    logger.info("[LLM] Reload complete. Loaded from: %s", _model_loaded_from)


# Initialize model & tokenizer at import time (best-effort)
try:
    # Initialize VectorStore global
    vs = VectorStore()
except Exception as e:
    logger.exception("Failed to initialize VectorStore at import: %s", e)
    vs = None

try:
    _load_tokenizer_and_model()
except Exception as e:
    logger.exception("Initial model load failed: %s", e)
    # Leave tokenizer/model as None; callers should handle None and possibly call reload_model()


# ------------------------
# Public inference function
# ------------------------
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

 # elimina colores ANSI

import requests

def call_llm_local(prompt: str, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
    """
    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    Returns the assistant text (string).
    """
    
    url = "http://127.0.0.1:8081/v1/chat/completions"
    payload = {
         "model": "local-model",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant specialized in answering questions using retrieved documentation. Rules: - Use ONLY the information provided in the context. - Do NOT use external knowledge. - Do NOT invent details. - If the answer cannot be found in the context, reply with: Information not available. - Keep the answer clear and concise."},
        {"role": "user", "content": prompt}
    ],
        "temperature": 0.7,
    }
    r = requests.post(url, json=payload, timeout=120)
    j=r.json()["choices"][0]["message"]["content"]
    
    
    return j



# ------------------------
# Compatibility wrapper: answer_query (RAG pipeline)
# ------------------------
def answer_query(query: str, top_k: int = 5) -> str:
    """
    Compatibility function used by the rest of the project.
    - Retrieve contexts from the global vectorstore `vs`
    - Build a prompt using prompts.build_prompt (or fallback)
    - Call call_llm_local and return the answer
    """
    global vs
    if vs is None:
        # try to initialize
        try:
            vs = VectorStore()
        except Exception as e:
            logger.exception("VectorStore not available: %s", e)
            return " Vector store not available."

    contexts = vs.retrieve(query, top_k=top_k)
    # Provide debug info
    logger.info("[RAG] Retrieved %d contexts for query.", len(contexts) if contexts else 0)

    if not contexts:
        return " No relevant documents found in the vector store."

    # contexts might be list of dicts; prompts.build_prompt expects list of strings in older contract
    # We'll coerce: if contexts are dicts, join their 'text' fields.
    if contexts and isinstance(contexts[0], dict):
        contexts_texts = [c.get("text", "") for c in contexts]
    else:
        contexts_texts = contexts

    prompt = build_prompt(query, contexts_texts)

    # Call LLM
    try:
        answer = call_llm_local(prompt)
    except Exception as e:
        logger.exception("answer_query generation error: %s", e)
        return f" Error generating answer: {e}"

    return answer



# A helper for interactive switching of adapter/model
# ------------------------
def get_loaded_model_info() -> Dict[str, Any]:
    return {
        "model_loaded_from": _model_loaded_from,
        "is_finetuned": _is_finetuned,
        "finetune_type": _finetune_type,
        "device": device,
        "tokenizer": getattr(tokenizer, "__class__", None).__name__ if tokenizer else None,
        "model": getattr(model, "__class__", None).__name__ if model else None,
        "vectorstore_info": vs.info() if vs is not None else None,
    }

# End of llm_client.py

