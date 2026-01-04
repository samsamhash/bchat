# config.py
"""
Configuration and utilities to detect fine-tuning artifacts (full models or PEFT adapters)
and help load the model correctly in environments such as Google Colab.

Exports (among others):
 - LLM_MODEL: path or ID to pass to AutoTokenizer / AutoModelForCausalLM.from_pretrained
 - LLM_ADAPTER_PATH: if not None, PEFT adapter directory (apply with apply_adapter_to_model)
 - LLM_IS_ADAPTER: True if LLM_ADAPTER_PATH is present / an adapter-only setup was detected
 - LLM_TEMPERATURE, LLM_MAX_TOKENS, SEARCH_TOP_K
 - apply_adapter_to_model(model, adapter_path): robust function to apply a PEFT adapter
 - load_tokenizer_and_model_from_config(...): helper to load the tokenizer/model and apply the adapter

"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# -------------------------
# Defaults 
# -------------------------
# Base model id or route local

LLM_BASE_MODEL: str = os.getenv("BASE_MODEL") or os.getenv("LLM_MODEL") or "Qwen/Qwen2.5-1.5B-Instruct"


_FINETUNE_SEARCH_ROOTS = os.getenv("FINETUNE_SEARCH_ROOTS", "models,./models").split(",")

# Inference
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", os.getenv("LLM_TEMP", "0.2")))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "256"))
SEARCH_TOP_K: int = int(os.getenv("SEARCH_TOP_K", "5"))


LLM_MODEL: str = LLM_BASE_MODEL
LLM_ADAPTER_PATH: Optional[str] = None
LLM_IS_ADAPTER: bool = False


_FULL_MODEL_MARKERS = {
    "pytorch_model.bin", "pytorch_model.safetensors", "model.safetensors", "tf_model.h5", "model.bin"
}
_ADAPTER_MARKERS = {
    "adapter_model.bin", "adapter_model.safetensors", "adapter_config.json", "pytorch_lora_weights.bin",
    "pytorch_lora.bin"
}


_TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "vocab.txt", "merges.txt", "tokenizer.model",
    "special_tokens_map.json"
}

# -------------------------
# Utilities
# -------------------------
def _dir_contains_any_files(p: Path, names: set) -> bool:
    try:
        files = {f.name for f in p.iterdir() if f.is_file()}
        return any(n in files for n in names)
    except Exception:
        return False


def _dir_is_full_model(p: Path) -> bool:
    return _dir_contains_any_files(p, _FULL_MODEL_MARKERS)


def _dir_is_adapter(p: Path) -> bool:
    return _dir_contains_any_files(p, _ADAPTER_MARKERS)


def _dir_has_tokenizer(p: Path) -> bool:
    return _dir_contains_any_files(p, _TOKENIZER_FILES)


def _iter_candidate_dirs(roots: List[str]) -> List[Tuple[Path, float, str]]:
    """
    Iterates over roots (not deep recursive: only checks immediate subdirectories) and returns
    tuples (path, mtime, kind) where kind is "full" or "adapter".
    """
    candidates: List[Tuple[Path, float, str]] = []
    for root in roots:
        if not root:
            continue
        root_p = Path(root).expanduser().resolve()
        if not root_p.exists():
            continue
        # examine immediate children
        if root_p.is_dir():
            # check subdirectories
            for child in root_p.iterdir():
                if not child.exists() or not child.is_dir():
                    continue
                try:
                    if _dir_is_full_model(child):
                        candidates.append((child, child.stat().st_mtime, "full"))
                        continue
                    if _dir_is_adapter(child):
                        candidates.append((child, child.stat().st_mtime, "adapter"))
                        continue
                except Exception:
                    continue
           
            try:
                if _dir_is_full_model(root_p):
                    candidates.append((root_p, root_p.stat().st_mtime, "full"))
                elif _dir_is_adapter(root_p):
                    candidates.append((root_p, root_p.stat().st_mtime, "adapter"))
            except Exception:
                pass
    return candidates


def _select_latest_candidate(cands: List[Tuple[Path, float, str]]) -> Optional[Tuple[Path, str]]:
   
   
    if not cands:
        return None
    cands_sorted = sorted(cands, key=lambda x: (x[1], 0 if x[2] == "full" else 1), reverse=True)
    chosen = cands_sorted[0]
    return (chosen[0], chosen[2])

# -------------------------
# Detection
# -------------------------
def _detect_finetuned_artifact():
    """
    Detects the most recent fine-tuning artifact under _FINETUNE_SEARCH_ROOTS.
Priority order:
 Environment override: LLM_MODEL_FORCE (use as a full model)
 Environment override: LLM_ADAPTER_PATH (if it points to an adapter or a full model, act accordingly)
 Automatic search: the latest directory containing a full model or an adapter.
 Updates the global variables LLM_MODEL, LLM_ADAPTER_PATH, and LLM_IS_ADAPTER
     """
    global LLM_MODEL, LLM_ADAPTER_PATH, LLM_IS_ADAPTER

    env_force = os.getenv("LLM_MODEL_FORCE")
    env_adapter = os.getenv("LLM_ADAPTER_PATH")

    if env_force:
        LLM_MODEL = env_force
        LLM_ADAPTER_PATH = None
        LLM_IS_ADAPTER = False
        logger.info("LLM_MODEL_FORCE set -> using %s", LLM_MODEL)
        return

    if env_adapter:
        p = Path(env_adapter).expanduser().resolve()
        if p.exists():
            if p.is_dir():
                if _dir_is_full_model(p):
                    LLM_MODEL = str(p)
                    LLM_ADAPTER_PATH = None
                    LLM_IS_ADAPTER = False
                    logger.info("LLM_ADAPTER_PATH override points to full model -> LLM_MODEL=%s", LLM_MODEL)
                else:
                    LLM_MODEL = LLM_BASE_MODEL
                    LLM_ADAPTER_PATH = str(p)
                    LLM_IS_ADAPTER = True
                    logger.info("LLM_ADAPTER_PATH override -> adapter will be applied on top of base %s (adapter=%s)", LLM_MODEL, LLM_ADAPTER_PATH)
            else:
                # If user supplied a file, not a directory, treat as model file if possible
                LLM_MODEL = str(p)
                LLM_ADAPTER_PATH = None
                LLM_IS_ADAPTER = False
                logger.info("LLM_ADAPTER_PATH override points to file -> using as LLM_MODEL=%s", LLM_MODEL)
            return
        else:
            logger.warning("LLM_ADAPTER_PATH override set but path not found: %s", env_adapter)

    # No overrides: search
    candidates = _iter_candidate_dirs(_FINETUNE_SEARCH_ROOTS)
    pick = _select_latest_candidate(candidates)
    if not pick:
        logger.info("No finetuned artifacts detected under %s. Using base model %s", _FINETUNE_SEARCH_ROOTS, LLM_BASE_MODEL)
        LLM_MODEL = LLM_BASE_MODEL
        LLM_ADAPTER_PATH = None
        LLM_IS_ADAPTER = False
        return

    chosen_path, kind = pick
    chosen_str = str(chosen_path)
    # If directory contains finetune_config_saved.json prefer it (more reliable)
    cfg_saved = chosen_path / "finetune_config_saved.json"
    if cfg_saved.exists():
        logger.info("Found finetune_config_saved.json in %s; treating as valid finetune output.", chosen_path)

    if kind == "full":
        LLM_MODEL = chosen_str
        LLM_ADAPTER_PATH = None
        LLM_IS_ADAPTER = False
        logger.info("Detected full fine-tuned model at %s. Set LLM_MODEL -> %s", chosen_str, LLM_MODEL)
    else:
        LLM_MODEL = LLM_BASE_MODEL
        LLM_ADAPTER_PATH = chosen_str
        LLM_IS_ADAPTER = True
        logger.info("Detected PEFT adapter at %s. LLM_MODEL=%s, LLM_ADAPTER_PATH=%s", chosen_str, LLM_MODEL, LLM_ADAPTER_PATH)


# Run detection on import
try:
    _detect_finetuned_artifact()
except Exception as e:
    logger.exception("Error during detection of finetuned artifacts: %s", e)
    LLM_MODEL = LLM_BASE_MODEL
    LLM_ADAPTER_PATH = None
    LLM_IS_ADAPTER = False

# -------------------------
# Helpers 
# -------------------------
def _resize_model_embeddings_if_needed(model, target_rows: int):
    """
    Resize embeddings safely for models or PEFT-wrapped models.
    """
    try:
        # Direct API
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(target_rows)
            logger.info("Resized embeddings via model.resize_token_embeddings -> %d", target_rows)
            return
        # Try base_model (PEFT wrappers)
        base = getattr(model, "base_model", None)
        if base is not None and hasattr(base, "resize_token_embeddings"):
            base.resize_token_embeddings(target_rows)
            logger.info("Resized embeddings via model.base_model.resize_token_embeddings -> %d", target_rows)
            return
        # Try nested attributes (some architectures)
        cand = getattr(model, "model", None) or getattr(model, "transformer", None)
        if cand is not None and hasattr(cand, "resize_token_embeddings"):
            cand.resize_token_embeddings(target_rows)
            logger.info("Resized embeddings via nested model.resize_token_embeddings -> %d", target_rows)
            return
        raise RuntimeError("No resize_token_embeddings method found on model or its base_model.")
    except Exception as e:
        logger.exception("Failed to resize embeddings: %s", e)
        raise


def _inspect_checkpoint_for_embedding_rows(checkpoint_file: Path) -> Optional[int]:
    """
   Attempts to read the embedding shape from checkpoints (.safetensors or .bin).
   Returns an int or None.
    """
    try:
        import torch
        suffix = checkpoint_file.suffix.lower()
        if suffix == ".safetensors":
            try:
                
                from safetensors import safe_open as _safe_open  # type: ignore
                with _safe_open(str(checkpoint_file), framework="pt") as f:
                    for k in f.keys():
                        if "embed_tokens.weight" in k or "embed_tokens" in k:
                            shape = f.get_shape(k)
                            if len(shape) == 2:
                                return shape[0]
            except Exception:
               
                try:
                    from safetensors.torch import load_file as _load_safetensors  # type: ignore
                    sd = _load_safetensors(str(checkpoint_file))
                    for k, v in sd.items():
                        if "embed_tokens.weight" in k or "embed_tokens" in k:
                            if hasattr(v, "shape"):
                                return int(v.shape[0])
                except Exception:
                    logger.debug("Could not inspect safetensors checkpoint shapes (safetensors not available).")
                    return None
        else:
           
            try:
                sd = torch.load(str(checkpoint_file), map_location="cpu")
            except Exception:
                return None
            if isinstance(sd, dict):
                # state_dict or nested {"state_dict": {...}}
                if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                    sd = sd["state_dict"]
                for k, v in sd.items():
                    if isinstance(k, str) and ("embed_tokens.weight" in k or "lm_head.weight" in k):
                        try:
                            return int(v.shape[0])
                        except Exception:
                            continue
        return None
    except Exception as e:
        logger.warning("Failed to inspect checkpoint %s: %s", checkpoint_file, e)
        return None


def apply_adapter_to_model(model, adapter_path: str):
    """
    Applies a PEFT adapter to a base model, attempting to:
    load the adapter’s tokenizer (if present) and resize embeddings
    or inspect the checkpoint to determine the vocab size and resize embeddings
    finally call PeftModel.from_pretrained to wrap the model
    Returns the wrapped model (PeftModel) or raises an exception with a clear error message.
    """
    try:
        from peft import PeftModel
    except Exception as exc:
        raise ImportError("PEFT no está instalado. Instala 'peft' para aplicar adapters.") from exc

    import torch  

    adapter_dir = Path(adapter_path).expanduser().resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_dir}")

    
    try:
        if _dir_has_tokenizer(adapter_dir):
            from transformers import AutoTokenizer
            logger.info("Loading tokenizer from adapter dir %s to determine vocab size.", adapter_dir)
            tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True, use_fast=True)
            target_vocab = getattr(tok, "vocab_size", None) or getattr(tok, "vocab_size", None)
            if target_vocab:
                current_rows = model.get_input_embeddings().weight.shape[0]
                if int(target_vocab) != int(current_rows):
                    logger.info("Adapter tokenizer vocab_size=%d vs model emb rows=%d -> resizing.", target_vocab, current_rows)
                    print("resize")
                    _resize_model_embeddings_if_needed(model, int(target_vocab))
    except Exception as e:
        logger.warning("Failed to load adapter tokenizer or determine vocab size: %s", e)

    
    try:
        
        candidates = ["adapter_model.safetensors", "adapter_model.bin", "pytorch_lora_weights.bin",
                      "pytorch_model.bin", "pytorch_model.safetensors", "pytorch_lora.bin"]
        for cand in candidates:
            candp = adapter_dir / cand
            if candp.exists():
                rows = _inspect_checkpoint_for_embedding_rows(candp)
                print("checkpoint exist")
                if rows:
                    current_rows = model.get_input_embeddings().weight.shape[0]
                    if int(rows) != int(current_rows):
                        logger.info("Checkpoint indicates embedding rows=%d but model has %d -> resizing.", rows, current_rows)
                        print("checkpoint resize")
                        _resize_model_embeddings_if_needed(model, int(rows))
                    break
    except Exception as e:
        logger.debug("No checkpoint inspected or error inspecting: %s", e)

    
    try:
        device_map = "auto" if (os.getenv("CUDA_VISIBLE_DEVICES") or (lambda: True)()) else None
        # prefer device_map 'auto' when CUDA available; 
        wrapped = PeftModel.from_pretrained(model, str(adapter_dir), device_map="auto" if _cuda_available() else None)
        logger.info("PEFT adapter aplicado desde %s", adapter_dir)
        print("peft adapt")
        return wrapped
    except Exception as e:
        msg = (
            f"Failed to load PEFT adapter from {adapter_dir}: {e}\n\n"
            
        )
        logger.exception(msg)
        raise RuntimeError(msg) from e


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# -------------------------
# Helper 
# -------------------------
def load_tokenizer_and_model_from_config(
    llm_model: Optional[str] = None,
    adapter_path: Optional[str] = None,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
    prefer_fp16_on_cuda: bool = True,
):
    """
   Loads the tokenizer and model based on the detection performed in this module.
   Optional parameters allow overriding llm_model and adapter_path.
   Returns (tokenizer, model, device_str).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    llm_model = llm_model or LLM_MODEL
    adapter_path = adapter_path if adapter_path is not None else LLM_ADAPTER_PATH
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading tokenizer/model. base_model=%s adapter=%s device=%s", llm_model, adapter_path, device)

    
    tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=trust_remote_code, use_fast=True)

    
    load_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if device == "cuda":
        torch_dtype = torch.float16 if prefer_fp16_on_cuda else torch.float32
        load_kwargs.update({"device_map": "auto", "torch_dtype": torch_dtype})
    else:
        load_kwargs.update({"device_map": None})

    
    try:
        model = AutoModelForCausalLM.from_pretrained(llm_model, **load_kwargs)
    except Exception as e:
        logger.warning("Carga con dtype/device_map falló: %s. Reintentando sin torch_dtype...", e)
        safe_kwargs = {"trust_remote_code": trust_remote_code}
        safe_kwargs["device_map"] = "auto" if device == "cuda" else None
        model = AutoModelForCausalLM.from_pretrained(llm_model, **safe_kwargs)

    
    if adapter_path:
        try:
            model = apply_adapter_to_model(model, adapter_path)
        except Exception as e:
            logger.exception("apply_adapter_to_model falló: %s", e)
            raise

    
    try:
        if device == "cuda":
            model.to(torch.device("cuda"))
        else:
            model.to(torch.device("cpu"))
    except Exception:
        
        pass

    return tokenizer, model, device

# -------------------------
# Resumen 
# -------------------------
def print_config_summary():
    lines = [
        f"LLM_BASE_MODEL: {LLM_BASE_MODEL}",
        f"LLM_MODEL (to pass to from_pretrained): {LLM_MODEL}",
        f"LLM_IS_ADAPTER: {LLM_IS_ADAPTER}",
        f"LLM_ADAPTER_PATH: {LLM_ADAPTER_PATH}",
        f"LLM_TEMPERATURE: {LLM_TEMPERATURE}",
        f"LLM_MAX_TOKENS: {LLM_MAX_TOKENS}",
        f"SEARCH_TOP_K: {SEARCH_TOP_K}",
        f"FINETUNE_SEARCH_ROOTS: {_FINETUNE_SEARCH_ROOTS}",
    ]
    logger.info("Config summary:\n%s", "\n".join(lines))


try:
    if "google.colab" in str(os.sys.modules.get("google.colab", "")) or os.getenv("COLAB_GPU") or os.getenv("CI") is None:
        print_config_summary()
except Exception:
    pass



if __name__ == "__main__":
    print_config_summary()
    print("Detected artifact:")
    print("  LLM_MODEL =", LLM_MODEL)
    print("  LLM_ADAPTER_PATH =", LLM_ADAPTER_PATH)
    print("  LLM_IS_ADAPTER =", LLM_IS_ADAPTER)

