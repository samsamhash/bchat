# finetune.py
"""
fine-tuning script (LoRA / PEFT) for causal LMs for qwen.

"""

from __future__ import annotations

import os
import json
import argparse
import logging
from typing import List, Dict, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Try to import project finetune_config helpers if present
try:
    from finetune_config import get_config, save_config  # type: ignore
    _HAS_PROJECT_CONFIG = True
except Exception:
    _HAS_PROJECT_CONFIG = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# -----------------------
# Defaults (used if no project config)
# -----------------------
DEFAULTS = {
    "MODEL_NAME": "Qwen/Qwen2.5-1.5B-Instruct",
    "OUTPUT_DIR": "./qwen-lora-ft",
    "MAX_LENGTH": 1024,
    "BATCH_SIZE": 2,
    "GRAD_ACC": 1,
    "EPOCHS": 3,
    "LR": 2e-4,
    "FP16": True,
    "LOAD_IN_8BIT": True,
    "LORA": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
}


# -----------------------
# Utilities
# -----------------------
def load_project_config_or_defaults() -> Dict:
    """Return a dict of configuration values"""
    if not _HAS_PROJECT_CONFIG:
        return DEFAULTS.copy()
    cfg_obj = get_config()
    out = DEFAULTS.copy()
    out["MODEL_NAME"] = getattr(cfg_obj, "BASE_MODEL", out["MODEL_NAME"])
    out["OUTPUT_DIR"] = getattr(cfg_obj, "OUTPUT_DIR", out["OUTPUT_DIR"])
    out["MAX_LENGTH"] = getattr(getattr(cfg_obj, "data", None), "max_seq_length", out["MAX_LENGTH"])
    t = getattr(cfg_obj, "training", None)
    if t:
        out["BATCH_SIZE"] = getattr(t, "per_device_train_batch_size", out["BATCH_SIZE"])
        out["GRAD_ACC"] = getattr(t, "gradient_accumulation_steps", out["GRAD_ACC"])
        out["EPOCHS"] = getattr(t, "num_train_epochs", out["EPOCHS"])
        out["LR"] = getattr(t, "learning_rate", out["LR"])
        out["FP16"] = getattr(t, "fp16", out["FP16"])
        out["GRADIENT_CHECKPOINTING"] = getattr(t, "gradient_checkpointing", True)
    out["LOAD_IN_8BIT"] = getattr(cfg_obj, "LOAD_IN_8BIT", out["LOAD_IN_8BIT"])
    l = getattr(cfg_obj, "lora", None)
    if l:
        out["LORA"] = {
            "r": getattr(l, "r", out["LORA"]["r"]),
            "lora_alpha": getattr(l, "lora_alpha", out["LORA"]["lora_alpha"]),
            "lora_dropout": getattr(l, "lora_dropout", out["LORA"]["lora_dropout"]),
            "target_modules": getattr(l, "target_modules", out["LORA"]["target_modules"]),
        }
    return out


def safe_save_config(cfg_dict: Dict, out_dir: str):
    """Save configuration: if project helpers present, try to save original object; otherwise dump dict."""
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "finetune_config_saved.json")
    if _HAS_PROJECT_CONFIG:
        try:
            original_cfg = get_config()
            save_config(original_cfg, save_path)
            logger.info("Saved project config object using save_config() -> %s", save_path)
            return
        except Exception as e:
            logger.warning("Could not call save_config on original object: %s. Falling back to JSON dump.", e)
    # fallback: JSON dump
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
    logger.info("Saved config dict to %s", save_path)


def resize_model_embeddings(model, new_size: int):
    """
    Resize model embedding ,works whether model is wrapped by PEFT or not.
    """
    try:
        # Some wrappers (PEFT) may forward resize_token_embeddings to base model; try directly
        resize_fn = getattr(model, "resize_token_embeddings", None)
        if callable(resize_fn):
            resize_fn(new_size)
            logger.info("Resized embeddings via model.resize_token_embeddings -> %d", new_size)
            return
        # Fallback: try base_model
        base = getattr(model, "base_model", None)
        if base is not None and hasattr(base, "resize_token_embeddings"):
            base.resize_token_embeddings(new_size)
            logger.info("Resized embeddings via model.base_model.resize_token_embeddings -> %d", new_size)
            return
    except Exception as e:
        logger.exception("Failed to resize embeddings to %d: %s", new_size, e)
        raise


# -----------------------
# Data processing 
# -----------------------
def process_example_messages(messages: List[Dict], tokenizer, max_length: int):
    """
    Given a list of messages [{'role':..., 'content':...}, ...], return a training example dict:
    {input_ids: [...], attention_mask: [...], labels: [...]}
    or None if example cannot be built or has no supervised signal.
    """
    # find last assistant reply
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant" and messages[i].get("content"):
            last_assistant_idx = i
            break
    if last_assistant_idx is None:
        return None

    prompt_msgs = messages[:last_assistant_idx]
    answer = messages[last_assistant_idx].get("content", "")

    # try to use tokenizer.apply_chat_template when available
    try:
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for m in prompt_msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(("User: " if role == "user" else "Assistant: ") + content)
        # add a generation prompt marker to signal assistant continuation
        prompt_text = "\n".join(parts) + ("\nAssistant: " if parts else "")

    full_text = prompt_text + answer

    # tokenize prompt to compute mask length (no special tokens)
    prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length, add_special_tokens=False)
    prompt_ids = prompt_enc.get("input_ids", []) or []

    # tokenize full (allow special tokens)
    full_enc = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=True)
    full_ids = full_enc.get("input_ids", [])
    attn = full_enc.get("attention_mask", [])

    if not full_ids:
        return None

    labels = full_ids.copy()
    n_mask = min(len(prompt_ids), len(labels))
    for i in range(n_mask):
        labels[i] = -100

    # if all labels are -100 -> no supervised signal
    if all(l == -100 for l in labels):
        return None

    return {"input_ids": full_ids, "attention_mask": attn, "labels": labels}


def collate_fn(batch: List[Dict], pad_token_id: int):
    """Pad batch to same length (simple collator)."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# -----------------------
# Sanity forward & validation
# -----------------------
def validate_and_fix_batch(model, batch: Dict[str, torch.Tensor], tokenizer, model_max_pos: Optional[int] = None):
    """
    Ensure batch dtypes & device, check token ids vs embedding rows and resize when needed,
    check seq length vs model_max_pos and truncate suffix to safe length.
    Returns the fixed batch on model device.
    """
    device = next(model.parameters()).device

    # ensure long type
    for k in ("input_ids", "attention_mask", "labels"):
        if k in batch:
            batch[k] = batch[k].long()

    # move to model device
    batch = {k: v.to(device) for k, v in batch.items()}

    # diagnostic
    try:
        max_id = int(batch["input_ids"].max().item())
    except Exception:
        max_id = -1
    model_emb_rows = model.get_input_embeddings().weight.shape[0]
    logger.info("Sanity batch check: max_token_id=%d model_emb_rows=%d", max_id, model_emb_rows)

    # resize embeddings if necessary
    if max_id >= model_emb_rows:
        new_size = max_id + 1
        logger.warning("Token id outside embeddings detected (%d >= %d). Resizing embeddings -> %d", max_id, model_emb_rows, new_size)
        resize_model_embeddings(model, new_size)
        model_emb_rows = model.get_input_embeddings().weight.shape[0]
        logger.info("Resized embeddings. New rows=%d", model_emb_rows)

    # check sequence length vs model positions
    seq_len = batch["input_ids"].shape[1]
    if model_max_pos:
        safe_max = max(1, int(model_max_pos) - 2)
        if seq_len > safe_max:
            logger.warning("Sequence length %d > model_max_pos %d. Truncating to last %d tokens.", seq_len, model_max_pos, safe_max)
            keep = safe_max
            batch["input_ids"] = batch["input_ids"][:, -keep:].to(device)
            batch["attention_mask"] = batch["attention_mask"][:, -keep:].to(device)
            if "labels" in batch:
                batch["labels"] = batch["labels"][:, -keep:].to(device)
            seq_len = keep

    # final validation
    final_max = int(batch["input_ids"].max().item())
    if final_max >= model.get_input_embeddings().weight.shape[0]:
        raise RuntimeError(f"After fixes, invalid token id {final_max} >= embeddings {model.get_input_embeddings().weight.shape[0]}")

    return batch


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Robust fine-tune (LoRA/PEFT) script")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL dataset (chat format)")
    parser.add_argument("--config-out", type=str, default=None, help="Output directory override")
    parser.add_argument("--max-seq", type=int, default=None, help="Max sequence length override")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit even if default config requests it")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from checkpoint (optional)")
    args = parser.parse_args()

    cfg = load_project_config_or_defaults()
    model_name = cfg["MODEL_NAME"]
    out_dir = args.config_out or cfg["OUTPUT_DIR"]
    max_length = int(args.max_seq) if args.max_seq else int(cfg["MAX_LENGTH"])
    batch_size = int(cfg["BATCH_SIZE"])
    grad_acc = int(cfg.get("GRAD_ACC", 1))
    epochs = int(cfg["EPOCHS"])
    lr = float(cfg["LR"])
    fp16 = bool(cfg["FP16"])
    load_in_8bit = bool(cfg["LOAD_IN_8BIT"]) and (not args.no_8bit)
    lora_cfg = cfg["LORA"]

    os.makedirs(out_dir, exist_ok=True)
    safe_save_config(cfg, out_dir)

    logger.info("Config: model=%s out_dir=%s max_length=%d batch=%d grad_acc=%d epochs=%d 8bit=%s",
                model_name, out_dir, max_length, batch_size, grad_acc, epochs, load_in_8bit)

    # -----------------------
    # Tokenizer
    # -----------------------
    logger.info("Loading tokenizer (trust_remote_code=True, use_fast=True)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    # Ensure pad/eos tokens exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": ""})

    # -----------------------
    # Model load
    # -----------------------
    logger.info("Loading base model (trust_remote_code=True) load_in_8bit=%s ...", load_in_8bit)
    load_kwargs = {"trust_remote_code": True}
    if load_in_8bit:
        load_kwargs.update({"load_in_8bit": True, "device_map": "auto"})
    else:
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            if fp16:
                load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # If tokenizer vocab differs from model embedding rows, resize embeddings
    try:
        tok_size = tokenizer.get_vocab_size()
    except Exception:
        tok_size = getattr(tokenizer, "vocab_size", None)
    model_emb_rows = model.get_input_embeddings().weight.shape[0]
    if tok_size and tok_size != model_emb_rows:
        logger.info("Resizing embeddings to tokenizer size %d (was %d)", tok_size, model_emb_rows)
        resize_model_embeddings(model, tok_size)

    # If loaded in 8-bit, prepare for k-bit training BEFORE LoRA
    if load_in_8bit:
        logger.info("prepare_model_for_kbit_training(model) for k-bit setup")
        model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing and disable use_cache for compatibility
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        # fallback older API variation
        try:
            model.gradient_checkpointing = True
        except Exception:
            pass
    try:
        model.config.use_cache = False
    except Exception:
        pass

    # -----------------------
    # Apply LoRA/PEFT
    # -----------------------
    logger.info("Applying LoRA: r=%s alpha=%s target_modules=%s", lora_cfg["r"], lora_cfg["lora_alpha"], lora_cfg["target_modules"])
    peft_conf = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_conf)

    # Diagnostic: print trainable parameter info
    try:
        model.print_trainable_parameters()
    except Exception:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("Trainable params: %d / %d", trainable, total)

    # -----------------------
    # Build dataset (manual)
    # -----------------------
    data_list = []
    logger.info("Reading input dataset and building examples: %s", args.data)
    with open(args.data, "r", encoding="utf-8") as f:
        for n, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                logger.warning("Skipping invalid JSON line %d: %s", n, e)
                continue

            messages = rec.get("messages") or rec.get("conversation") or (rec.get("example") or {}).get("conversation")
            if not isinstance(messages, list):
                logger.debug("Skipping record %d: no messages list", n)
                continue

            ex = process_example_messages(messages, tokenizer, max_length)
            if ex:
                data_list.append(ex)

    logger.info("Built %d examples (after filtering).", len(data_list))
    if len(data_list) == 0:
        raise RuntimeError("No valid training examples after processing input data.")

    dataset = Dataset.from_list(data_list)
    logger.info("Dataset size: %d", len(dataset))

    # -----------------------
    # Collator & TrainingArguments
    # -----------------------
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = lambda batch: collate_fn(batch, pad_token_id=pad_token_id)

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=fp16,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        # keep gradient_checkpointing behavior as model config already set
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    
    # Sanity forward: validate batch and check loss.requires_grad
    # -----------------------
    logger.info("Performing sanity forward check with a small batch...")
    sample_n = min(2, len(dataset))
    sample_items = [dataset[i] for i in range(sample_n)]
    small_batch = collator(sample_items)

    # Validate and fix small batch (resize embeddings / truncate sequences if needed)
    model_max_pos = getattr(model.config, "max_position_embeddings", None)
    # temporarily enable synchronous CUDA errors to get deterministic tracebacks if something fails
    prev_env = os.environ.get("CUDA_LAUNCH_BLOCKING")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    try:
        small_batch = validate_and_fix_batch(model, small_batch, tokenizer, model_max_pos=model_max_pos)

        model.train()
        with torch.cuda.amp.autocast(enabled=False):
            out = model(**small_batch)
        loss = getattr(out, "loss", None)
        if loss is None:
            raise RuntimeError("Model forward did not return loss on sanity batch.")

        logger.info("Sanity forward: loss dtype=%s requires_grad=%s", loss.dtype, loss.requires_grad)

        if not loss.requires_grad:
            logger.warning("Sanity forward loss.requires_grad False. Attempting to enable LoRA params or all params and retry.")
            # try to enable LoRA-like params
            enabled = 0
            for name, p in model.named_parameters():
                if any(token in name.lower() for token in ("lora", "adapter", "alpha")):
                    p.requires_grad = True
                    enabled += p.numel()
            if enabled == 0:
                logger.warning("No LoRA-like params found; enabling all parameters as fallback.")
                for p in model.parameters():
                    p.requires_grad = True

            # retry forward
            small_batch = {k: v.to(next(model.parameters()).device) for k, v in small_batch.items()}
            model.train()
            with torch.cuda.amp.autocast(enabled=False):
                out2 = model(**small_batch)
            loss2 = getattr(out2, "loss", None)
            if loss2 is None or not getattr(loss2, "requires_grad", False):
                raise RuntimeError("Even after enabling parameters, loss.requires_grad is False. Aborting.")
            logger.info("After enabling grads, loss.requires_grad=%s", loss2.requires_grad)
    finally:
        # restore previous env var
        if prev_env is None:
            os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
        else:
            os.environ["CUDA_LAUNCH_BLOCKING"] = prev_env

    # -----------------------
    # Start training
    # -----------------------
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # -----------------------
    # Save artifacts
    # -----------------------
    logger.info("Saving final artifacts to %s", out_dir)
    try:
        model.save_pretrained(out_dir)
    except Exception:
        logger.exception("model.save_pretrained failed; falling back to trainer.save_model + tokenizer.save_pretrained")
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
    logger.info("Training complete. Artifacts saved at %s", out_dir)


if __name__ == "__main__":
    main()

