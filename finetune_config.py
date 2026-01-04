# finetune_config.py
"""
Configuración centralizada para el pipeline de fine-tuning (LoRA/QLoRA) — listo para usar.

Propósito:
- Centralizar hiperparámetros, rutas y opciones de entrenamiento.
- Permitir sobreescritura por variables de entorno o por fichero JSON/YAML.
- Proveer validación mínima y helpers para crear directorios.

Cómo usar:
    from finetune_config import get_config, save_config
    cfg = get_config()
    print(cfg.BASE_MODEL, cfg.OUTPUT_DIR)
    # Pasar cfg a tu script de entrenamiento finetune.py

Notas:
- Este archivo no ejecuta el entrenamiento, sólo define la configuración.
- Diseñado para usarse con transformers + peft (LoRA / QLoRA). Ajusta los nombres
  de los módulos objetivo (target_modules) según la arquitectura del modelo base.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


# -----------------------------
# Helpers
# -----------------------------
def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v is not None else default


def _mkdirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Dataclasses de configuración
# -----------------------------
@dataclass
class LoRAConfig:
    # Parámetros LoRA (PEFT)
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # "none" | "all" | "lora_only"
    task_type: str = "CAUSAL_LM"  # Typically "CAUSAL_LM" for decoder-only models

    def as_peft_dict(self) -> Dict[str, Any]:
        """
        Representación útil para pasar a PEFT/Trainer wrappers.
        """
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }


@dataclass
class TrainingArgs:
    # Hyperparameters de entrenamiento
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False  # use bf16 if supported (Ampere+)
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: Optional[int] = 500
    save_total_limit: int = 5
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"  # 'linear','cosine','cosine_with_restarts',...
    max_steps: Optional[int] = None
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataConfig:
    # Rutas a datasets y opciones
    train_file: Optional[str] = "data/datasets/chat_finetune.jsonl"
    validation_file: Optional[str] = "data/datasets/chat_finetune.val.jsonl"
    text_column: Optional[str] = None  # si usas datasets library y columnas nombradas
    max_seq_length: int = 1024
    preprocessing_num_workers: int = 4
    pad_to_max_length: bool = False


@dataclass
class CheckpointingConfig:
    output_dir_root: str = "models"
    project_name: str = "finetune"
    timestamped: bool = True

    def make_output_dir(self, suffix: Optional[str] = None) -> str:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") if self.timestamped else ""
        parts = [self.output_dir_root, self.project_name]
        if ts:
            parts.append(ts)
        if suffix:
            parts.append(suffix)
        out = os.path.join(*parts)
        _mkdirs(out)
        return out


@dataclass
class FinetuneConfig:
    # Model & runtime
    BASE_MODEL: str = field(default_factory=lambda: _env("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    OUTPUT_DIR: str = field(default_factory=lambda: _env("OUTPUT_DIR", "models/qwen-rag-ft"))
    DEVICE: str = field(default_factory=lambda: _env("TRAIN_DEVICE", "cuda"))  # "cuda" or "cpu"
    LOAD_IN_8BIT: bool = False  # use bitsandbytes 8-bit loading if supported
    use_peft: bool = True  # whether to apply LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingArgs = field(default_factory=TrainingArgs)
    data: DataConfig = field(default_factory=DataConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)

    # Misc
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    save_lora_only: bool = True  # save only LoRA adapters instead of full model
    local_rank: Optional[int] = None
    ddp: bool = False

    def validate(self) -> None:
        """
        Validaciones sencillas para detectar errores de configuración comunes.
        """
        if self.training.fp16 and self.training.bf16:
            raise ValueError("fp16 and bf16 cannot both be True.")
        if self.data.max_seq_length <= 0:
            raise ValueError("data.max_seq_length must be > 0.")
        if self.checkpointing.output_dir_root is None:
            raise ValueError("checkpointing.output_dir_root must be set.")
        # create output dir
        _mkdirs(self.OUTPUT_DIR)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # convert nested dataclasses to dicts where needed
        d["lora"] = self.lora.as_peft_dict()
        d["training"] = self.training.to_dict()
        d["data"] = asdict(self.data)
        d["checkpointing"] = asdict(self.checkpointing)
        return d


# -----------------------------
# Convenience functions
# -----------------------------
_default_config: Optional[FinetuneConfig] = None


def get_config(overrides: Optional[Dict[str, Any]] = None) -> FinetuneConfig:
    """
    Returns a FinetuneConfig instance.
    Optionally pass overrides as a dict to change runtime parameters.
    """
    global _default_config
    if _default_config is None:
        cfg = FinetuneConfig()
        # ensure output dir exists
        cfg.validate()
        _default_config = cfg
    else:
        _default_config = _default_config  # reuse

    if overrides:
        # shallow update; for nested updates provide nested dicts explicitly
        d = _default_config.to_dict()
        d.update(overrides)
        # reconstruct dataclasses from dict (simple approach)
        cfg = FinetuneConfig(
            BASE_MODEL=d.get("BASE_MODEL", _default_config.BASE_MODEL),
            OUTPUT_DIR=d.get("OUTPUT_DIR", _default_config.OUTPUT_DIR),
            DEVICE=d.get("DEVICE", _default_config.DEVICE),
            LOAD_IN_8BIT=d.get("LOAD_IN_8BIT", _default_config.LOAD_IN_8BIT),
            use_peft=d.get("use_peft", _default_config.use_peft),
            lora=LoRAConfig(**(d.get("lora") or _default_config.lora.as_peft_dict())),
            training=TrainingArgs(**(d.get("training") or _default_config.training.to_dict())),
            data=DataConfig(**(d.get("data") or asdict(_default_config.data))),
            checkpointing=CheckpointingConfig(**(d.get("checkpointing") or asdict(_default_config.checkpointing))),
            push_to_hub=d.get("push_to_hub", _default_config.push_to_hub),
            hub_model_id=d.get("hub_model_id", _default_config.hub_model_id),
            save_lora_only=d.get("save_lora_only", _default_config.save_lora_only),
            local_rank=d.get("local_rank", _default_config.local_rank),
            ddp=d.get("ddp", _default_config.ddp),
        )
        cfg.validate()
        return cfg
    return _default_config


def save_config(cfg: FinetuneConfig, path: str) -> None:
    """
    Save configuration to JSON (human-readable).
    """
    out = cfg.to_dict()
    _mkdirs(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# -----------------------------
# Quick demo / sanity
# -----------------------------
if __name__ == "__main__":
    cfg = get_config()
    print("Finetune configuration (summary):")
    summary = cfg.to_dict()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # create the output dir (timestamped)
    ts_dir = cfg.checkpointing.make_output_dir(suffix="run")
    print("Timestamped output dir created:", ts_dir)
    save_path = os.path.join(ts_dir, "finetune_config.json")
    save_config(cfg, save_path)
    print("Saved config to", save_path)
