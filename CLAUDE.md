# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch training framework built around a **registry-based plugin architecture**. Supports classification, segmentation, and object detection (DETR-family) tasks. The entire pipeline — models, datasets, transforms, losses, optimizers, schedulers, and training tasks — is assembled dynamically from YAML config files.

## Common Commands

```bash
# Train (single GPU)
python tools/train.py --config config/yamls/resnet18_train.yaml --epochs 50 --save_dir ./checkpoints/exp1

# Train (multi-GPU DDP via torchrun)
torchrun --nproc_per_node=4 tools/train.py --config config/yamls/dinov3_det.yaml --epochs 50

# List all registered components
python scripts/list_available.py --models --datasets --losses --optimizers --schedulers

# Regenerate layers/__init__.py after adding a new layer module
python scripts/update_layer_init.py
```

Tests are standalone scripts in `test/` — run them directly with `python test/test_models.py`, etc.

## Architecture: Registry Pattern

Every pluggable component uses the same pattern from `utils/registry.py`:

1. A `Registry` instance is created in a `_xxxRegistry.py` file (e.g., `models/_modelRegistry.py` creates `MODEL_REGISTRY`)
2. Components are registered via `@register_xxx("name")` decorator
3. An `auto_scan_and_import()` call in each package's `__init__.py` scans the directory and imports all non-underscore-prefixed `.py` files, triggering their `@register` decorators
4. Factory functions like `build_model("resnet18", num_classes=10)` resolve the name and instantiate the class

**Six registries exist**: models, datasets, transforms, losses, optimizers/schedulers, and training tasks.

### Adding a New Component

1. Create a `.py` file in the appropriate package (e.g., `models/my_model.py`)
2. Use the package's register decorator (e.g., `@register_model("my_model")`)
3. List public symbols in `__all__`
4. For **layers/** only: run `python scripts/update_layer_init.py` to regenerate the auto-generated `__init__.py`

Files prefixed with `_` (registries, `__init__`) are excluded from auto-import scanning.

## Config-Driven Training Pipeline

YAML configs (in `config/yamls/`) define the entire training run. Keys: `task`, `data`, `model`, `loss`, `optim`, `scheduler`. The entry point `tools/train.py` reads the YAML and assembles all components via registry factory functions.

- **`task`**: selects the training/validation engine (e.g., `"train_classification"`, `"train_detection"`)
- **`data.train_transforms` / `val_transforms`**: ordered list of transform names + kwargs, including branching logic (`random_select`)
- **`model`**: `name` resolves via model registry; `kwargs` forwarded. Special case: `"dinov3_det"` takes a `backbone_name` key to separately build the backbone
- **`data.train_dataset` / `val_dataset`**: `name` resolves via dataset registry

### Dynamic DAG Model

`models/dynamic_model.py` (`DynamicGraphModel`) builds a network from a `layers_cfg` list in YAML. Each entry specifies a layer `name` (resolved from `layers/` or `torch.nn`), `args`, and `from` (routing: `"input"`, `-1` for previous layer, or a list of indices for multi-input layers like skip connections). See `config/model/dynamic_model.yaml` for an example.

## Training Engine

- `engine/train_engine.py` — `BaseTrainer` (abstract): implements the epoch loop, checkpoint saving (best metric), TensorBoard logging, DDP device handling. Subclasses must implement `train_step(batch)` → `(loss, log_dict)` and `evaluate()` → `(main_metric, log_dict)`
- `engine/val_engine.py` — `BaseValidator` (abstract): standalone validation with its own `evaluate()`
- `engine/tasks.py` — concrete task trainers: `ClassificationTrainer`, `DetectionTrainer`, and their validators. Uses composition (validator is a member, not inheritance)
- Best model is saved when `main_metric > best_metric` (higher is better). Detection validator returns `-avg_loss` to invert the loss-minimization direction

## Key Conventions

- All registry names are **case-insensitive** (lowercased internally)
- Models should extend `BaseModel` and accept `**kwargs` (unused kwargs trigger a warning)
- Task names follow the pattern `"train_<task>"` and `"val_<task>"`
- The project uses Chinese comments extensively — maintain this style when modifying existing code
- `layers/__init__.py` is **auto-generated** — never edit it manually; run `scripts/update_layer_init.py` after adding layers
