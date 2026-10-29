"""Staged recipe defaults, independent of model dependencies."""

from policy.config import resolve

DEFAULTS = {
    "stage": "all",
    "tactile_steps": 10000,
    "tactile_batch_size": 16,
    "tactile_lr": 1e-5,
    "train_scope": "encoder_only",
    "validation_fraction": 0.1,
    "validation_batches": 100,
    "sigreg_weight": 0.09,
    "frs_steps": 10000,
    "frs_batch_size": 2,
    "frs_lr": 1e-4,
    "diffusion_steps": 10,
    "run_frs": True,
    "marker_cache_size": 256,
    "action_backend": "tensorrt",
    "cuda_graphs": True,
}
PATHS = {"resources", "world_weights", "tactile_checkpoint"}


def normalize_model(model):
    model = {**DEFAULTS, **model}
    for key in PATHS & model.keys():
        model[key] = str(resolve(model[key]))
    return model
