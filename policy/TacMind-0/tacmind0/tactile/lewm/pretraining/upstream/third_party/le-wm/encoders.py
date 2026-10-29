import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


class DinoV2WithRegistersEncoder(nn.Module):
    """Adapter around DINOv2-with-registers models that exposes ViT-style tokens.

    The wrapped DINOv2 model may internally use register tokens, but the
    returned `last_hidden_state` keeps only `[CLS] + patch_tokens` so existing
    downstream code can continue to interpret `[:, 1:]` as pure patch tokens.
    """

    def __init__(
        self,
        backend: str = "huggingface",
        model_name: str = "dinov2_vits14_reg",
        repo_or_dir: str = "facebookresearch/dinov2",
        model_id: str = "facebook/dinov2-with-registers-small",
        pretrained: bool = True,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        disable_proxy: bool = False,
        freeze_backbone: bool = True,
        freeze_mask_token: bool = True,
    ):
        super().__init__()
        self.backend = backend
        self.model_name = model_name
        self.repo_or_dir = repo_or_dir
        self.model_id = model_id
        self.pretrained = pretrained
        self.cache_dir = str(Path(cache_dir).expanduser()) if cache_dir else None
        self.local_files_only = local_files_only
        self.disable_proxy = disable_proxy
        self.freeze_backbone = freeze_backbone
        self.freeze_mask_token = freeze_mask_token

        if backend == "huggingface":
            self.model = self._load_huggingface()
        elif backend == "torch_hub":
            self.model = self._load_torch_hub()
        else:
            raise ValueError(
                f"Unsupported DINOv2 backend '{backend}'. "
                "Expected 'huggingface' or 'torch_hub'."
            )

        if self.freeze_backbone:
            self.model.requires_grad_(False)
            self.model.eval()

        # This training pipeline never supplies bool_masked_pos, so DINOv2's
        # pretraining-only mask token cannot contribute to the loss. Leaving it
        # trainable causes multi-GPU DDP to fail on the second iteration.
        if self.freeze_mask_token:
            embeddings = getattr(self.model, "embeddings", None)
            mask_token = getattr(embeddings, "mask_token", None)
            if mask_token is not None:
                mask_token.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self

    @contextmanager
    def _network_environment(self):
        if not self.disable_proxy:
            yield
            return

        proxy_keys = (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        )
        saved = {key: os.environ.pop(key) for key in proxy_keys if key in os.environ}
        try:
            yield
        finally:
            os.environ.update(saved)

    def _load_huggingface(self):
        from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

        if self.pretrained:
            try:
                with self._network_environment():
                    return Dinov2WithRegistersModel.from_pretrained(
                        self.model_id,
                        cache_dir=self.cache_dir,
                        local_files_only=self.local_files_only,
                    )
            except OSError as exc:
                raise RuntimeError(
                    f"Could not load DINOv2 weights '{self.model_id}'. "
                    "Set HF_ENDPOINT to an accessible Hugging Face mirror, or "
                    "pre-download the model and set encoder.model_id to its local path."
                ) from exc

        with self._network_environment():
            config = Dinov2WithRegistersConfig.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
        return Dinov2WithRegistersModel(config)

    def _load_torch_hub(self):
        try:
            return torch.hub.load(
                self.repo_or_dir,
                self.model_name,
                pretrained=self.pretrained,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Could not load DINOv2 through Torch Hub. The GitHub repository "
                "must be reachable or already present in the Torch Hub cache. "
                "Use backend='huggingface' on restricted DLC nodes."
            ) from exc

    def _forward_model(self, pixels):
        if self.backend == "huggingface":
            features = self.model(pixel_values=pixels)
            hidden = features.last_hidden_state
            num_register_tokens = self.model.config.num_register_tokens
            cls_token = hidden[:, :1]
            register_tokens = hidden[:, 1 : 1 + num_register_tokens]
            patch_tokens = hidden[:, 1 + num_register_tokens :]
        else:
            features = self.model.forward_features(pixels)
            cls_token = features["x_norm_clstoken"].unsqueeze(1)
            register_tokens = features.get("x_norm_regtokens")
            patch_tokens = features["x_norm_patchtokens"]

        last_hidden_state = torch.cat([cls_token, patch_tokens], dim=1)
        output = SimpleNamespace(last_hidden_state=last_hidden_state)
        if register_tokens is not None:
            output.register_tokens = register_tokens
        return output

    def forward(self, pixels, interpolate_pos_encoding: bool = True):
        del interpolate_pos_encoding

        if self.freeze_backbone:
            self.model.eval()
            with torch.no_grad():
                return self._forward_model(pixels)

        return self._forward_model(pixels)
