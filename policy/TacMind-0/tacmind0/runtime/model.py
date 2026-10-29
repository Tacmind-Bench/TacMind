"""BaseModelConfig: shared TacMind-0 runtime behavior."""

from dataclasses import dataclass
from typing import Literal

import torch
from loguru import logger

from tacmind0.constants.precision import (
    BF16_MIXED_PRECISION_POLICY,
    FP32_MIXED_PRECISION_POLICY,
    MODEL_DTYPE,
)
from tacmind0.model.modeling import (
    TacMind0Config,
    TacMind0ForConditionalGeneration,
)
from tacmind0.model.operations import (
    configure_fp32_precision_runtime,
)


@dataclass
class BaseModelConfig:
    model_name_or_path: str | None = "./checkpoints/TacMind0"
    chunk_size: int = 50
    precision_policy: Literal["bf16_mixed", "fp32_mixed"] = BF16_MIXED_PRECISION_POLICY
    bf16: bool = True
    llm_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = (
        "flex_attention"
    )
    vision_attn_implementation: Literal[
        "auto", "eager", "sdpa", "flash_attention_2"
    ] = "flash_attention_2"
    action_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = (
        "sdpa"
    )
    liger_kernel: bool = True
    freeze_vlm_embedding: bool = True
    vlm_gradient_checkpointing: bool = False
    ae_gradient_checkpointing: bool = False

    def _torch_dtype(self) -> torch.dtype:
        if self.precision_policy == BF16_MIXED_PRECISION_POLICY:
            return torch.bfloat16 if self.bf16 else torch.float32
        if self.precision_policy == FP32_MIXED_PRECISION_POLICY:
            return MODEL_DTYPE
        raise ValueError(f"Invalid precision_policy: {self.precision_policy!r}")

    def _load_base_checkpoint_model(self) -> TacMind0ForConditionalGeneration:
        config = TacMind0Config.from_pretrained(self.model_name_or_path)
        config.chunk_size = self.chunk_size
        return TacMind0ForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            config=config,
            dtype=self._torch_dtype(),
        )

    def _apply_precision_policy(self, model: torch.nn.Module) -> None:
        if self.precision_policy == BF16_MIXED_PRECISION_POLICY:
            return

        if self.precision_policy == FP32_MIXED_PRECISION_POLICY:
            model.to(dtype=MODEL_DTYPE)
            configure_fp32_precision_runtime()
            return

        raise ValueError(f"Invalid precision_policy: {self.precision_policy!r}")

    def _apply_runtime_model_options(
        self,
        model: torch.nn.Module,
    ) -> None:
        model.set_attention_implementation(
            llm_attn_implementation=self.llm_attn_implementation,
            vision_attn_implementation=self.vision_attn_implementation,
            action_attn_implementation=self.action_attn_implementation,
            bf16=self.bf16,
        )
        model.enable_gradient_checkpointing(
            vlm_gradient_checkpointing=self.vlm_gradient_checkpointing,
            ae_gradient_checkpointing=self.ae_gradient_checkpointing,
        )
        if self.freeze_vlm_embedding:
            model.freeze_vlm_embedding()

    def _apply_full_model_runtime_optimizations(
        self, model: TacMind0ForConditionalGeneration
    ) -> None:
        if self.liger_kernel and torch.cuda.is_available():
            model._apply_liger_kernel()
        elif self.liger_kernel:
            logger.info("CUDA unavailable; skipping Liger Kernel for TacMind0 model.")

    def build_model(self) -> TacMind0ForConditionalGeneration:
        model = self._load_base_checkpoint_model()
        model.set_precision_policy(self.precision_policy)
        self._apply_runtime_model_options(model)
        self._apply_full_model_runtime_optimizations(model)
        self._apply_precision_policy(model)
        return model
