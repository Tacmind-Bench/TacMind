"""TacMind0 model architecture built on a Gemma3 VLM and Action Expert."""

import logging
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.cache_utils import Cache
from transformers.models.gemma3.configuration_gemma3 import (
    Gemma3Config,
    Gemma3TextConfig,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3CausalLMOutputWithPast,
    Gemma3DecoderLayer,
    Gemma3ForConditionalGeneration,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
    Gemma3TextModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

from tacmind0.constants.precision import (
    BF16_MIXED_PRECISION_POLICY,
    FP32_MIXED_PRECISION_POLICY,
    MODEL_DTYPE,
)
from tacmind0.constants.robot import HISTORY_POOL_SIZE
from tacmind0.constants.tactile import TACTILE_TOKEN_ID
from tacmind0.model.base import (
    BasePolicyConfig,
    BasePolicyModel,
)
from tacmind0.model.operations import (
    HISTORY_PAD_TOKEN_ID,
    VLADynamicCache,
    flow_euler_schedule,
    flow_euler_timesteps,
    is_flash_attention_2_available,
    is_flex_attention_available,
    linear_fp32,
    make_suffix_attn_mask,
    mask_history_pad_tokens_in_attention,
    patch_decoder_layers,
    posemb_sincos,
    prepare_action_chunk,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TacMind0Config(BasePolicyConfig):
    """TacMind0 model configuration."""

    model_type = "tacmind0"
    tie_word_embeddings: bool = True

    def __init__(
        self,
        vlm_config=None,
        action_config=None,
        action_dim: int = 32,
        chunk_size: int = 50,
        tie_word_embeddings: bool = True,
        tactile_config: dict | None = None,
        **kwargs,
    ):
        if isinstance(vlm_config, dict):
            vlm_config = Gemma3Config(**vlm_config)
        if isinstance(action_config, dict):
            action_config = Gemma3TextConfig(**action_config)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vlm_config = vlm_config
        self.action_config = action_config
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.tactile_config = tactile_config
        self.tie_word_embeddings = tie_word_embeddings


class TacMind0ActionExpert(Gemma3TextModel):
    """Gemma3 action expert with TacMind0 suffix-only forward.

    The module takes ownership of the original text model's layers/norm so
    parameter names stay under ``model.action_expert.layers.*`` after wrapping.
    """

    def __init__(
        self,
        config: Gemma3TextConfig,
    ):
        super(Gemma3TextModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = None
        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config)
        self.gradient_checkpointing = False
        hidden_size = config.hidden_size
        self.input_time_modulators = nn.ModuleList(
            nn.Linear(hidden_size, 3 * hidden_size) for _ in self.layers
        )
        self.mlp_time_modulators = nn.ModuleList(
            nn.Linear(hidden_size, 3 * hidden_size) for _ in self.layers
        )
        self.final_time_modulator = nn.Linear(hidden_size, 3 * hidden_size)
        self.precision_policy = BF16_MIXED_PRECISION_POLICY
        self.post_init()
        self._init_time_modulators()
        self.set_action_attention_backend("eager")

    def _init_time_modulators(self) -> None:
        std = float(getattr(self.config, "initializer_range", 0.02))
        modulators = (
            list(self.input_time_modulators)
            + list(self.mlp_time_modulators)
            + [self.final_time_modulator]
        )
        for modulator in modulators:
            nn.init.normal_(modulator.weight, mean=0.0, std=std)
            if modulator.bias is not None:
                nn.init.zeros_(modulator.bias)

    def set_action_attention_backend(self, action_attn_implementation: str) -> None:
        if action_attn_implementation == "eager":
            self._suffix_attn_backend = "eager"
            self._attn_fn = self._eager_attention
        elif action_attn_implementation == "sdpa":
            self._suffix_attn_backend = "sdpa"
            self._attn_fn = self._sdpa_attention
        elif action_attn_implementation == "flex_attention":
            self._suffix_attn_backend = "flex_attention"
            self._attn_fn = self._flex_attention
        else:
            raise ValueError(
                "Unsupported suffix attention implementation: "
                f"{action_attn_implementation!r}"
            )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    @staticmethod
    def _sdpa_attention(query_states, key_states, value_states, attention_mask, layer):
        kv_states_k = repeat_kv(key_states, layer.self_attn.num_key_value_groups)
        kv_states_v = repeat_kv(value_states, layer.self_attn.num_key_value_groups)
        bool_mask = attention_mask
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            bool_mask = attention_mask == 0
        attn_output = F.scaled_dot_product_attention(
            query_states,
            kv_states_k,
            kv_states_v,
            attn_mask=bool_mask,
            scale=layer.self_attn.scaling,
        )
        return attn_output.transpose(1, 2).contiguous()

    @staticmethod
    def _eager_attention(query_states, key_states, value_states, attention_mask, layer):
        attn_output, _ = eager_attention_forward(
            layer.self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=layer.self_attn.scaling,
        )
        return attn_output

    @staticmethod
    def _flex_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        layer,
    ):
        from transformers.integrations.flex_attention import flex_attention_forward

        attn_output, _ = flex_attention_forward(
            layer.self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=layer.self_attn.scaling,
        )
        return attn_output

    def forward(
        self,
        *,
        suffix_embeds: torch.Tensor | None = None,
        attention_mask: Any = None,
        position_ids: torch.LongTensor | None = None,
        prefix_cache_keys: tuple[torch.Tensor, ...] | None = None,
        prefix_cache_values: tuple[torch.Tensor, ...] | None = None,
        adarms_cond: torch.Tensor | None = None,
        tactile_modulations: torch.Tensor | None = None,
        rotary_cache: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if suffix_embeds is None:
            raise RuntimeError(
                "TacMind0ActionExpert only supports suffix forward. "
                "Pass suffix_embeds, position_ids, prefix_cache_keys, and "
                "prefix_cache_values."
            )
        if position_ids is None:
            raise ValueError(
                "position_ids is required for TacMind0ActionExpert suffix forward."
            )
        if prefix_cache_keys is None or prefix_cache_values is None:
            raise ValueError(
                "prefix_cache_keys and prefix_cache_values are required for "
                "TacMind0ActionExpert suffix forward."
            )
        if adarms_cond is None:
            raise ValueError(
                "adarms_cond is required for TacMind0ActionExpert suffix forward."
            )
        return self._suffix_forward_layers(
            suffix_embeds=suffix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prefix_cache_keys=prefix_cache_keys,
            prefix_cache_values=prefix_cache_values,
            adarms_cond=adarms_cond,
            tactile_modulations=tactile_modulations,
            rotary_cache=rotary_cache,
        )

    def _suffix_forward_layers(
        self,
        suffix_embeds: torch.Tensor,
        attention_mask: Any,
        position_ids: torch.LongTensor,
        prefix_cache_keys: tuple[torch.Tensor, ...],
        prefix_cache_values: tuple[torch.Tensor, ...],
        adarms_cond: torch.Tensor,
        tactile_modulations: torch.Tensor | None = None,
        rotary_cache: dict | None = None,
    ) -> torch.Tensor:
        hidden_states = suffix_embeds
        num_layers = len(self.layers)
        # Flow integration supplies a cache for its fixed positions. Standalone
        # forwards own a fresh cache, so unrelated requests remain isolated.
        if rotary_cache is None:
            rotary_cache = {}
        if (
            self._suffix_attn_backend == "sdpa"
            and attention_mask is not None
            and attention_mask.dtype != torch.bool
        ):
            attention_mask = attention_mask == 0

        for layer_idx in range(num_layers):
            layer_cache_keys = prefix_cache_keys[layer_idx]
            layer_cache_values = prefix_cache_values[layer_idx]
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self._compute_suffix_layer,
                    layer_idx,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    adarms_cond,
                    layer_cache_keys,
                    layer_cache_values,
                    None if tactile_modulations is None else tactile_modulations[layer_idx],
                    rotary_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states = self._compute_suffix_layer(
                    layer_idx,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    adarms_cond,
                    layer_cache_keys,
                    layer_cache_values,
                    None if tactile_modulations is None else tactile_modulations[layer_idx],
                    rotary_cache,
                )

        hidden_states, _ = self._adaptive_rmsnorm(
            self.norm,
            hidden_states,
            adarms_cond,
            self.final_time_modulator,
        )
        return hidden_states

    def _adaptive_rmsnorm(
        self,
        norm: nn.Module,
        x: torch.Tensor,
        adarms_cond: torch.Tensor,
        modulator: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        eps = getattr(norm, "eps", None)
        if eps is None:
            eps = norm.variance_epsilon
        normed = x * torch.rsqrt(var + eps)

        if self.precision_policy == FP32_MIXED_PRECISION_POLICY:
            modulation = linear_fp32(adarms_cond, modulator)
        else:
            modulation = modulator(adarms_cond.to(dtype=modulator.weight.dtype))
        if modulation.ndim == 2:
            modulation = modulation[:, None, :]
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)

    def _compute_suffix_layer(
        self,
        layer_idx: int,
        suffix_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Any,
        adarms_cond: torch.Tensor,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
        tactile_modulation: torch.Tensor | None = None,
        rotary_cache: dict | None = None,
    ) -> torch.Tensor:
        ae_layer = self.layers[layer_idx]
        layer_type = ae_layer.self_attn.layer_type

        prenorm, attn_gate = self._adaptive_rmsnorm(
            ae_layer.input_layernorm,
            suffix_embeds,
            adarms_cond,
            self.input_time_modulators[layer_idx],
        )
        if tactile_modulation is not None:
            scale_a, shift_a, scale_m, shift_m = tactile_modulation.chunk(4, dim=-1)
            prenorm = (prenorm.float() * (1 + scale_a) + shift_a).to(prenorm.dtype)
        batch_size, seq_len, _ = prenorm.shape
        hidden_shape = (batch_size, seq_len, -1, ae_layer.self_attn.head_dim)

        query_states = (
            ae_layer.self_attn.q_proj(prenorm).view(hidden_shape).transpose(1, 2)
        )
        key_states = (
            ae_layer.self_attn.k_proj(prenorm).view(hidden_shape).transpose(1, 2)
        )
        value_states = (
            ae_layer.self_attn.v_proj(prenorm).view(hidden_shape).transpose(1, 2)
        )

        query_states = ae_layer.self_attn.q_norm(query_states)
        key_states = ae_layer.self_attn.k_norm(key_states)

        rotary_key = (layer_type, query_states.dtype)
        if rotary_cache is None:
            cos, sin = self.rotary_emb(query_states, position_ids, layer_type=layer_type)
        else:
            if rotary_key not in rotary_cache:
                rotary_cache[rotary_key] = self.rotary_emb(
                    query_states, position_ids, layer_type=layer_type
                )
            cos, sin = rotary_cache[rotary_key]
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_states = torch.cat([cache_keys, key_states], dim=2)
        value_states = torch.cat([cache_values, value_states], dim=2)
        if (
            attention_mask is not None
            and attention_mask.shape[-1] != key_states.shape[-2]
        ):
            attention_mask = attention_mask[..., -key_states.shape[-2] :]

        attn_output = self._attn_fn(
            query_states, key_states, value_states, attention_mask, ae_layer
        )
        attn_output = attn_output.view(batch_size, seq_len, -1)

        attn_embeds = ae_layer.self_attn.o_proj(attn_output)
        attn_embeds = ae_layer.post_attention_layernorm(attn_embeds)
        residual = suffix_embeds + attn_embeds * attn_gate

        mlp_input, mlp_gate = self._adaptive_rmsnorm(
            ae_layer.pre_feedforward_layernorm,
            residual,
            adarms_cond,
            self.mlp_time_modulators[layer_idx],
        )
        if tactile_modulation is not None:
            mlp_input = (mlp_input.float() * (1 + scale_m) + shift_m).to(mlp_input.dtype)
        mlp_out = ae_layer.mlp(mlp_input)
        mlp_out = ae_layer.post_feedforward_layernorm(mlp_out)
        return residual + mlp_out * mlp_gate


# ---------------------------------------------------------------------------
# Model Outputs
# ---------------------------------------------------------------------------


@dataclass
class TacMind0OutputWithPast(Gemma3CausalLMOutputWithPast):
    fm_loss: torch.FloatTensor | None = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TacMind0Model(BasePolicyModel):
    """Core TacMind0 model.

    The model owns the Gemma3 VLM, the action expert, and the projection layers
    used to bridge action vectors to the action expert hidden size.
    - ``action_in_proj`` / ``action_out_proj`` for action-dimension and action
      expert hidden-size projection.
    - ``time_mlp_in`` / ``time_mlp_out`` for action expert AdaRMSNorm time
      conditioning.
    """

    def __init__(self, config: TacMind0Config):
        super().__init__(config)

        vlm_config_ref = config.vlm_config
        action_config_ref = config.action_config

        logger.info(f"Final TacMind0Config: {config}")

        # Initialize the VLM.
        self.vlm = Gemma3ForConditionalGeneration(vlm_config_ref)

        # Initialize the action expert (Gemma3TextModel without lm_head), preserving suffix-only semantics.
        self.action_expert = TacMind0ActionExpert(action_config_ref)

        ae_hidden_size = action_config_ref.hidden_size

        # Action projection layers.
        self.action_in_proj = nn.Linear(config.action_dim, ae_hidden_size)
        self.action_out_proj = nn.Linear(ae_hidden_size, config.action_dim)

        # Time-conditioning layers.
        self.time_mlp_in = nn.Linear(ae_hidden_size, ae_hidden_size)
        self.time_mlp_out = nn.Linear(ae_hidden_size, ae_hidden_size)

    @property
    def visual(self):
        """Return the VLM vision module; Gemma3 uses ``vision_tower``."""
        return self.vlm.model.vision_tower

    @property
    def language_model(self):
        return self.vlm.model.language_model

    @property
    def lm_head(self):
        return self.vlm.lm_head

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.vlm.set_input_embeddings(value)

# ---------------------------------------------------------------------------
# Action policy
# ---------------------------------------------------------------------------


class TacMind0ForConditionalGeneration(BasePolicyModel):
    """Full TacMind0 model with training and action inference support.

    Methods:
    - ``forward()``: training with flow matching.
    - ``inference_action()``: Euler sampling; optional Flow Reversal Steering
      when ``reference_action`` is provided.
    """

    config_class = TacMind0Config
    _FSDP_VLM_LAYERS_TO_WRAP = 24
    _tied_weights_keys = {
        "model.vlm.lm_head.weight": "model.vlm.model.language_model.embed_tokens.weight",
    }

    def __init__(self, config: TacMind0Config):
        super().__init__(config)
        config.model_type = self.config_class.model_type
        self.all_tied_weights_keys = dict(getattr(self, "_tied_weights_keys", {}))
        self.precision_policy = getattr(
            config,
            "precision_policy",
            BF16_MIXED_PRECISION_POLICY,
        )
        self._real_init(config)
        self.set_precision_policy(self.precision_policy)
        saved_action_backend = getattr(config, "tac_frs_action_attention_backend", None)
        if saved_action_backend is not None:
            self.model.action_expert.set_action_attention_backend(saved_action_backend)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        result = super().from_pretrained(*args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        if getattr(model.config, "tactile_config", None) is not None:
            model.tactile_encoder.rebuild_position_buffers()
            model.tactile_encoder.requires_grad_(False)
            model.tactile_encoder.eval()
        return result

    def _real_init(self, config: TacMind0Config):
        if config.tactile_config is not None:
            from tacmind0.policy.contracts import validate_contract

            validate_contract(config.tactile_config)
        self.model = TacMind0Model(config)
        if config.tactile_config is not None:
            from tacmind0.tactile.encoder import FrozenTactileEncoder

            self.tactile_encoder = FrozenTactileEncoder(
                config.tactile_config["backbone"]
            )
            self.tactile_projector = nn.Linear(
                384, config.vlm_config.text_config.hidden_size
            )
        self.post_init()

        # Patch decoder layers to safely handle KV cache during checkpointing.
        patch_decoder_layers(self.model.vlm.model)

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_action_flow_graphs", None)
        return state

    def set_precision_policy(self, precision_policy: str) -> None:
        if precision_policy not in (
            BF16_MIXED_PRECISION_POLICY,
            FP32_MIXED_PRECISION_POLICY,
        ):
            raise ValueError(f"Invalid precision_policy: {precision_policy!r}")
        self.precision_policy = precision_policy
        self.config.precision_policy = precision_policy
        self.model.action_expert.precision_policy = precision_policy

    def fsdp_wrap_modules(self) -> list[nn.Module]:
        """Return the fixed child-unit boundaries for TacMind0 FSDP wrapping."""
        vlm_layers = self.model.vlm.model.language_model.layers
        if len(vlm_layers) < self._FSDP_VLM_LAYERS_TO_WRAP:
            raise ValueError(
                "TacMind0 FSDP requires at least "
                f"{self._FSDP_VLM_LAYERS_TO_WRAP} VLM layers, got {len(vlm_layers)}"
            )
        return [
            self.model.action_expert,
            *vlm_layers[-self._FSDP_VLM_LAYERS_TO_WRAP :],
        ]

    def freeze_vlm_embedding(self):
        """Freeze the Gemma VLM token embedding layer.

        Because ``lm_head.weight`` and ``embed_tokens.weight`` are tied through
        ``_tied_weights_keys`` as the same tensor, ``lm_head`` is frozen as
        well.
        """
        embed = self.model.vlm.model.language_model.embed_tokens
        for p in embed.parameters():
            p.requires_grad = False
        logger.info("Frozen VLM token embedding (and tied lm_head).")

    def _resolve_llm_attn_implementation(
        self,
        requested: Literal["auto", "eager", "sdpa", "flex_attention"],
    ) -> str:
        if requested == "auto":
            return "flex_attention" if is_flex_attention_available() else "sdpa"

        if requested == "flex_attention":
            if not is_flex_attention_available():
                raise ImportError(
                    "flex_attention requires PyTorch FlexAttention support "
                    "(for example torch>=2.5 with torch.nn.attention.flex_attention)"
                )
            return requested

        return requested

    def _resolve_vision_attn_implementation(
        self,
        requested: Literal["auto", "eager", "sdpa", "flash_attention_2"],
        *,
        bf16: bool,
    ) -> str:
        if requested == "auto":
            if (
                self.precision_policy == FP32_MIXED_PRECISION_POLICY or bf16
            ) and is_flash_attention_2_available():
                return "flash_attention_2"
            return "sdpa"

        if requested == "flash_attention_2":
            if self.precision_policy != FP32_MIXED_PRECISION_POLICY and not bf16:
                raise ValueError("flash_attention_2 requires bf16=True")
            if not torch.cuda.is_available():
                raise RuntimeError("flash_attention_2 requires CUDA")
            if not is_flash_attention_2_available():
                raise ImportError("flash_attention_2 requires the flash_attn package")
            return requested

        return requested

    def _resolve_action_attn_implementation(
        self,
        requested: Literal["auto", "eager", "sdpa", "flex_attention"],
    ) -> str:
        if requested == "auto":
            return "flex_attention" if is_flex_attention_available() else "sdpa"

        if requested == "flex_attention":
            if not is_flex_attention_available():
                raise ImportError(
                    "flex_attention requires PyTorch FlexAttention support "
                    "(for example torch>=2.5 with torch.nn.attention.flex_attention)"
                )
            return requested

        return requested

    def set_attention_implementation(
        self,
        *,
        llm_attn_implementation: Literal[
            "auto", "eager", "sdpa", "flex_attention"
        ] = "auto",
        vision_attn_implementation: Literal[
            "auto", "eager", "sdpa", "flash_attention_2"
        ] = "auto",
        action_attn_implementation: Literal[
            "auto", "eager", "sdpa", "flex_attention"
        ] = "auto",
        bf16: bool = True,
    ) -> None:
        """Configure runtime VLM and action expert attention backends."""
        llm_attn_valid = llm_attn_implementation in {
            "auto",
            "eager",
            "sdpa",
            "flex_attention",
        }
        vision_attn_valid = vision_attn_implementation in {
            "auto",
            "eager",
            "sdpa",
            "flash_attention_2",
        }
        action_attn_valid = action_attn_implementation in {
            "auto",
            "eager",
            "sdpa",
            "flex_attention",
        }
        assert llm_attn_valid and vision_attn_valid and action_attn_valid, (
            "Invalid TacMind0 attention implementation: "
            f"llm={llm_attn_implementation!r}, "
            f"vision={vision_attn_implementation!r}, "
            f"action={action_attn_implementation!r}"
        )
        llm_attn = self._resolve_llm_attn_implementation(llm_attn_implementation)
        vision_attn = self._resolve_vision_attn_implementation(
            vision_attn_implementation,
            bf16=bf16,
        )
        action_attn = self._resolve_action_attn_implementation(
            action_attn_implementation
        )
        self.model.vlm.set_attn_implementation(
            {
                "": llm_attn,
                "text_config": llm_attn,
                "vision_config": vision_attn,
            }
        )
        self.model.action_expert.set_action_attention_backend(action_attn)
        logger.info(
            "Configured runtime attention implementations: "
            "LLM=%s, Vision=%s, Action=%s",
            llm_attn,
            vision_attn,
            action_attn,
        )

    def _apply_liger_kernel(self):
        """Apply Liger Kernel fused optimizations to the VLM and Action Expert."""
        try:
            from liger_kernel.transformers import (
                apply_liger_kernel_to_gemma3,
                apply_liger_kernel_to_gemma3_text,
            )
        except ImportError:
            logger.warning(
                "liger-kernel is not installed; skipping Liger optimizations. "
                "Install it with: pip install liger-kernel"
            )
            return

        # Patch the VLM (Gemma3ForConditionalGeneration):
        # RMSNorm / GeGLU / RoPE / SigLIP LayerNorm.
        apply_liger_kernel_to_gemma3(
            model=self.model.vlm,
            rope=True,
            rms_norm=True,
            geglu=True,
            layer_norm=True,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        )

        # Patch the Action Expert (Gemma3TextModel):
        # RMSNorm / GeGLU. RoPE and loss are handled by suffix forward.
        apply_liger_kernel_to_gemma3_text(
            model=self.model.action_expert,
            rope=False,
            rms_norm=True,
            geglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        )

        logger.info(
            "Liger Kernel applied: VLM (RMSNorm+GeGLU+RoPE+LayerNorm), "
            "AE (RMSNorm+GeGLU)"
        )

    def enable_gradient_checkpointing(
        self,
        *,
        vlm_gradient_checkpointing: bool = True,
        ae_gradient_checkpointing: bool = True,
    ):
        """Configure runtime VLM prefix and AE suffix gradient checkpointing."""
        vlm_gradient_checkpointing = bool(vlm_gradient_checkpointing)
        ae_gradient_checkpointing = bool(ae_gradient_checkpointing)

        if vlm_gradient_checkpointing:
            self.model.vlm.model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            vision_tower = getattr(self.model.vlm.model, "vision_tower", None)
            if vision_tower is not None and hasattr(
                vision_tower, "gradient_checkpointing_enable"
            ):
                vision_tower.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            elif vision_tower is not None and hasattr(
                vision_tower, "gradient_checkpointing"
            ):
                vision_tower.gradient_checkpointing = True
        else:
            self.model.vlm.model.language_model.gradient_checkpointing_disable()
            vision_tower = getattr(self.model.vlm.model, "vision_tower", None)
            if vision_tower is not None and hasattr(
                vision_tower, "gradient_checkpointing_disable"
            ):
                vision_tower.gradient_checkpointing_disable()
            elif vision_tower is not None and hasattr(
                vision_tower, "gradient_checkpointing"
            ):
                vision_tower.gradient_checkpointing = False

        if ae_gradient_checkpointing:
            self.model.action_expert.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            self.model.action_expert.gradient_checkpointing_disable()

    # -----------------------------------------------------------------------
    # Training forward pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        pixel_values: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        action: torch.FloatTensor | None = None,
        action_mask: torch.BoolTensor | None = None,
        history_pixel_values: torch.Tensor | None = None,
        history_mask: torch.BoolTensor | None = None,
        tactile_pixel_values: torch.Tensor | None = None,
        **kwargs,
    ) -> Gemma3CausalLMOutputWithPast:
        """Run the training forward pass and compute flow-matching loss."""
        batch_size = input_ids.shape[0]
        if self.precision_policy == FP32_MIXED_PRECISION_POLICY:
            action = action.to(dtype=MODEL_DTYPE)
        action_mask = torch.broadcast_to(action_mask, action.shape)
        valid_actions = action_mask.sum(dim=(1, 2))
        if (valid_actions <= 0).any():
            raise ValueError("Each training sample must contain a valid action")

        # Step 1: prefix forward — history via unused0 scatter, current views via VLM.
        kv_cache, prefix_hidden_states = self._compute_prefix_cache(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            history_pixel_values=history_pixel_values,
            history_mask=history_mask,
            tactile_pixel_values=tactile_pixel_values,
        )
        prefix_len = prefix_hidden_states.shape[1]

        # Step 2: run suffix forward for flow matching.
        noise = torch.randn_like(action)
        time = (
            torch.distributions.Beta(1.5, 1.0)
            .sample((batch_size,))
            .to(action.device, dtype=action.dtype)
            * 0.999
            + 0.001
        )
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * action
        x_t = x_t * action_mask
        u_t = noise - action

        # Build suffix embeddings and per-layer time conditioning.
        suffix_embeds = self._action_input_proj(x_t)
        adarms_cond = self._build_adarms_cond(time)

        # Run suffix forward with fused attention.
        suffix_attn_mask, suffix_position_ids = self._build_suffix_metadata(
            input_ids=input_ids,
            prefix_len=prefix_len,
            suffix_len=suffix_embeds.shape[1],
            device=suffix_embeds.device,
            dtype=suffix_embeds.dtype,
        )

        suffix_out = self._suffix_forward(
            suffix_embeds=suffix_embeds,
            attention_mask=suffix_attn_mask,
            position_ids=suffix_position_ids,
            past_key_values=kv_cache,
            adarms_cond=adarms_cond,
        )

        # Compute the flow-matching loss.
        v_t = self._action_output_proj(suffix_out)
        u_t = u_t.to(dtype=v_t.dtype)

        elem_mse = F.mse_loss(v_t, u_t, reduction="none")  # [B, T, D]

        per_sample_fm = (elem_mse * action_mask).sum(dim=(1, 2)) / valid_actions
        fm_loss = per_sample_fm.mean()
        loss = fm_loss
        if self.precision_policy == FP32_MIXED_PRECISION_POLICY:
            # FSDP observes wrapped-module backward through returned tensors, while
            # the action expert also consumes KV tensors written into the cache.
            loss = loss + prefix_hidden_states.sum(dtype=torch.float32) * 0.0

        return TacMind0OutputWithPast(
            loss=loss,
            fm_loss=fm_loss,
            logits=None,
            past_key_values=None,
        )

    # -----------------------------------------------------------------------
    # Action inference
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def inference_action(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        states: torch.FloatTensor | None = None,
        image_masks: torch.BoolTensor | None = None,
        diffusion_steps: int = 10,
        past_key_values: Cache | None = None,
        action_mask: torch.BoolTensor | None = None,
        history_pixel_values: torch.Tensor | None = None,
        history_mask: torch.BoolTensor | None = None,
        tactile_pixel_values: torch.Tensor | None = None,
        reference_action: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        kv_cache, prefix_hidden_states = self._compute_prefix_cache(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            history_pixel_values=history_pixel_values,
            history_mask=history_mask,
            tactile_pixel_values=tactile_pixel_values,
        )
        prefix_len = prefix_hidden_states.shape[1]
        del prefix_hidden_states

        batch_size = input_ids.shape[0]
        device = input_ids.device
        dtype = self.model.action_in_proj.weight.dtype

        if reference_action is None:
            x_t = torch.randn(
                batch_size,
                self.model.config.chunk_size,
                self.model.config.action_dim,
                device=device,
                dtype=dtype,
            )
        else:
            x_t = prepare_action_chunk(
                reference_action,
                batch_size=batch_size,
                chunk_size=self.model.config.chunk_size,
                action_dim=self.model.config.action_dim,
                dtype=dtype,
                device=device,
                action_mask=action_mask,
            )
            x_t = self._sample_action_flow(
                x_t,
                input_ids=input_ids,
                kv_cache=kv_cache,
                prefix_len=prefix_len,
                diffusion_steps=diffusion_steps,
                action_mask=action_mask,
                reverse=True,
            )
            x_t = x_t.to(dtype=dtype)

        return self._sample_action_flow(
            x_t,
            input_ids=input_ids,
            kv_cache=kv_cache,
            prefix_len=prefix_len,
            diffusion_steps=diffusion_steps,
            action_mask=action_mask,
            reverse=False,
        )

    def _sample_action_flow(self, noise, **kwargs):
        graphs = getattr(self, "_action_flow_graphs", None)
        if graphs is not None:
            return graphs(noise, **kwargs)
        return self._integrate_action_flow(noise, **kwargs)

    def _integrate_action_flow(
        self,
        x_t: torch.Tensor,
        *,
        input_ids: torch.LongTensor,
        kv_cache: Cache,
        prefix_len: int,
        diffusion_steps: int,
        action_mask: torch.Tensor | None,
        reverse: bool = False,
        tactile_modulations: torch.Tensor | None = None,
        checkpoint_steps: bool = False,
        timesteps: torch.Tensor | None = None,
        suffix_runner=None,
    ) -> torch.Tensor:
        batch_size = int(input_ids.shape[0])
        device = x_t.device
        dtype = self.model.action_in_proj.weight.dtype
        _, dt = flow_euler_schedule(diffusion_steps, reverse=reverse)
        # Time conditioning does not depend on the evolving action. Batch its
        # MLP across steps while preserving the Euler schedule's scalar rounding.
        # A graph caller owns this fixed input buffer for the graph's lifetime.
        # Ordinary callers retain the same schedule construction and behavior.
        if timesteps is None:
            timesteps = flow_euler_timesteps(int(diffusion_steps), reverse, device, dtype)
        elif (
            timesteps.shape != (int(diffusion_steps),)
            or timesteps.device != device
            or timesteps.dtype != dtype
        ):
            raise ValueError("timesteps must match diffusion_steps, action device and dtype")
        time_tensor = timesteps
        time_tensor = time_tensor[:, None].expand(-1, batch_size).reshape(-1)
        conditions = self._build_adarms_cond(time_tensor).unflatten(
            0, (int(diffusion_steps), batch_size)
        )
        suffix_attn_mask = None
        suffix_position_ids = None
        rotary_cache = {}
        suffix_call = self._suffix_forward if suffix_runner is None else suffix_runner
        if checkpoint_steps and torch.is_grad_enabled():
            from functools import partial

            from torch.utils.checkpoint import checkpoint
            suffix_call = partial(checkpoint, self._suffix_forward, use_reentrant=False)
        for adarms_cond in conditions:
            if action_mask is not None:
                x_t = x_t * action_mask
            suffix_embeds = self._action_input_proj(x_t)
            if suffix_attn_mask is None:
                suffix_attn_mask, suffix_position_ids = self._build_suffix_metadata(
                    input_ids=input_ids,
                    prefix_len=prefix_len,
                    suffix_len=int(suffix_embeds.shape[1]),
                    device=suffix_embeds.device,
                    dtype=suffix_embeds.dtype,
                )

            suffix_out = suffix_call(
                suffix_embeds=suffix_embeds,
                attention_mask=suffix_attn_mask,
                position_ids=suffix_position_ids,
                past_key_values=kv_cache,
                adarms_cond=adarms_cond,
                tactile_modulations=tactile_modulations,
                rotary_cache=rotary_cache,
            )

            v_t = self._action_output_proj(suffix_out)
            x_t = x_t + v_t * dt
        return x_t

    def _build_suffix_metadata(
        self,
        *,
        input_ids: torch.LongTensor,
        prefix_len: int,
        suffix_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        invisible_prefix_token_ids = (HISTORY_PAD_TOKEN_ID,)
        attention_mask = make_suffix_attn_mask(
            input_ids=input_ids,
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            batch_size=int(input_ids.shape[0]),
            device=device,
            dtype=dtype,
            pad_token_id=self.model.vlm.model.language_model.padding_idx,
            invisible_prefix_token_ids=invisible_prefix_token_ids,
        )
        position_ids = self._build_suffix_position_ids(
            prefix_len,
            suffix_len,
            device,
            input_ids=input_ids,
            pad_token_id=self.model.vlm.model.language_model.padding_idx,
            invisible_prefix_token_ids=invisible_prefix_token_ids,
        )
        return attention_mask, position_ids

    def _compute_prefix_cache(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        token_type_ids: torch.LongTensor | None,
        history_pixel_values: torch.Tensor | None = None,
        history_mask: torch.BoolTensor | None = None,
        tactile_pixel_values: torch.Tensor | None = None,
        cache_cls: type[Cache] = VLADynamicCache,
    ) -> tuple[Cache, torch.Tensor]:
        """Fill VLM KV cache; history images are injected separately from current views.

        History uses ``<unused0>`` placeholders replaced via vision features +
        ``masked_scatter``. Unavailable slots use ``<unused1>`` pads; those
        positions are zeroed in embeds and masked in attention/position ids.
        Current camera images go through the
        standard Gemma3 multimodal path (``pixel_values`` + ``token_type_ids``).
        """
        # Action suffixes attend to the full prefix in every layer. Keep the
        # same KV tensors in training and inference, including sliding layers.
        kv_cache = cache_cls(config=self.model.language_model.config)
        prefix_inputs_embeds = None
        vlm_model = self.model.vlm.model
        embed_tokens = vlm_model.get_input_embeddings()
        has_history_pixels = (
            history_pixel_values is not None
            and history_mask is not None
            and bool(history_mask.any().item())
        )
        # Tactile prefixes always need embeddings. Avoid synchronizing just to
        # decide whether zeroing history-pad positions is necessary.
        needs_embeddings = (
            self.config.tactile_config is not None
            or has_history_pixels
            or bool((input_ids == HISTORY_PAD_TOKEN_ID).any().item())
        )
        if needs_embeddings:
            inputs_embeds = embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.masked_fill(
                (input_ids == HISTORY_PAD_TOKEN_ID).unsqueeze(-1), 0
            )

            if has_history_pixels:
                pixels = history_pixel_values.to(
                    device=inputs_embeds.device,
                    dtype=next(vlm_model.vision_tower.parameters()).dtype,
                )
                image_features = vlm_model.get_image_features(
                    pixels, return_dict=True
                ).pooler_output

                spatial = int(image_features.shape[1] ** 0.5)
                hidden = image_features.shape[-1]
                grid = image_features.view(-1, spatial, spatial, hidden).permute(
                    0, 3, 1, 2
                )
                grid = F.adaptive_avg_pool2d(
                    grid, output_size=(HISTORY_POOL_SIZE, HISTORY_POOL_SIZE)
                )
                image_features = grid.permute(0, 2, 3, 1).reshape(
                    -1, HISTORY_POOL_SIZE * HISTORY_POOL_SIZE, hidden
                )

                history_mask_expanded = history_mask.unsqueeze(-1).expand_as(
                    inputs_embeds
                )
                prefix_inputs_embeds = inputs_embeds.masked_scatter(
                    history_mask_expanded, image_features
                )
            else:
                prefix_inputs_embeds = inputs_embeds

        if self.config.tactile_config is not None:
            if tactile_pixel_values is None:
                raise ValueError("TacMind-0 requires tactile_pixel_values")
            if pixel_values is None:
                raise ValueError("TacMind-0 requires current vision pixels")
            if input_ids.shape[1] == 0:
                raise ValueError("TacMind-0 requires exactly one tactile CLS token")
            tactile_mask = input_ids == TACTILE_TOKEN_ID
            image_mask = input_ids == vlm_model.config.image_token_id
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            tactile_position = tactile_mask.long().argmax(dim=1, keepdim=True)
            # Check the whole batch in one device-to-host transfer. Boolean
            # indexing here also synchronizes CUDA to size its dynamic output.
            checks = [
                (tactile_mask.sum(dim=1) == 1).all(),
                (image_mask.sum(dim=1) == 2 * vlm_model.config.mm_tokens_per_image).all()
                & (pixel_values.shape[0] == 2 * input_ids.shape[0]),
                ~(image_mask & (positions >= tactile_position)).any(),
            ]
            messages = [
                "TacMind-0 requires exactly one tactile CLS token",
                "TacMind-0 requires two current views per sample",
                "Tactile CLS must follow all image tokens",
            ]
            if attention_mask is not None:
                checks.append((~tactile_mask | attention_mask.bool()).all())
                messages.append("Tactile CLS must be visible in the attention mask")
            for valid, message in zip(
                torch.stack(checks).cpu().tolist(), messages, strict=True
            ):
                if not valid:
                    raise ValueError(message)
            z = self.tactile_encoder(tactile_pixel_values)
            if z.shape[0] != input_ids.shape[0]:
                raise ValueError("Tactile batch does not match language batch")
            tactile_token = self.tactile_projector(
                z.to(dtype=self.tactile_projector.weight.dtype)
            )
            if prefix_inputs_embeds is None:
                prefix_inputs_embeds = embed_tokens(input_ids)
            prefix_inputs_embeds = prefix_inputs_embeds.masked_scatter(
                tactile_mask.unsqueeze(-1).expand_as(prefix_inputs_embeds),
                tactile_token.to(prefix_inputs_embeds),
            )
            # Vision features follow Gemma3's normal path; only the CLS slot is replaced.
            if token_type_ids is not None:
                token_type_ids = token_type_ids.masked_fill(tactile_mask, 0)
        elif tactile_pixel_values is not None:
            raise ValueError("Tactile input supplied to a vision-only checkpoint")

        prefix_attention_mask = mask_history_pad_tokens_in_attention(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        prefix_position_ids = (prefix_attention_mask.cumsum(dim=-1) - 1).clamp_min_(0)
        model_kwargs = {
            "input_ids": None if prefix_inputs_embeds is not None else input_ids,
            "attention_mask": prefix_attention_mask,
            "position_ids": prefix_position_ids,
            "past_key_values": kv_cache,
            "inputs_embeds": prefix_inputs_embeds,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids,
            "use_cache": True,
        }

        prefix_outputs = self.model.vlm.model(**model_kwargs)
        return kv_cache, prefix_outputs.last_hidden_state

    def _extract_prefix_cache_tensors(
        self,
        kv_cache: Cache,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        n_layers = len(self.model.action_expert.layers)
        keys = []
        values = []
        for layer_idx in range(n_layers):
            layer = kv_cache.layers[layer_idx]
            keys.append(layer.keys)
            values.append(layer.values)
        return tuple(keys), tuple(values)

    def _build_adarms_cond(
        self,
        time: torch.Tensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        ae_hidden = self.model.action_in_proj.out_features
        if self.precision_policy == FP32_MIXED_PRECISION_POLICY:
            time_emb = posemb_sincos(time, ae_hidden, max_period=4.0).to(MODEL_DTYPE)
            cond = linear_fp32(time_emb, self.model.time_mlp_in)
            cond = F.silu(cond)
            cond = linear_fp32(cond, self.model.time_mlp_out)
            return F.silu(cond)

        dtype = self.model.time_mlp_in.weight.dtype if dtype is None else dtype
        time_emb = posemb_sincos(time, ae_hidden, max_period=4.0).to(dtype)
        cond = self.model.time_mlp_in(time_emb)
        cond = F.silu(cond)
        cond = self.model.time_mlp_out(cond)
        return F.silu(cond)

    def _action_input_proj(self, x: torch.Tensor) -> torch.Tensor:
        return linear_fp32(x, self.model.action_in_proj).to(self._suffix_hidden_dtype())

    def _action_output_proj(self, x: torch.Tensor) -> torch.Tensor:
        return linear_fp32(x, self.model.action_out_proj)

    def _suffix_hidden_dtype(self) -> torch.dtype:
        return next(self.model.action_expert.parameters()).dtype

    def _build_suffix_position_ids(
        self,
        prefix_len,
        suffix_len,
        device,
        input_ids: torch.Tensor,
        pad_token_id=0,
        invisible_prefix_token_ids: tuple[int, ...] = (),
    ):
        valid_prefix = input_ids[:, :prefix_len] != pad_token_id  # [B, prefix_len]
        for token_id in invisible_prefix_token_ids:
            valid_prefix = valid_prefix & (input_ids[:, :prefix_len] != token_id)
        effective_prefix_len = valid_prefix.sum(dim=1)  # [B]

        # suffix position ids = [effective_prefix_len, effective_prefix_len+1, ...]
        suffix_offsets = torch.arange(suffix_len, device=device).unsqueeze(
            0
        )  # [1, suffix_len]
        pos = effective_prefix_len.unsqueeze(1) + suffix_offsets  #  [B, suffix_len]
        return pos

    def _suffix_forward(
        self,
        suffix_embeds: torch.Tensor,
        attention_mask: Any,
        position_ids: torch.LongTensor,
        past_key_values: Cache,
        adarms_cond: torch.Tensor,
        tactile_modulations: torch.Tensor | None = None,
        rotary_cache: dict | None = None,
    ) -> torch.Tensor:
        """Extract cache tensors and run the action expert suffix forward."""
        prefix_cache_keys, prefix_cache_values = self._extract_prefix_cache_tensors(
            past_key_values,
        )

        return self.model.action_expert(
            suffix_embeds=suffix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prefix_cache_keys=prefix_cache_keys,
            prefix_cache_values=prefix_cache_values,
            adarms_cond=adarms_cond,
            tactile_modulations=tactile_modulations,
            rotary_cache=rotary_cache,
        )


AutoConfig.register("tacmind0", TacMind0Config)
AutoModelForCausalLM.register(TacMind0Config, TacMind0ForConditionalGeneration)
