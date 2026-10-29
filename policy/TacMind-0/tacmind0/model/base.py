from transformers import PretrainedConfig, PreTrainedModel

class BasePolicyConfig(PretrainedConfig):
    model_type = "tacmind0"

class BasePolicyModel(PreTrainedModel):
    config: BasePolicyConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
