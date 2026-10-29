"""Bounded TensorRT engines and CUDA graphs for serialized action inference."""

import logging
from collections import OrderedDict
from types import SimpleNamespace

import torch

from .operations import flow_euler_timesteps

logger = logging.getLogger(__name__)


class ActionFlowGraphs:
    """Capture the ordinary integrator; callers must serialize GPU requests.

    Four exact-shape profiles cover ordinary sampling and the three Tac-FRS
    flows without retaining an unbounded number of prompt lengths. A first
    occurrence runs eagerly; only a repeated shape pays the capture cost.
    """

    def __init__(self, policy, *, backend, cuda_graphs=True):
        self.policy = policy
        if backend not in ("torch", "tensorrt"):
            raise ValueError(f"Unsupported action backend: {backend}")
        self.backend = backend
        self.cuda_graphs = cuda_graphs
        self.profiles = OrderedDict()
        self.engines = {}
        self.validations = OrderedDict()
        self.weight_signature = None

    def _signature(self):
        model = self.policy.model
        modules = (
            model.action_expert,
            model.action_in_proj,
            model.action_out_proj,
            model.time_mlp_in,
            model.time_mlp_out,
        )
        tensors = [
            tensor
            for module in modules
            for tensor in (*module.parameters(), *module.buffers())
        ]
        return (
            self.policy.precision_policy,
            model.action_expert._suffix_attn_backend,
            model.vlm.model.language_model.padding_idx,
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
            torch.is_inference_mode_enabled(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            tuple((id(t), t.data_ptr(), t._version) for t in tensors),
        )

    def clear(self):
        for profile in self.profiles.values():
            if isinstance(profile, tuple):
                profile[0].reset()
        self.profiles.clear()
        self.engines.clear()
        self.validations.clear()

    def __call__(
        self,
        noise,
        *,
        input_ids,
        kv_cache,
        prefix_len,
        diffusion_steps,
        action_mask,
        reverse=False,
        tactile_modulations=None,
        checkpoint_steps=False,
    ):
        policy = self.policy
        flow = dict(
            input_ids=input_ids,
            kv_cache=kv_cache,
            prefix_len=prefix_len,
            diffusion_steps=diffusion_steps,
            action_mask=action_mask,
            reverse=reverse,
            tactile_modulations=tactile_modulations,
            checkpoint_steps=checkpoint_steps,
        )
        if (
            torch.is_grad_enabled()
            or policy.training
            or not noise.is_cuda
            or policy.model.action_expert.training
            or policy.model.action_expert._suffix_attn_backend != "sdpa"
        ):
            self.clear()
            return policy._integrate_action_flow(noise, **flow)
        signature = self._signature()
        if signature != self.weight_signature:
            self.clear()
            self.weight_signature = signature
        keys, values = policy._extract_prefix_cache_tensors(kv_cache)
        inputs = (noise, input_ids, action_mask, tactile_modulations, *keys, *values)
        key = (
            prefix_len,
            int(diffusion_steps),
            reverse,
            tuple(None if x is None else (x.shape, x.dtype, x.device) for x in inputs),
        )
        if self.backend == "tensorrt":
            from .tensorrt_suffix import TensorRTSuffix, check_actions

            # Forward/reverse flows and zero/nonzero modulations share one
            # expert engine. Graphs are cleared before retiring their engines.
            if tactile_modulations is not None:
                modulation_dtype = tactile_modulations.dtype
            elif torch.is_autocast_enabled("cuda"):
                modulation_dtype = torch.get_autocast_dtype("cuda")
            else:
                modulation_dtype = torch.float32
            engine_key = (
                noise.shape,
                input_ids.shape,
                prefix_len,
                tuple((x.shape, x.dtype) for x in (*keys, *values)),
                modulation_dtype,
            )
            if engine_key not in self.engines:
                if len(self.engines) == 2:
                    self.clear()
                with (
                    torch.cuda.device(noise.device),
                    torch.random.fork_rng(devices=[noise.device.index]),
                ):
                    self.engines[engine_key] = TensorRTSuffix(
                        policy, noise, flow, modulation_dtype=modulation_dtype
                    )
            runner = self.engines[engine_key]
            if key not in self.validations:
                expected = policy._integrate_action_flow(noise, **flow)
                actual = policy._integrate_action_flow(
                    noise, **flow, suffix_runner=runner
                )
                self.validations[key] = check_actions(
                    actual,
                    expected,
                    bf16=(
                        policy._suffix_hidden_dtype() == torch.bfloat16
                        or torch.is_autocast_enabled("cuda")
                    ),
                )
                logger.info("TensorRT action validation: %s", self.validations[key])
                if len(self.validations) > 8:
                    self.validations.popitem(last=False)
            flow["suffix_runner"] = runner
        if not self.cuda_graphs:
            return policy._integrate_action_flow(noise, **flow)
        if key not in self.profiles:
            if len(self.profiles) == 4:
                _, retired = self.profiles.popitem(last=False)
                if isinstance(retired, tuple):
                    retired[0].reset()
            self.profiles[key] = None
            return policy._integrate_action_flow(noise, **flow)
        self.profiles.move_to_end(key)
        profile = self.profiles[key]
        if profile is False:
            return policy._integrate_action_flow(noise, **flow)
        if profile is None:
            try:
                profile = self._capture(inputs, len(keys), flow)
            except (RuntimeError, AssertionError):
                self.profiles[key] = False
                logger.warning(
                    "CUDA Graph capture failed; using ordinary action integration",
                    exc_info=True,
                )
                return policy._integrate_action_flow(noise, **flow)
            self.profiles[key] = profile
        graph, buffers, output, _timesteps = profile
        for target, source in zip(buffers, inputs, strict=True):
            if target is not None:
                target.copy_(source)
        graph.replay()
        # Each request owns its result, including intermediate Tac-FRS actions.
        return output.clone()

    def _capture(self, inputs, depth, flow):
        buffers = tuple(None if x is None else x.detach().clone() for x in inputs)
        noise, ids, mask, mods, *kv = buffers
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(keys=kv[i], values=kv[depth + i]) for i in range(depth)
            ]
        )
        timesteps = flow_euler_timesteps(
            flow["diffusion_steps"],
            flow["reverse"],
            noise.device,
            self.policy.model.action_in_proj.weight.dtype,
        )
        kwargs = dict(
            flow,
            input_ids=ids,
            kv_cache=cache,
            action_mask=mask,
            tactile_modulations=mods,
            timesteps=timesteps,
        )
        stream = torch.cuda.Stream(device=noise.device)
        current = torch.cuda.current_stream(noise.device)
        stream.wait_stream(current)
        with (
            torch.cuda.stream(stream),
            torch.autocast(
                "cuda",
                enabled=torch.is_autocast_enabled("cuda"),
                dtype=torch.get_autocast_dtype("cuda"),
                cache_enabled=False,
            ),
        ):
            for _ in range(3):
                expected = self.policy._integrate_action_flow(noise, **kwargs)
        current.wait_stream(stream)
        current.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with (
                torch.cuda.graph(graph, stream=stream),
                torch.autocast(
                    "cuda",
                    enabled=torch.is_autocast_enabled("cuda"),
                    dtype=torch.get_autocast_dtype("cuda"),
                    cache_enabled=False,
                ),
            ):
                output = self.policy._integrate_action_flow(noise, **kwargs)
            current.wait_stream(stream)
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
        except (RuntimeError, AssertionError):
            graph.reset()
            raise
        # Keep all static inputs alive, including the device timestep schedule.
        return graph, buffers, output, timesteps
