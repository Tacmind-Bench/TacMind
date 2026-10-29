# Optional FP8 prefix MLP inference

[中文](../zh/dm05_prefix_mlp_fp8.md)

The fast backend can replace the Gemma3 prefix MLPs with checkpoint-specific
TensorRT FP8 engines. It quantizes the gate, up and down matrix multiplications
using per-tensor E4M3 weight and activation scales. Engine input/output tensors
are FP16; the surrounding prefix computation remains BF16. Vision, attention,
the action expert, action masks and the number of denoising steps retain their
existing configuration. Fast automatically uses a valid prepared generation
under its checkpoint. Without one, it keeps the original BF16 MLPs. Discovery
does not calibrate or build during service startup.

FP8 changes numerical precision; evaluate task success with the deployment configuration.

## One-command preparation

`prepare_fast` prepares a complete offline bundle for fast inference. Currently
it requires the explicit choice `--prefix-mlp-precision fp8`; this selects the
precision of the prefix MLP only, not the whole model:

| Component | Prepared artifact | Calibration required? |
| --- | --- | --- |
| Vision encoder and projector | TensorRT FP16 engine | No FP8 calibration |
| Prefix MLP gate/up/down | TensorRT FP8 engines, with FP16 inputs/outputs | Yes, using representative real inputs |

The surrounding prefix computation remains BF16; attention and the action expert
retain their existing fast implementation. The command runs preprocessing,
builds the vision FP16 engine, uses that same engine to collect MLP calibration
data, then builds the MLP FP8 engines. You do not need to build vision separately.

Install the fast dependencies described below, then use representative raw
deployment requests with the exact selected playground preprocessing:

```bash
python -m opendm.infer.prepare_fast \
  --checkpoint ./checkpoints/DM05-libero \
  --playground libero \
  --prefix-mlp-precision fp8 \
  --requests-jsonl ./calibration_requests.jsonl \
  --samples 128
```

This prepares artifacts; it does not start an inference service. The CLI currently
supports only `fp8` for `--prefix-mlp-precision`, with no BF16-only preparation
mode. Ordinary BF16 fast inference can be started without this calibration step.
Selecting `--inference-config.backend fast` alone never calibrates the MLP or
builds MLP FP8 engines during service startup.

The old `python -m opendm.infer.prepare_fast_fp8 ...` command remains a compatible
alias: it selects FP8 when the precision option is omitted and prepares the same
vision FP16 + prefix MLP FP8 bundle. Existing scripts and compatible prepared
generations keep working. New scripts should use `prepare_fast` with the explicit
precision option. The output directory and `prepared.json` format are unchanged.

A request row contains `text`, `states`, and an ordered `images` list of paths
or `{"type":"video","url":"...","frame_idx":N}` descriptors. Actual dataset
rows with `prompt`, `state`, and registered `images_1`, `images_2`, ... fields
are accepted with `--dataset-root` or `--jsonl`. Add `--media-root` for media
stored outside the JSONL tree. Existing transformed tensor files are accepted
with `--inputs file.pt`.

Sampling is deterministic and rotates across tasks and episodes. Increase the
default 128 samples when the deployment has more tasks or needs broader state
coverage. RoboTwin2 uses `--playground robotwin2`. Custom deployments can use
`--playground-module package.module` (or a `.py` path), `--dataset-name`, and
repeatable `--config-override dotted.path=JSON`; custom modules must provide
`--min-seq-len` and `--opt-seq-len`.

One-command FP8 preparation currently supports `--prefix-capacity` up to
`1024`, and `--bucket` must not exceed that capacity. A value such as `2048`
is rejected before an output directory or any preprocessing/build subprocess is
created. Use the BF16 fast path for larger capacities; this tool does not build
a partial 1024-token FP8 profile for them.

The preparation stages run in separate sequential processes. The command
publishes `prepared.json` only after every stage succeeds. The default output is
`<checkpoint>/.opendm/fast_fp8`; a failed attempt preserves the previous ready
generation. With this default output, start fast normally to reuse compatible
vision and MLP engines automatically:

```bash
python playground/dm05_libero.py --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-libero \
  --inference-config.backend fast
```

If preparation uses `--output-dir ./artifacts/libero-fast-fp8`, pass that same
directory when starting the service:

```bash
python playground/dm05_libero.py --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-libero \
  --inference-config.backend fast \
  --inference-config.prepared-fast-artifacts-dir ./artifacts/libero-fast-fp8
```

Set `--inference-config.no-auto-prepared-fast-artifacts` to disable automatic
selection. Explicit legacy vision and FP8 engine paths retain their fail-fast
behavior and take precedence. LoRA adapters must first be merged and exported
as a standalone checkpoint.

Preparation checks artifact and deployment compatibility. It is not task
success-rate certification; run the task evaluator before promotion.

## Advanced manual workflow

### 1. Prepare the environment and vision engine

Install OpenDM and `pip install -e ".[fast-infer]"` as described in the
[inference guide](dm05_inference.md). Build on a GPU supported by TensorRT FP8;
the installed TensorRT must provide its strongly typed network and FP8 APIs.
The generated bundle records GPU model/capability, CUDA and TensorRT versions.
The runtime checks those fields and requires a rebuild when they differ.

Example for the public LIBERO checkpoint:

```bash
python -m opendm.infer.build_vision_trt \
  --checkpoint ./checkpoints/DM05-libero \
  --num-images 2 \
  --onnx-path ./artifacts/libero/vision.onnx \
  --engine-path ./artifacts/libero/vision.engine \
  --force-rebuild
```

Use a separate artifact directory for each checkpoint. Calibration and FP8
serving must load the same vision engine bytes; rebuilding that engine requires
new calibration and a new FP8 bundle.

### 2. Save representative transformed observations

Create a `torch.save` file containing a list of tensor dictionaries from the
same configured OpenDM inference transform used by your deployment:

```python
import torch

keys = ("pixel_values", "input_ids", "attention_mask", "token_type_ids")
samples = []
for observation in representative_observations:
    # inference_config is initialized by the matching playground experiment.
    observation = dict(observation)
    observation.setdefault("robot_type", None)  # Use the playground robot default.
    raw = inference_config._prepare_model_input(**observation)
    transformed = inference_config.input_transform(raw)
    samples.append({key: transformed[key].detach().cpu() for key in keys})
torch.save(samples, "calibration_inputs.pt")
```

Each observation supplies `text`, decoded RGB PIL `images`, and `states`, plus
the same optional robot/control metadata used for serving. Use real inputs
covering the intended tasks, viewpoints and states; the collector requires at
least two distinct samples, which is a format minimum rather than a sufficient
quality-validation set. Use separate episodes for subsequent validation.

Each dictionary contains only the four required tensors and optionally
`diffusion_input_noise` and `action_mask`. Token tensors have shape `[1, S]`;
pixels have shape `[num_images, 3, H, W]`. Optional noise is `[1, chunk_size, 32]`
and the optional mask is `[1, 1, 32]`. Files are loaded with
`torch.load(..., weights_only=True)`. Oversized samples are rejected, never
truncated by the collector. Ensure the preceding transform also preserved the
complete prompt and every camera's tokens.

### 3. Calibrate and build

```bash
python -m opendm.infer.collect_prefix_mlp_calibration \
  --checkpoint ./checkpoints/DM05-libero \
  --vision-trt-engine-path ./artifacts/libero/vision.engine \
  --inputs ./calibration_inputs.pt \
  --num-images 2 --chunk-size 10 --bucket 1024 \
  --output ./artifacts/libero/calibration.json

python -m opendm.infer.build_prefix_mlp_fp8_trt \
  --checkpoint ./checkpoints/DM05-libero \
  --calibration-json ./artifacts/libero/calibration.json \
  --output-dir ./artifacts/libero/prefix-fp8 \
  --min-seq-len 576 --opt-seq-len 704 --max-seq-len 1024
```

Keep the bundle directory and its adjacent `.<bundle-name>.lock` file together
when copying or mounting artifacts. Runtime readers support read-only mounts
when this lock already exists; creating a missing lock requires a writable
parent directory. Do not delete or replace the lock while readers or builders
are running.

For RoboTwin2, use its checkpoint and **three images, chunk size 50**, with
`--min-seq-len 896 --opt-seq-len 896 --max-seq-len 1024`. Calibrate it separately.

## 4. Serve

```bash
python playground/dm05_libero.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-libero \
  --model-config.chunk-size 10 \
  --inference-config.backend fast \
  --inference-config.diffusion-steps 10 \
  --inference-config.vision-trt-engine-path ./artifacts/libero/vision.engine \
  --inference-config.prefix-mlp-fp8-engine-dir ./artifacts/libero/prefix-fp8
```

All configured prefix buckets must fit the engine's optimization profile. An explicit FP8 request with
missing, incompatible or corrupt artifacts fails at startup. History inputs and
requests outside the configured buckets are rejected instead of silently using
BF16. Without selected FP8 artifacts, the BF16 fast path can execute a prefix
above its largest bucket eagerly when it remains within `fast_prefix_len`. This
implementation supports the BF16 static fast backend and standard
Gemma3 MLPs; merge and export adapters before building their engines.
