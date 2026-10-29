# DM05 Inference Guide

This is the canonical guide for starting the DM05 inference service and calling
its HTTP API. Benchmark and training guides link here instead of duplicating
inference setup.

[中文](../zh/dm05_inference.md)

For the opt-in TensorRT FP8 prefix MLP backend, see
[calibration, engine building and deployment](dm05_prefix_mlp_fp8.md).

## 1. Before You Start

Run all commands from the OpenDM repository root. Prepare:

- An OpenDM environment installed according to the main README.
- If you will use `--inference-config.backend fast`, install the additional
  dependency layer in that same environment with `pip install -e ".[fast-infer]"`.
- A DM05 checkpoint compatible with the selected playground entry point.
- The matching `norm_stats.json` in the checkpoint directory. If it is absent,
  OpenDM falls back to the matching file under `./norm_stats/`.
- One NVIDIA GPU.

One `norm_stats.json` may contain multiple robot profiles under
`norm_stats_by_robot`. OpenDM selects the exact profile using
`observation.robot_type`; if it is omitted, the configured dataset default or
the file's `default_robot_type` is used. Historical single-profile files remain
supported. An unknown robot never falls back silently to the default profile.

The checkpoint, playground entry point, `chunk_size`, image keys, state/action
dimensions, and normalization statistics must come from the same training
configuration.

For the fast backend, also keep these runtime prerequisites aligned with the
code path:

- TensorRT is required, not optional. Fast startup builds or loads the vision
  TensorRT engine before serving requests.
- Triton is required, not optional. The fast suffix big-kernel path and the
  fast prefix decoder path both import and dispatch Triton kernels.
- PyTorch FlexAttention support is required. Fast inference forces the LLM
  attention backend to `flex_attention`, and the static prefix fastpath checks
  every decoder layer for that backend.
- Use a PyTorch build that provides
  `torch.nn.attention.flex_attention` (for example `torch>=2.5`).

### Fast Backend Preflight Checklist

Before the first fast launch, confirm:

- CUDA GPU is visible in the target environment.
- `pip install -e ".[fast-infer]"` has completed in that environment.
- `python -c "import tensorrt"` succeeds.
- `python -c "import triton"` succeeds.
- `python -c "import torch.nn.attention.flex_attention"` succeeds.
- The checkpoint, playground entry point, `chunk_size`, action dimension, and
  `image_prompts` all come from the same training run.
- The uploaded image count and order match `--inference-config.image-prompts`.
- Extra time is reserved for the first startup to export ONNX and build the
  TensorRT engine before the HTTP service becomes ready.

## 2. Choose an Entry Point

| Use case | Entry point | Typical checkpoint | Chunk size | Images | Action dimension |
| --- | --- | --- | ---: | ---: | ---: |
| Base pretrained model | `opendm/exp/dm05_exp.py` | `Dexmal/DM05` | 50 | 3 | 14 |
| LIBERO | `playground/dm05_libero.py` | `Dexmal/DM05-libero` | 10 | 2 | 7 |
| RoboTwin 2.0 | `playground/dm05_robotwin2.py` | `Dexmal/DM05-robotwin2` | 50 | 3 | 14 |
| RoboColiseum G2 | `playground/dm05_robocoliseum.py` | Matching RoboColiseum checkpoint for 'Instruction Following', 'spatial reasoning' and 'Robustbess' suite | 50 | 3 | 16 / 16 (22-D internal) |
| RoboColiseum manipulation | `playground/dm05_robocoliseum_manipulation.py` | Matching RoboColiseum checkpoint for 'General Manipulation' suite'  | 30 | 3 | 21 / 21 (22-D internal) |
| DM05-MEM | `playground/dm05_mem_sft_demo.py` | `Dexmal/DM05-MEM` | 50 | 3 | 14 |
| Demo or custom SFT | `playground/dm05_sft_demo.py` or your own entry | Your SFT checkpoint | Training value | Training value | Training value |
| LIBERO LoRA | `playground/dm05_libero_lora.py` | A LIBERO LoRA step checkpoint | 10 | 2 | 7 |

Do not mix a benchmark checkpoint with another benchmark's entry point or
inference dimensions.

## 3. Start the Default Backend

The default backend uses the standard PyTorch path for the VLM prefix. On CUDA
in evaluation mode, the default `sdpa` action-attention backend automatically
uses CUDA Graph replay for the action suffix; unsupported configurations safely
remain on eager execution. No additional `fast-infer` dependencies are needed.

Suffix Graph profiles are created lazily per execution shape. The first request
in a new profile runs eagerly, the second captures the Graph, and later requests
reuse it. Prefix lengths are rounded up to 16-token buckets, and up to eight
profiles are retained. This supports varying prompts without a fixed Graph
prefix limit; a capture or replay failure falls back to eager execution.

The inference launcher starts one Python process directly, so
`--nproc_per_node` is not needed.

### DM05 Base Pretrained Model

Download the base pretrained checkpoint:

```bash
hf download Dexmal/DM05 \
  --local-dir ./checkpoints/DM05
```

Start the service:

```bash
script/dm05_launcher.sh \
  --exp opendm/exp/dm05_exp.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
  --inference-config.port 7891
```

The base pretrained checkpoint uses three images and a 14-dimensional
state/action. Set `observation.robot_type` to `DOS W1` or `Aloha` to select the
matching normalization profile. If the field is omitted, the checkpoint's
default profile, `DOS W1`, is used.

Requests to the base pretrained model should explicitly provide
`observation.control_mode` and `observation.speed` so that the input matches
the pretraining conditions. If `observation.speed` is omitted, the service
defaults it to `"0.5"`.

### LIBERO

Download the released checkpoint:

```bash
hf download Dexmal/DM05-libero \
  --local-dir ./checkpoints/DM05-libero
```

Start the service:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_libero.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-libero \
  --model-config.chunk-size 10 \
  --inference-config.output-action-dim 7 \
  --inference-config.image-prompts "Head" "Left wrist" \
  --inference-config.port 7891
```

The LIBERO entry point uses two images, an 8-dimensional state, a 7-dimensional
action, and the `Franka` robot type.

### RoboTwin 2.0

Download the released checkpoint:

```bash
hf download Dexmal/DM05-robotwin2 \
  --local-dir ./checkpoints/DM05-robotwin2-bf16
```

Start the service:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_robotwin2.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-robotwin2-bf16 \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
  --inference-config.port 7891
```

The RoboTwin 2.0 entry point uses three images, a 14-dimensional state/action,
and the `Aloha RoboTwin2` robot type.

### RoboColiseum G2

Use a checkpoint trained for the selected RoboColiseum layout. The base G2
service is started with:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_robocoliseum.py \
  --task inference \
  --model-config.model-name-or-path /path/to/robocoliseum-g2-checkpoint \
  --inference-config.port 7891
```

The API accepts a 16-dimensional state in native RoboColiseum order:

```text
[left joints (7), left gripper, right joints (7), right gripper]
```

It returns 50 actions in the same 16-dimensional order. The entry point maps
the request and response to the checkpoint's shared 22-dimensional layout
internally, so keep `--inference-config.output-action-dim` at its entry-point
default of `22`. It requires three current images in `Head`, `Left wrist`,
`Right wrist` order and does not accept history images.

For the 30-step whole-body manipulation checkpoint, use:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_robocoliseum_manipulation.py \
  --task inference \
  --model-config.model-name-or-path /path/to/robocoliseum-manipulation-checkpoint \
  --inference-config.port 7891
```

This variant accepts and returns the native 21-dimensional layout:

```text
[left joints (7), left gripper, right joints (7), right gripper, torso (5)]
```

The response contains 30 actions. Only the final torso dimension is predicted;
the first four torso values are copied from the request state into every action.
The entry point performs the shared-layout and gripper-sign conversions
internally. It also uses three current images, no history images, and an
internal `output_action_dim` of `22`.

Both entries read normalization statistics from `norm_stats.json` in the
checkpoint directory by default. If the file is stored elsewhere, pass
`--norm-stats /path/to/norm_stats.json`.

### DM05-MEM

`playground/dm05_mem_sft_demo.py` is the DM05-MEM training and inference entry.
See the [DM05-MEM SFT and Validation Guide](dm05_mem_sft.md) for the full
training flow.

Download the released checkpoint:

```bash
hf download Dexmal/DM05-MEM \
  --local-dir ./checkpoints/DM05-MEM
```

Start the service:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_mem_sft_demo.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-MEM \
  --inference-config.port 7891
```

The DM05-MEM entry point uses three images, a 14-dimensional
state/action, and the `DOS W1` robot type. `observation.control_mode`
defaults to `joint`; set it to `eef` for end-effector control. `speed`
defaults to `"0.1"`. History defaults to 32 frames on both backends;
FastInfer needs a matching 35-image TensorRT engine. To validate a
checkpoint this entry just trained, point `model-name-or-path` at that
step directory and send a request with `history_images`; see section 6 of
the [DM05-MEM SFT and Validation Guide](dm05_mem_sft.md).

### Demo or Custom SFT

Use the checkpoint produced by training and keep all data-dependent settings
consistent with that run. For the built-in demo configuration:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_sft_demo.py \
  --task inference \
  --data-config.dataset-name demo \
  --model-config.model-name-or-path ./user_checkpoints/dm05_sft_demo/checkpoint-50000 \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
  --inference-config.port 7891
```

For custom data, replace the entry point, dataset name, checkpoint, chunk size,
image keys, and action dimension with the values used during training.

### LIBERO LoRA

Pass a LoRA step checkpoint as `model-name-or-path`. The loader reads its
`adapter_config.json`, loads the recorded base model, and merges the adapter for
inference.

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_libero_lora.py \
  --task inference \
  --model-config.model-name-or-path ${TRAINING_OUTPUT_DIR}/checkpoint-50000 \
  --model-config.chunk-size 10 \
  --inference-config.output-action-dim 7 \
  --inference-config.image-prompts "Head" "Left wrist" \
  --inference-config.port 7891
```

## 4. Start the Fast Backend

The fast backend uses a TensorRT vision encoder, optimized attention and MLP
kernels, and startup-captured CUDA Graph profiles to reduce latency.

Install the required fast-backend dependency layer:

```bash
pip install -e ".[fast-infer]"
```

`fast-infer` installs `onnx`, `triton==3.6.0`, and `tensorrt`. The fast backend
does not fall back when these pieces are missing: TensorRT is used to prepare
the vision engine, Triton is required by the fast prefix/suffix kernels, and
`flex_attention` must be available in PyTorch.

To enable the optional prefix MLP FP8 optimization, first prepare both the
**vision FP16 engine and prefix MLP FP8 engines** offline:

```bash
python -m opendm.infer.prepare_fast \
  --checkpoint ./checkpoints/DM05-libero \
  --playground libero \
  --prefix-mlp-precision fp8 \
  --requests-jsonl ./calibration_requests.jsonl \
  --samples 128
```

The precision option applies only to the MLP. Vision stays FP16 and needs no FP8
calibration; MLP calibration uses the built vision engine and real deployment
inputs. The command prepares artifacts without starting a service. It currently
requires `--prefix-mlp-precision fp8` and has no BF16-only preparation mode.
Ordinary BF16 fast inference does not require this step. The old
`prepare_fast_fp8` command remains a compatible alias with FP8 selected.
See [one-command preparation](dm05_prefix_mlp_fp8.md#one-command-preparation)
for calibration data, custom output paths, and the current 1024-token limit.

Add `--inference-config.backend fast` to the matching default backend command.
When neither an explicit vision/FP8 path nor an external prepared directory is
set, startup first selects a compatible prepared FP8 generation under the
checkpoint. If none is selected, the common BF16 fast path uses the default
vision engine path `checkpoints/trt_engines/dm05_vision.engine`. For LIBERO:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_libero.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-libero \
  --model-config.chunk-size 10 \
  --inference-config.backend fast \
  --inference-config.output-action-dim 7 \
  --inference-config.image-prompts "Head" "Left wrist" \
  --inference-config.port 7891
```

On the BF16 fast path, if the vision engine does not exist, the launcher does not
start serving immediately. It first exports ONNX and builds the TensorRT engine,
then captures the configured CUDA Graph profiles before the HTTP service becomes
ready. Use a separate engine path for each checkpoint and image layout. An
existing engine is currently reused based on its image count, so changing the
checkpoint requires a new path or `--inference-config.force-rebuild`.

To build a vision FP16 engine manually for the BF16 fast path:

```bash
python -m opendm.infer.build_vision_trt \
  --checkpoint ./checkpoints/DM05-libero \
  --onnx-path checkpoints/trt_engines/dm05_vision.onnx \
  --engine-path checkpoints/trt_engines/dm05_vision.engine \
  --num-images 2
```

`--num-images` defaults to the number of `--inference-config.image-prompts`.
With `--data-config.is-history`, use
`len(image_prompts) + max_history_images`. The parent default is 5 history
slots (three current views: `--num-images 8`). The MEM demo playground uses 32
slots (`--num-images 35`); RoboDojo `cover_blocks` uses 20 (`--num-images 23`).

### Fast Backend Constraints

- Startup readiness includes TensorRT engine preparation and CUDA Graph profile
  capture; the first launch is expected to take longer than the default backend.
- Requests use batch size 1 and are processed serially by the service.
- `diffusion_steps` is fixed when the service captures its profiles at startup.
- The processed multimodal prefix is limited by
  `--inference-config.fast-prefix-len` (parent default `1024`). MEM demo and
  RoboDojo `cover_blocks` set this to `2048`.
- The parent default prefix buckets are `576 704 768 896 1024`. Defaults too
  small for the configured image count are skipped automatically. MEM demo and
  `cover_blocks` use a single bucket `2048`. The default backend does not use
  prefix buckets.
- RoboColiseum manipulation uses a `4096`-token limit on both backends. Its fast
  backend buckets default to `896 1024 2048 4096`.
- A custom bucket list must be non-empty, strictly increasing, and no larger
  than `fast_prefix_len`.
- More buckets increase service startup time and GPU memory use.
- On the BF16 fast path, a request longer than the largest custom bucket uses a
  slower eager fallback when it is still within `fast_prefix_len`. Prepared or
  explicitly configured FP8 requires every request to fit a configured bucket
  and rejects longer requests.

Override the defaults only when the workload requires different prefix shapes:

```bash
--inference-config.prefix-seq-len-buckets 576 704 768 896 1024
```

All configured profiles are captured before the HTTP service becomes ready.

## 5. Call the HTTP API

The service exposes `POST /v1/infer` with a JSON body. Images are sent as
base64 strings keyed by contiguous 1-based slot names, which keeps the API
consistent across benchmark checkpoints, demo checkpoints, and custom SFT
services.

`POST /process_frame` remains available as a legacy multipart endpoint for
existing clients. New integrations should use `/v1/infer`. The legacy endpoint
will be phased out over time.

A LIBERO request with two images looks like:

```bash
curl -X POST http://127.0.0.1:7891/v1/infer \
  -H 'Content-Type: application/json' \
  --data @- <<'EOF'
{
  "observation": {
    "prompt": "pick up the black bowl and place it on the plate",
    "state": [0, 0, 0, 0, 0, 0, 0, 0],
    "images": {
      "1": "<base64-agentview>",
      "2": "<base64-wrist>"
    },
    "robot_type": "Franka"
  }
}
EOF
```

Request fields:

- `observation.prompt`: task instruction. It defaults to an empty string.
- `observation.state`: required one-dimensional JSON array. Its length and
  ordering must match the checkpoint's normalization statistics.
- `observation.images`: required JSON object of base64-encoded images. Keys must
  be contiguous 1-based strings (`"1"`, `"2"`, …) and map **one-to-one in order**
  to `--inference-config.image-prompts` (e.g. `"1"` → first prompt such as
  `Head`, `"2"` → second such as `Left wrist`).
- `observation.history_images`: optional JSON array of base64 history frames,
  **oldest to newest**. The parent default cap is 5. Both Default and Fast honor
  `--inference-config.max-history-images`: MEM demo uses `32` (also set
  `--trainer-config.model-max-length 2048` and `fast_prefix_len=2048`); RoboDojo
  `cover_blocks` uses `20`. Only valid when the service was started with
  `--data-config.is-history`; omit or use `[]` when there is no history.
- `observation.robot_type`: optional robot embodiment used for state/action
  semantics. Benchmark entry points inherit dataset defaults such as `Franka`
  and `Aloha RoboTwin2`. Multi-robot checkpoints select profiles by the exact
  value, for example `Aloha` or `DOS W1`; custom relative-action entries may
  need this field explicitly.
- `observation.control_mode` and `observation.speed`: text-conditioning fields.
  Provide both explicitly when serving the `Dexmal/DM05` base pretrained
  checkpoint. For SFT checkpoints, they are needed only when the training data
  included them. The service defaults `speed` to `"0.5"`.
- `sampling`: optional JSON object. `num_steps` must match the service's fixed
  diffusion steps, and `seed` can be used for deterministic sampling.

A successful response contains the action chunk and end-to-end API latency in
milliseconds:

```json
{
  "actions": [
    [0.012, -0.034, 0.18, 0.0, 0.0, 0.0, -1.0],
    [0.015, -0.031, 0.17, 0.0, 0.0, 0.0, -1.0]
  ],
  "metadata": {
    "latency_ms": 123.4
  }
}
```

### VLM Chat Completions API

`POST /v1/chat/completions` implements the OpenAI Chat Completions response
schema for multimodal VLM evaluation. Unlike `/v1/infer`, this endpoint returns
text, does not require robot state, accepts any number of images interleaved
with text, and does not add the `Robot:`, `Control mode:`, `Overall speed:`,
`Task:`, or `States:` action-inference prefixes. Both endpoints reuse the same
loaded checkpoint and processor.

A request can contain plain-string content or OpenAI-style content parts:

```bash
curl -X POST http://127.0.0.1:7891/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data @- <<'EOF'
{
  "model": "dm05",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {
          "type": "image_url",
          "image_url": {"url": "data:image/png;base64,<base64-image>"}
        }
      ]
    }
  ],
  "max_tokens": 128,
  "temperature": 0
}
EOF
```

Images must be data URLs or raw base64 strings. External HTTP(S) image URLs
are rejected, and video/audio parts are not supported. `max_tokens` and
`max_completion_tokens` are both accepted; the default is 1024. A zero or
omitted `temperature` uses greedy decoding, while a positive value enables
sampling. `stop` accepts a string or string array. Unknown top-level fields are
ignored for lmms-eval compatibility, but `stream: true` is rejected because the
server does not implement SSE.

A successful response follows the OpenAI schema:

```json
{
  "id": "chatcmpl-1a2b3c4d",
  "object": "chat.completion",
  "created": 1785000000,
  "model": "dm05",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "A robot workcell..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 812,
    "completion_tokens": 37,
    "total_tokens": 849
  },
  "metadata": {"latency_ms": 1234.5}
}
```

VLM chat configuration:

- `--inference-config.enable-vlm-chat`: enable the endpoint; defaults to `true`.
- `--inference-config.chat-max-new-tokens`: server-side output cap; defaults to
  2048. Larger request values are clamped with a warning.
- `--inference-config.chat-max-input-tokens`: maximum tokenized prompt length;
  defaults to 8192. Longer prompts return HTTP 400.

With `--inference-config.backend fast`, action inference still uses the fast
runtime, but chat generation deliberately uses the standard PyTorch VLM path.
If chat generation fails in that configuration, retry with
`--inference-config.backend default`.

To run an image benchmark through the lmms-eval OpenAI backend:

```bash
OPENAI_API_KEY=EMPTY python3 -m lmms_eval \
  --model openai \
  --model_args model=dm05,base_url=http://127.0.0.1:7891/v1,api_key=EMPTY,num_concurrent=1 \
  --tasks mme \
  --batch_size 1 \
  --log_samples \
  --output_path ./lmms_results/dm05
```

Use `num_concurrent=1` to `4` for this service. For tasks whose `max_tokens` is
in the thousands, always use `num_concurrent=1`: Flask runs this server serially
with `threaded=False`. With lmms-eval's default 32-way concurrency, requests at
the back of the queue can hit the OpenAI SDK's default 600-second timeout, and
the resulting retries can overload the queue further.

### Legacy `/process_frame` API

The legacy compatibility endpoint accepts `multipart/form-data` with repeated
`image` file fields. Prefer `/v1/infer` JSON for new clients; legacy uses repeated
form fields instead of `"1"` / `"2"` keys.

```bash
curl -X POST http://127.0.0.1:7891/process_frame \
  -F 'text=pick up the black bowl and place it on the plate' \
  -F 'states=[0,0,0,0,0,0,0,0]' \
  -F 'robot_type=Franka' \
  -F image=@/path/to/agentview.jpg \
  -F image=@/path/to/wrist.jpg
```

Legacy request fields:

- `text`: task instruction. It defaults to an empty string.
- `states`: required one-dimensional JSON array. Its length and ordering must
  match the checkpoint's normalization statistics.
- `image`: repeated image file field. The count and order must match
  `image_prompts` (e.g. Head, Left wrist, Right wrist).
- `history_images`: optional repeated history frames (oldest to newest). The
  parent default cap is 5; both backends follow
  `--inference-config.max-history-images` (MEM demo `32`, `cover_blocks` `20`).
  Only valid when the service was started with `--data-config.is-history`.
- `robot_type`: optional robot embodiment used for state/action semantics.
  Benchmark entry points inherit dataset defaults such as `Franka` and `Aloha
  RoboTwin2`. Multi-robot checkpoints select profiles by the exact value, for
  example `Aloha` or `DOS W1`; custom relative-action entries may need this
  field explicitly.
- `control_mode` and `speed`: text-conditioning fields. Provide both explicitly
  when serving the `Dexmal/DM05` base pretrained checkpoint. For SFT
  checkpoints, they are needed only when the training data included them. The
  service defaults `speed` to `"0.5"`.

A successful legacy response returns the historical shape:

```json
{
  "response": [
    [0.012, -0.034, 0.18, 0.0, 0.0, 0.0, -1.0],
    [0.015, -0.031, 0.17, 0.0, 0.0, 0.0, -1.0]
  ],
  "model_latency_ms": 71.884
}
```

## 6. Common Parameters

| Parameter | Meaning |
| --- | --- |
| `--exp` | Playground entry point matching the checkpoint and task. |
| `--model-config.model-name-or-path` | Full model or LoRA checkpoint directory. |
| `--model-config.chunk-size` | Action horizon; must match training and the client. |
| `--trainer-config.model-max-length` | Maximum tokenized multimodal prefix length. |
| `--inference-config.diffusion-steps` | Number of action diffusion steps; default `10`. |
| `--inference-config.output-action-dim` | Internal normalized action dimension. It normally equals the API action dimension; both RoboColiseum entries keep it at `22` and remap the response to 16-D or 21-D. |
| `--inference-config.image-prompts` | Ordered camera labels, one-to-one with `observation.images` keys `"1"`, `"2"`, …. |
| `--inference-config.max-history-images` | History request cap for both backends. Parent default `5`; MEM demo `32`; RoboDojo `cover_blocks` `20`. |
| `--inference-config.fast-prefix-len` | Fast backend prefix token capacity. Parent default `1024`; MEM demo and `cover_blocks` use `2048`; RoboColiseum manipulation uses `4096`. |
| `--data-config.is-history` | Required to accept `history_images`; plain `curl_demo.sh` does not need it. |
| `--inference-config.backend` | `default` or `fast`. |
| `--inference-config.vision-trt-engine-path` | Explicit checkpoint-specific TensorRT vision engine path; bypasses prepared auto-selection. The BF16 fallback default is `checkpoints/trt_engines/dm05_vision.engine`. |
| `--inference-config.force-rebuild` | Rebuild the vision engine before fast inference. |
| `--inference-config.prefix-seq-len-buckets` | Optional non-empty custom fast-backend buckets; each value must be `<= fast_prefix_len`. |
| `--inference-config.no-auto-prepared-fast-artifacts` | Disable the default automatic selection of a compatible checkpoint-local offline FP8 generation. |
| `--inference-config.prepared-fast-artifacts-dir` | External prepared generation root produced by `prepare_fast --prefix-mlp-precision fp8` (legacy alias: `prepare_fast_fp8`); includes vision FP16 and prefix MLP FP8 engines. |
| `--inference-config.port` | HTTP service port; default `7891`. |

## 7. Troubleshooting

| Error or symptom | Check |
| --- | --- |
| Missing or mismatched normalization statistics | Use the checkpoint's `norm_stats.json` and the same dataset/action/chunk configuration as training. |
| State or action dimension error | Match `observation.state` and `output_action_dim` to the normalization vectors. |
| Wrong number of uploaded images | Make `observation.images` use the same count and order as `image_prompts`. |
| Fast backend fails during startup with import errors | In the active environment, reinstall `pip install -e ".[fast-infer]"` and verify `import tensorrt`, `import triton`, and `import torch.nn.attention.flex_attention`. |
| TensorRT image-count mismatch | Rebuild with `--num-images` equal to `len(image_prompts)`; with history use `len(image_prompts) + max_history_images`. |
| Results change after switching checkpoints with the same engine | Use a checkpoint-specific engine path or pass `--inference-config.force-rebuild`. |
| Expected BF16 but startup enabled prepared FP8 | Pass `--inference-config.no-auto-prepared-fast-artifacts`; explicit engine paths still take precedence. |
| Empty, unsorted, or oversized prefix buckets | Pass a non-empty increasing list whose values are at most `fast_prefix_len`. |
| Fast prefix exceeds the largest bucket | BF16 may run it eagerly up to `fast_prefix_len`; FP8 rejects it, so use a prepared profile that covers the request or disable FP8 auto-selection. |
| An occasional default-backend request is slower | A new execution profile runs eagerly once and captures on its second occurrence; later matching requests reuse the Graph. |
| Fast service takes time to become ready | Wait for engine preparation and all configured CUDA Graph profiles to finish at startup. |

## 8. Related Guides

- [DM05 LIBERO Training and Evaluation](dm05_libero.md)
- [DM05 RoboTwin 2.0 Training and Evaluation](dm05_robotwin2.md)
- [DM05 RoboDojo-Sim Training and Evaluation](dm05_robodojo.md)
- [DM05 SFT and Validation](dm05_finetuning.md)
- [DM05-MEM SFT and Validation](dm05_mem_sft.md)
- [DM05 LIBERO LoRA Training](dm05_libero_lora_training.md)
