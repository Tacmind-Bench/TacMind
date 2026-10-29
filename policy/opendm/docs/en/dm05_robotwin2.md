# DM05 RoboTwin 2.0 Training and Evaluation Guide

Train, serve, and evaluate `DM05` on RoboTwin 2.0.

## Reference Results

| Method | Clean | Randomized | Average |
| --- | ---: | ---: | ---: |
| DM0.5 | 93.6 | 93.3 | 93.5 |

## RoboTwin 2.0 Training

### Prerequisites

- Install and initialize OpenDM following the official instructions.
- Use NVIDIA GPUs, such as A100, H100, H20, or RTX 4090. Full SFT may run out
  of memory during FSDP initialization on lower-memory GPUs such as RTX 4090;
  use LoRA fine-tuning in that case.
- Install the Hugging Face CLI if `hf` is unavailable:

```bash
pip install -U huggingface_hub
```

### Data Preparation

Download the dataset and base model from Hugging Face:

- Complete RoboTwin 2.0 dataset: [Dexmal/robotwin2-full](https://huggingface.co/datasets/Dexmal/robotwin2-full)
- DM05 model: [Dexmal/DM05](https://huggingface.co/Dexmal/DM05)

Prepare the dataset and model from the OpenDM repository root:

```bash
cd opendm

# Download all parts of the RoboTwin 2.0 dataset archive.
mkdir -p data/.hf_downloads/robotwin
hf download Dexmal/robotwin2-full \
  --repo-type dataset \
  --local-dir data/.hf_downloads/robotwin

# Join and extract the archive to ./data/robotwin2.0.
cat data/.hf_downloads/robotwin/robotwin2.tar.part-* \
  | tar -xf - -C data

# Download the DM05 base checkpoint.
hf download Dexmal/DM05 --local-dir checkpoints/DM05
```

Expected layout (`<setting>` is `clean` or `randomized`):

```text
data/robotwin2.0/
├── jsonl/<task>/<setting>/
└── video/<task>/<setting>/
```

`opendm/dataset/robotwin2.py` registers this archive as `robotwin2_generalist`
with root `./data/robotwin2.0`. The default recipe uses 100 weighted views named
`robotwin2_component_<task>_<setting>`, covering all 50 tasks in both settings.
Each view reads `jsonl/<task>/<setting>/` and shares the `video/` root.

The ALOHA RoboTwin2 recipe uses:

- Head, left-wrist, and right-wrist RGB views (`images_1`, `images_2`, `images_3`).
- Absolute joint-position targets from future `state` values, ignoring raw `action`.
- Native 14D state/action validation and normalization without quantile clipping;
  only normalized actions and masks are padded to 32D.
- Task/subtask prompt sampling and weighted sampling of datasets, then samples.

Missing normalization statistics are computed automatically at startup and saved
to `<norm-stats-root>/norm_stats.json` (by default
`./norm_stats/robotwin2_absolute_action/14d/norm_stats.json`). Override the
directory with `--data-config.norm-stats-root`, or compute statistics before training:

```bash
python playground/dm05_robotwin2.py --task compute_norm_stats
```

Use the same dataset selection and paths as training. Statistics are copied to
each checkpoint as `norm_stats.json`; keep this file with the weights, as
inference reads it first.

### Start Training

The RoboTwin 2.0 reference training configuration must use 4 nodes, with 8 GPUs
per node. Run the following from the OpenDM repository root on each node,
setting `<NODE_RANK>` to `0`, `1`, `2`, and `3` respectively:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_robotwin2.py \
  --task train \
  --nproc_per_node 8 \
  --nnodes 4 \
  --node_rank <NODE_RANK> \
  --master_addr <MASTER_ADDR> \
  --master_port 29500 \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --trainer-config.per-device-train-batch-size 32 \
  --trainer-config.gradient-accumulation-steps 1 \
  --trainer-config.num-train-steps 100000
```

Set `<MASTER_ADDR>` to the rank-0 node's address and ensure port `29500` is
reachable from every node.

The global batch size must be `1024`, with `gradient_accumulation_steps=1`:

```text
global batch size = number of nodes x GPUs per node x per-device train batch size x gradient accumulation steps
                  = 4 x 8 x 32 x 1
                  = 1024
```

This configuration cannot be changed to single-node training. If you adjust
the GPU count or per-device batch size, retain 4 nodes, a global batch size of
`1024`, and `gradient_accumulation_steps=1`.

To sample the archive as one dataset instead of using the default weighted
views, pass both `--data-config.dataset-name robotwin2_generalist` and
`--data-config.dataset-weights 1.0`.

Defaults: output directory `user_checkpoints/dm05_robotwin2`, absolute actions,
chunk size 50, maximum token length 1152, FSDP1, BF16, and Muon AdamW. Override
trainer settings with the corresponding CLI options, preserving the batch-size
constraints above. Keep chunk size consistent across training, inference, and
evaluation, and use the checkpoint's normalization statistics for inference.

## RoboTwin 2.0 Inference

See the [DM05 Inference Guide](dm05_inference.md) for the released RoboTwin 2.0 checkpoint, the complete service command, fast backend setup, and HTTP API usage.

## RoboTwin 2.0 Evaluation

### Prepare dexbotic-benchmark

Use separate GPUs for inference and simulation when possible. Prepare the
official benchmark Docker image:

```bash
git clone https://github.com/dexmal/dexbotic-benchmark.git
cd dexbotic-benchmark
git submodule update --init --recursive RoboTwin
docker pull dexmal/dexbotic_benchmark
```

Before evaluation, download the assets, object-data texture library, and
embodiments following the
[RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html#4-download-assets-robotwin-od-texture-library-and-embodiments).

### Configure an Evaluation Task

Use `evaluation/configs/robotwin2/adjust_bottle.yaml` as the reference
configuration. Set `base_url` to the inference server address:

```yaml
# Basic experiment configuration (keep unchanged)
policy_name: dexbotic
task_name: adjust_bottle
task_config: demo_clean
ckpt_setting: dexbotic
seed: 0
instruction_type: seen

# Add Parameters You Need
base_url: http://localhost:7891
output_dir: ./result_test/robotwin2_evaluation
cameras: "head_camera_rgb,left_camera_rgb,right_camera_rgb"
action_horizon: 50
action_mode: absolute
```

Important fields:

- `task_name`: one of 50 tasks, evaluated independently.
- `task_config`: `demo_clean` or `demo_randomized`.
- `base_url`: DM05 service address; use `http://<SERVER_IP>:7891` for a remote service.
- `cameras`: preserve head, left-wrist, right-wrist order to match `images_1/2/3`.
- `action_horizon`: actions returned and executed per request; must match chunk size 50.
- `action_mode`: `absolute` for this checkpoint (absolute joint positions).
- `output_dir`: result and rollout video directory.

Each task/setting runs 100 episodes. For aggregate Clean and Randomized scores,
evaluate all 50 tasks in both settings and aggregate their success rates.

### Run Evaluation with Docker

From the `dexbotic-benchmark` repository root, set `ROBOTWIN_ASSETS` to the
absolute path of the downloaded RoboTwin assets and run:

```bash
ROBOTWIN_ASSETS=/absolute/path/to/robotwin/assets

docker run --rm --gpus all --network host \
  -v "$ROBOTWIN_ASSETS":"$ROBOTWIN_ASSETS" \
  -v "$ROBOTWIN_ASSETS":/app/assets \
  -v "$ROBOTWIN_ASSETS":/app/RoboTwin/assets \
  -v "$PWD/evaluation":/app/evaluation \
  -v "$PWD/scripts":/app/scripts \
  -v "$PWD/result_test":/app/result_test \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  dexmal/dexbotic_benchmark \
  bash scripts/env_sh/robotwin2.sh \
  evaluation/configs/robotwin2/adjust_bottle.yaml
```

### Run Evaluation Locally

For simulator debugging, install the environment using
`dexbotic-benchmark/docs/local_install.md`, then run:

```bash
cd dexbotic-benchmark
conda activate RoboTwin

# Recommended entry point.
bash scripts/env_sh/robotwin2.sh \
  evaluation/configs/robotwin2/adjust_bottle.yaml

# Or invoke Python directly, optionally overriding configuration values.
python evaluation/run_robotwin2_evaluation.py \
  --config evaluation/configs/robotwin2/adjust_bottle.yaml \
  --set base_url http://localhost:7891 \
  --set output_dir ./result_test/robotwin2_evaluation
```

### Check Evaluation Results

Results are written below `output_dir` using the following hierarchy:

```text
<output_dir>/<task_name>/<task_config>/<timestamp>/
├── _result.txt
└── *.mp4
```

`_result.txt` records the task/setting success rate. Rollout videos are saved
alongside it when video logging is enabled in the RoboTwin task configuration.
