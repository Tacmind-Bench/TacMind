# DM05 VLA-Arena Training and Evaluation Guide

This document describes how to use `DM05` for model training, inference service startup, and benchmark evaluation in VLA-Arena scenarios.

## VLA-Arena Training

### Prerequisites

Before training, make sure the following preparation is complete:

- **OpenDM environment installation and source initialization have been completed according to the official steps.**
- Training and inference require GPU resources. Recommended GPUs include A100, H100, H20, and 4090. On lower-memory GPUs such as RTX 4090, Full SFT may fail due to peak memory use during FSDP initialization; LoRA fine-tuning is recommended.

### Data Preparation

The VLA-Arena dataset and the base model can be downloaded from Hugging Face:

- VLA-Arena dataset: [Dexmal/vla_arena_L0_L](https://huggingface.co/datasets/Dexmal/vla_arena_L0_L)
- DM05 model: [Dexmal/DM05](https://huggingface.co/Dexmal/DM05)

Download and organize the data in the OpenDM project root:

```bash
# Run from the OpenDM repository root.
cd opendm

# Download the VLA-Arena dataset
huggingface-cli download Dexmal/vla_arena_L0_L --repo-type dataset \
  --local-dir ./data/vla_arena_L0_L

# Extract split tar archives
cd ./data/vla_arena_L0_L
cat vla_arena_L0_L.tar.gz.part-* | tar -xzf -
cd -

# Download the DM05 base model
huggingface-cli download Dexmal/DM05 --local-dir ./checkpoints/DM05
```

Confirm that the data directory is ready. The dataset layout should be:

```text
├── data
│   └── vla_arena_L0_L
│       ├── jsonl
│       │   ├── episode_00000.jsonl
│       │   ├── episode_00001.jsonl
│       │   └── ...
│       └── images
│           ├── episode_00000
│           │   ├── episode_00000_top_0000.png
│           │   ├── episode_00000_wrist_0000.png
│           │   └── ...
│           ├── episode_00001
│           └── ...
└── ...
```

Prepare the dataset registration file and confirm how normalization statistics are generated:

```text
# Dataset registration file
opendm/dataset/vla_arena.py
```

Notes:

- `opendm/dataset/vla_arena.py` registers the VLA-Arena dataset. The default registered dataset name is `vla_arena_eef_L0_L`.
- When training starts, if the corresponding normalization statistics file does not exist, the script automatically computes it based on the current dataset, action mode, and action chunk length, then saves it under `./norm_stats/`.
- When saving a checkpoint, the normalization statistics used for training are copied to `norm_stats.json` under the checkpoint directory. Inference first reads `norm_stats.json` from the checkpoint directory. If it does not exist, inference looks for the matching file under `./norm_stats/` based on the current dataset, action mode, and action chunk length.
- `--data-config.dataset-name` in the training command must match a dataset name registered in `opendm/dataset/vla_arena.py`.

### Start Training

After the data, model, and registration file are ready, start VLA-Arena training:

```bash
# Run from the OpenDM repository root.
cd opendm

script/dm05_launcher.sh \
  --exp ./playground/dm05_vla_arena.py \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name vla_arena_eef_L0_L \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --trainer-config.num-train-steps 60000
```

Arguments:

- `--exp ./playground/dm05_vla_arena.py`: training entry for VLA-Arena scenarios. This entry presets the action mode and default chunk size required by VLA-Arena.
- `--task train`: run in training mode.
- `--nproc_per_node 8`: number of GPUs used on a single node. 8 GPUs are recommended.
- `--data-config.dataset-name vla_arena_eef_L0_L`: VLA-Arena dataset name used for training.
- `--model-config.model-name-or-path ./checkpoints/DM05`: base model checkpoint path.
- `--trainer-config.num-train-steps 60000`: total number of training steps.

## VLA-Arena Inference

**After completing environment installation and source initialization**, you can start a model inference service with the DM05 VLA-Arena checkpoint trained in the previous step.

### Prerequisites

Before inference, make sure **OpenDM environment installation and source initialization have been completed according to the official steps.** Inference requires at least 1 GPU.

### Start Inference

- Inference first reads `norm_stats.json` from the model checkpoint directory. Make sure the checkpoint comes from a VLA-Arena training workflow that matches the current inference configuration.

```bash
# Start the inference service
script/dm05_launcher.sh \
  --exp playground/dm05_vla_arena.py \
  --task inference \
  --nproc_per_node 1 \
  --model-config.model-name-or-path ./checkpoints/DM05-vla-arena-checkpoint \
  --inference-config.output-action-dim 7
```

Arguments:

- `--exp playground/dm05_vla_arena.py`: inference entry for VLA-Arena scenarios, consistent with training.
- `--task inference`: task type. Use `inference` for inference.
- `--nproc_per_node 1`: number of GPUs used on a single node. 1 GPU is sufficient for inference.
- `--model-config.model-name-or-path ./checkpoints/DM05-vla-arena-checkpoint`: model checkpoint path.
- `--inference-config.output-action-dim 7`: VLA-Arena action output dimension.

Keep the inference service running after startup; the evaluation script communicates with it through an HTTP API.

## VLA-Arena Evaluation

### Preparation

1. Use at least 2 GPUs for the evaluation workflow when possible: one GPU for the OpenDM inference service and another for VLA-Arena simulation and EGL rendering.
2. Start the DM05 inference service according to the steps above, and keep it running. dexbotic-benchmark sends HTTP requests to this service through `base_url`, for example `http://<SERVER_IP>:7891`; do not append `/process_frame`.
3. Configure the benchmark environment. VLA-Arena evaluation uses the Arena runner in `dexbotic-benchmark`; copying OpenDM evaluation files into the upstream VLA-Arena repository is no longer needed.

```bash
# Get evaluation scripts and configuration files.
git clone https://github.com/dexmal/dexbotic-benchmark.git
cd dexbotic-benchmark

# If only evaluating VLA-Arena, initialize only the Arena submodule.
git submodule update --init --recursive arena
```

4. Create the local VLA-Arena evaluation environment. Arena evaluation currently runs in a local Conda environment. For complete system dependency details, see the VLA-Arena Environment section in `dexbotic-benchmark/docs/local_install.md`.

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config ffmpeg \
  libegl1 libegl1-mesa libegl1-mesa-dev libgl1 libglvnd-dev \
  libosmesa6-dev libsm6 libxext6 libxrender-dev

conda create -n arena python=3.11 pip -y
conda activate arena
python -m pip install --upgrade pip setuptools wheel

python -m pip install torch==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -e arena
python -m pip install \
  numpy==1.26.4 opencv-python-headless==4.11.0.86 mujoco==3.10.0 \
  omegaconf requests tqdm Pillow PyYAML imageio-ffmpeg

conda env config vars set -n arena \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  EGL_PLATFORM=device
conda deactivate
conda activate arena
```

### Evaluation

#### Modify the Evaluation Configuration

Use and modify the VLA-Arena example configuration at `evaluation/configs/arena/example_arena.yaml`:

```yaml
# Task selection. all evaluates the non-LIBERO Arena suites.
task_suite_name: all
task_level: all
parallelized: false
num_trials_per_task: 10
num_steps_wait: 10
seeds: [7, 42, 1000]

# Simulation environment and robustness switches.
env_img_res: 256
apply_mujoco3_patch: true
add_noise: false
adjust_light: false
randomize_color: false
camera_offset: false
safety: false
use_pruned_init: false
init_state_selection_mode: first
init_state_offset: 0
init_state_offset_random: false
rotate_images_180: false

# OpenDM inference service.
base_url: http://localhost:7891
api_style: legacy
replan_step: 10
send_state: true
send_image:
  - image
  - wrist_image
discrete_gripper: false
discrete_action_dims: [0, 1, 2, 6]
discrete_action_threshold: 0.5
clip_actions: false
use_text_template: false

# Legacy multipart request parameters.
batch_size: 1
speed: "0.5"
action_horizon: 20
request_timeout: 30

# Instruction robustness.
replacements_file: VLA-Arena/language_replacements
use_replacements: false
replacement_probability: 1.0
replacement_level: 1

# Output.
save_video_mode: first_success_failure
video_fps: 30
result_json_path: default
output_dir: experiments/eval_results
log_level: INFO
```

Config values you may modify include:

- `base_url`: running DM05 inference service address. Usually `http://<SERVER_IP>:7891`. dexbotic-benchmark appends `/process_frame` automatically when `api_style: legacy` is used.
- `api_style`: keep `legacy` for the OpenDM VLA-Arena multipart service.
- `task_suite_name`: task suite to evaluate. Set to `all` to run all Arena suites, or specify a suite such as `safety_static_obstacles` or `extrapolation_unseen_objects`.
- `task_level`: task level to evaluate. Valid values are `0`, `1`, `2`, and `all`.
- `parallelized`: when `true`, starts one evaluator process for each of L0, L1, and L2. It requires `task_level: all`, and the inference service must handle three concurrent request streams.
- `seeds`: list of random seeds. For CLI overrides, use a comma-separated value such as `--set seeds 7,42,1000`.
- `num_trials_per_task`: number of episodes per task.
- `replan_step`: number of predicted actions executed before requesting the model again. The default is 10.
- `action_horizon`: action chunk length returned by the service. Keep it aligned with the inference service `--model-config.chunk-size`.
- `use_replacements`: whether to enable VLA-Arena instruction replacement robustness testing. The default is `false`, matching the reference config.
- `save_video_mode`: video saving mode. Valid values include `all`, `first_success_failure`, and `none`.
- `output_dir`: directory for evaluation results.

#### Start VLA-Arena Evaluation

Run the evaluation from the `dexbotic-benchmark` repository root:

```bash
cd dexbotic-benchmark
conda activate arena

python3 evaluation/run_arena_evaluation.py \
  --config evaluation/configs/arena/example_arena.yaml \
  --set base_url http://<SERVER_IP>:7891 \
  --set output_dir experiments/eval_results
```

To evaluate L0, L1, and L2 concurrently, enable `parallelized`:

```bash
python3 evaluation/run_arena_evaluation.py \
  --config evaluation/configs/arena/example_arena.yaml \
  --set task_level all \
  --set parallelized true \
  --set base_url http://<SERVER_IP>:7891 \
  --set output_dir experiments/eval_results_parallel
```

Parallel mode shares one inference service. If the server uses process-global stochastic sampling, default-session memory, or cannot handle concurrent requests, results may differ from serial evaluation. Use the default serial mode when strict trajectory reproduction is required.

#### Check Evaluation Results

Evaluation results are saved under `output_dir`. Main files include:

- `results.json`: detailed and summary results for the current run.
- `leaderboard.json`: results in the VLA-Arena leaderboard format.
- `config.yaml`: resolved evaluation configuration.
- `logs/evaluation.log`: evaluation log.
- `videos/`: rollout videos saved according to `save_video_mode`.
- `seed_<N>/`: per-seed result directories for multi-seed evaluation.
- `batch_summary_<timestamp>.json`: multi-seed SR/CC aggregate statistics.
- `parallel_levels/`: intermediate outputs for per-level workers in parallel evaluation.

With the default `seeds` configuration, each seed's `leaderboard.json` and
`videos/` are saved under `seed_<N>/`; the top-level `results.json` aggregates
the seed runs.

After evaluation finishes, first inspect `results.json` and `batch_summary_<timestamp>.json`. For safety suites, CC is cumulative cost.
