<h1 align="center"><img src="docs/media/tacmind.svg" width="42" height="42" alt=""> TacMind</h1>

<p align="center"><b>Benchmarking and Modeling Tactile Interaction States<br>for Robotic Manipulation</b></p>

<p align="center">
  Jiaxin Shi<sup>1,4</sup> · Zibin Wu<sup>2,4</sup> · Yichi Zhang<sup>3,4</sup> · Jianghua Duan<sup>4</sup> · Siyu Zhu<sup>5</sup> · Weihao Yuan<sup>4,6</sup>
</p>

<p align="center">
  <sup>1</sup> Shanghai Jiaotong University · <sup>2</sup> The Hong Kong University of Science and Technology<br>
  <sup>3</sup> Harbin Institute of Technology · <sup>4</sup> Daimon Robotics<br>
  <sup>5</sup> Fudan University · <sup>6</sup> Nanjing University
</p>

<p align="center">
  <a href="https://tacmind-bench.github.io/"><img src="docs/media/link-website.svg" height="40" alt="Project website"></a>
  <a href="https://tacmind-bench.github.io/#leaderboard"><img src="docs/media/link-leaderboard.svg" height="40" alt="Leaderboard"></a>
  <a href="https://huggingface.co/datasets/JiaxinShi/TacMind-Bench"><img src="docs/media/link-huggingface.svg" height="40" alt="Hugging Face datasets"></a>
  <a href="https://modelscope.cn/datasets/shijiaxin/TacMind-Bench"><img src="docs/media/link-modelscope.svg" height="40" alt="ModelScope datasets"></a>
</p>

<p align="center">English · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center">
  <a href="#benchmark">Benchmark</a> · <a href="#tacmind-0">TacMind-0</a> · <a href="#results">Results</a> · <a href="#usage-guide">Usage guide</a>
</p>

https://github.com/user-attachments/assets/8b76b294-3eef-4678-963d-1d717e91d473

<p align="center"><b>Whiteboard wiping.</b> Two camera views and both fingertips, synchronized in one episode.<br>Touch is shown as RGB, marker motion and deformation. <a href="https://tacmind-bench.github.io/#demo">Watch the demonstration ↗</a></p>

<a id="benchmark"></a>
## <img src="docs/media/icon-benchmark.svg" width="28" height="28" alt=""> TacMind-Bench

**TacMind-Bench studies tactile interaction across 128 robotic manipulation tasks.**
Built on Isaac Sim and Isaac Lab, the released simulation pipeline uses a Franka arm
with two GelSight Mini sensors. It records external and wrist RGB, tactile RGB,
marker motion, robot states and language instructions together.

The repository provides expert demonstration collection, conversion to LeRobot datasets,
training for seven policies, and shared task-success evaluation. **TacMind-0** is
our policy for modeling tactile interaction states and using them to guide manipulation.

### Task collection

| Suite | Tasks | Capabilities and examples |
| --- | ---: | --- |
| **Precision** | 6 | Fine contact control: insert tubes, keys, connectors, gears, plugs and bulbs |
| **Recog** | 11 | Resolve visual ambiguity through touch: distinguish size, shape and surface texture, including in covered scenes |
| **Dynamic** | 11 | Manipulate moving objects or targets, including insertion into fixtures on a conveyor |
| **Extra** | 100 | Broader contact behaviors: open drawers, turn valves, cut objects and wipe surfaces |

**Standard = Precision + Recog + Dynamic (28 tasks).** The 100 Extra tasks expand the
range of objects and interaction skills. The [interactive task gallery](https://tacmind-bench.github.io/#tasks)
includes synchronized sensor views and additional Daimon demonstrations for Standard tasks.

### Evaluation conditions

Standard tasks support nominal evaluation and **12 controlled perturbations** applied
at evaluation time. Extra tasks support nominal evaluation.

| Perturbation group | Conditions | What changes |
| --- | ---: | --- |
| **Lighting** | 4 | Warm, cool, fluorescent and night lighting |
| **Camera** | 6 | Left/right viewpoints, high/low elevation and two farther viewpoints |
| **Object scale** | 2 | Objects scaled to 0.85× and 0.90× |

Use the same held-out seeds across models and conditions. Results report success rates,
with suite-balanced **InD** (nominal) and **OOD** (perturbed) averages; the complete scoring
procedure and commands are in [Robustness evaluation](#robustness-evaluation).

<a id="tacmind-0"></a>
## <img src="docs/media/icon-model.svg" width="28" height="28" alt=""> TacMind-0

TacMind-0 combines tactile interaction-state modeling with vision, language and robot
state to generate actions. Its tactile input is a history of **left/right Deformation images**
showing marker displacement as arrows.

- **Tac-LeWM** learns tactile interaction representations and action-conditioned predictions of future tactile features. World-model training also uses visual context.
- **VTLA policy** combines these tactile representations with external/wrist images, the language instruction and robot state to generate action sequences.
- **Tac-FRS** predicts future tactile features under candidate actions and uses those predictions to modulate subsequent action generation.

Training starts with Tac-LeWM pretraining on multitask demonstrations. The target-task
pipeline then adapts the tactile model, trains the VTLA policy and trains Tac-FRS.
The [training guide below](#train) includes environments, pretrained dependencies,
commands and checkpoint paths for this complete sequence.

<a id="results"></a>
## <img src="docs/media/icon-results.svg" width="28" height="28" alt=""> Benchmark results

Success rate (%) on the **GelSight Mini Standard tasks**. Higher is better; bold values
mark the best score in each column. The three suite columns show nominal performance.

| Model | Precision InD | Recog InD | Dynamic InD | Avg InD | Avg OOD |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACT+Tactile† | 30.0 | **69.1** | **15.9** | **39.8** | 16.4 |
| DP† | 23.3 | 27.7 | 13.2 | 21.1 | 8.1 |
| OpenPI0.5 | 5.0 | 30.0 | 0.5 | 13.0 | 5.5 |
| OpenDM0.5 | 20.0 | 38.6 | 8.2 | 22.7 | 11.4 |
| N0-VTLA | 25.8 | 45.9 | 5.9 | 25.9 | 7.8 |
| **TacMind-0** | **40.0** | 50.0 | 10.5 | 32.3 | **20.7** |

† **Single-task training:** ACT+Tactile and DP train a separate policy for each task.
The [full leaderboard](https://tacmind-bench.github.io/#leaderboard) also reports lighting,
camera and object-scale scores for each suite.

<a id="usage-guide"></a>

## Usage guide

[Installation](#installation) · [Data](#data) · [Train TacMind-0](#train) · [Evaluation](#policy-evaluation) · [Other policies and configs](#policies) · [Advanced usage](#advanced)

The example below uses Precision 0: install → prepare data → train → evaluate nominal and perturbed conditions. Run all commands from the repository root.

<a id="installation"></a>
## <img src="docs/media/icon-install.svg" width="28" height="28" alt=""> Install the environment and assets

Linux x86_64, Conda and Git LFS are required; collection and evaluation also need
an Isaac-compatible GPU. Run from the repository root.
Environment versions: Ubuntu 22.04, CUDA 12.8, Isaac Lab 2.3.2 and Isaac Sim 5.1.0.

### Base environment

```bash
git clone https://github.com/Tacmind-Bench/TacMind.git
cd TacMind
conda create -n tacmind python=3.11.16 pip
conda activate tacmind
git lfs install
git lfs pull
python -m pip install -e .
```

<a id="training-and-evaluation"></a>
### Learning dependencies

For conversion, ACT/DP training and policy evaluation. Other models need
[their environments](#source-and-environment-setup).
Install system `ffmpeg`, then run in a fresh terminal:

```bash
conda activate tacmind
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r benchmark/config/learning.txt
python -m pip install --no-deps lerobot==0.4.4
```

<a id="simulation-environment"></a>
### Simulation dependencies

Install the NVIDIA driver and CUDA 12.8 toolkit with `nvcc`.
System tools: `cmake`, `build-essential`, `ffmpeg`, `rsync`, `git-lfs`, `unzip`.
Download the **Isaac Sim 5.1.0 Linux x86_64 standalone ZIP** from the
[official download page](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
and save it as `data/downloads/isaac-sim-standalone-5.1.0-linux-x86_64.zip`.
Extract it into a new directory and run its setup script:

```bash
export TACMIND_ISAAC_SIM="$HOME/isaacsim-5.1.0"
mkdir -p "$TACMIND_ISAAC_SIM"
unzip data/downloads/isaac-sim-standalone-5.1.0-linux-x86_64.zip -d "$TACMIND_ISAAC_SIM"
(cd "$TACMIND_ISAAC_SIM" && ./post_install.sh)
```

Keep `TACMIND_ISAAC_SIM` set to that directory when installing Isaac Lab:

```bash
conda activate tacmind
mkdir -p third_party
git clone https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
git -C third_party/IsaacLab checkout b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8
ln -s "$TACMIND_ISAAC_SIM" third_party/IsaacLab/_isaac_sim
./third_party/IsaacLab/isaaclab.sh --install none
```

Restore Conda's Torch after the installer, then install shared dependencies:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r benchmark/config/simulation.txt
```

Source Isaac before building cuRobo to match its Torch/CUDA libraries:

```bash
git clone https://github.com/NVlabs/curobo.git third_party/curobo
git -C third_party/curobo checkout 78fd485fa82d9b9a063fb4985e371814587e666a
python -m pip install 'setuptools-scm>=6.2' wheel ninja
tacmind_root="$PWD"
source third_party/IsaacLab/_isaac_sim/setup_conda_env.sh
cd "$tacmind_root"
python -m pip install --no-build-isolation ./third_party/curobo
```

<a id="run-terminals"></a>
### Run terminals

Use separate terminals; open a fresh shell when switching from simulation to learning.
Isaac loads its bundled libraries; the PyTorch 2.10 installation above applies to the learning environment.

<a id="learning-terminal"></a>
#### Conversion and training

```bash
conda activate tacmind
```

<a id="simulation-terminal"></a>
#### Collection and evaluation

```bash
conda activate tacmind
tacmind_root="$PWD"
source third_party/IsaacLab/_isaac_sim/setup_conda_env.sh
cd "$tacmind_root"
export HEADLESS=1 LIVESTREAM=0
```

Set `CUDA_VISIBLE_DEVICES` to your allocated GPU; evaluation uses logical `cuda:0`.

<a id="assets"></a>
### Simulation assets

Download `tacmind-assets.tar.gz` from [Hugging Face](https://huggingface.co/datasets/JiaxinShi/TacMind-Bench)
or [ModelScope](https://modelscope.cn/datasets/shijiaxin/TacMind-Bench) at `assets/tacmind-assets.tar.gz`. Extract to
`data/downloads/tacmind-assets/`, with `assets/` directly inside. In the simulation terminal:

```bash
tacmind assets import --source data/downloads/tacmind-assets
```

Complete the asset import before starting scenes. Existing files are preserved;
move a file aside before importing its replacement.

<a id="data"></a>
## <img src="docs/media/icon-data.svg" width="28" height="28" alt=""> Get and convert data

Run from the repository root. Use the [simulation terminal](#simulation-terminal)
for collection/evaluation and the [learning terminal](#learning-terminal)
for conversion/training. This example uses Precision 0.

<a id="collection"></a>
### 1. Get data

Download and extract a task package from [Hugging Face](https://huggingface.co/datasets/JiaxinShi/TacMind-Bench)
or [ModelScope](https://modelscope.cn/datasets/shijiaxin/TacMind-Bench), keeping `collection.json`
and recordings in their relative locations. To collect 100 successful demonstrations yourself:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmark.expert.collection_queue \
  --task precision 0 --gpu 0 --target 100 \
  --output data/expert/gsmini/precision0_100
```

Outputs are HDF5 recordings, tactile previews and `summary.json`; convert when its
`status="complete"`. `--gpu` uses a physical index or UUID from `CUDA_VISIBLE_DEVICES`,
as shown by `nvidia-smi` in the current container. Use a new directory for each task
or changed collection setting. To resume, stop the collector and its workers, then
rerun the command. The default limit is 500 attempts; increase it with `--max-attempts`.
Read failure reasons in the summary and worker logs.

<a id="tasks"></a>

| Collection suite | Conversion suite | IDs |
| --- | --- | --- |
| `precision` | `tac_precision` | 0–5 |
| `recog` | `tac_recog` | 0–10 |
| `dynamic` | `tac_dynamic` | 0–10 |
| `extra` | `tac_extra` | 0–99 |

List task names and IDs with `tacmind catalog --suite precision`.

<a id="dataset-conversion"></a>
### 2. Convert data

```bash
mkdir -p tmp
cp examples/learning/convert_precision0.json tmp/convert_precision0.json
```

Set `collection_manifest` to the downloaded `collection.json` or your collection's `summary.json`:

```bash
tacmind learning convert --config tmp/convert_precision0.json
```

Output: `data/lerobot/gsmini_precision0`, with vision, tactile RGB, marker images and Deformation images.
The directory must be absent or empty. Config paths resolve from the repository root;
recording paths resolve from the manifest. Move the entire collection directory together.
For other data, edit these fields and match the training config's `dataset.root` / `dataset.repo_id`:

| Field | Value |
| --- | --- |
| `suite` / `task_id` | Conversion suite and task ID from the table above |
| `episodes` | Positive successful demonstration count, no greater than the available count; selected in manifest order |
| `dataset_root` / `repo_id` | New dataset directory and identifier |
| `camera_names` | The template keeps `agentview`, `wristview`, `gsmini_left`, `gsmini_right` |
| `include_markers` / `include_deformation` | The template enables both; TacMind-0 requires `include_deformation: true` |

<a id="train"></a>
## <img src="docs/media/icon-train.svg" width="28" height="28" alt=""> Train TacMind-0

Inputs must include `agentview` and `wristview` RGB, joint states, language instructions and
`observation.images.deformation_left/right`, the lossless FOTS displacement-arrow images.
Keep the generated `tactile_input.json` with each training bundle.

<a id="tacmind0-setup"></a>
### 1. Prepare the environment and pretrained weights

Complete [TacMind installation](#installation). Run from the repository root.
In a fresh terminal without Isaac, prepare the Python 3.12 training environment:

```bash
conda create -n tacmind0-python python=3.12 pip
conda run -n tacmind0-python python -m venv policy/TacMind-0/.venv
policy/TacMind-0/.venv/bin/python -m pip install -r policy/TacMind-0/requirements.txt
policy/TacMind-0/.venv/bin/python -m pip install --no-deps -e policy/TacMind-0
```

Download these pretrained weights, retaining their configuration and processor/tokenizer files:

| Resource | Source and local location | Use |
| --- | --- | --- |
| OpenDM 0.5 | [Dexmal/DM05](https://huggingface.co/Dexmal/DM05) → `data/weights/DM05/` | `pretrained_path` |
| DINOv2 Small | [facebook/dinov2-with-registers-small](https://huggingface.co/facebook/dinov2-with-registers-small) → `data/weights/dinov2-with-registers-small/` | Tac-LeWM `backbone_dir` |
| DINOv3 ViT-B/16 | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) → `data/weights/dinov3-vitb16/` | Tac-LeWM `vision_backbone_dir` |

<a id="taclewm-pretraining"></a>
### 2. Train Tac-LeWM

Use a GSmini multitask LeRobot dataset with vision, FOTS Deformation images and absolute joint actions.

If you already have Tac-LeWM weights pretrained on Deformation images and matching `resources/`, continue with step 3.

Install the pretraining environment and copy its configuration:

```bash
conda run -n tacmind0-python python -m venv policy/TacMind-0/.venv-pretraining
policy/TacMind-0/.venv-pretraining/bin/python -m pip install -r policy/TacMind-0/requirements-pretraining.txt
mkdir -p tmp
cp policy/configs/taclewm_pretrain.json tmp/pretrain.json
```

Use the following local paths in `tmp/pretrain.json`, replacing `dataset_path` with your multitask dataset. Keep the remaining recipe parameters from the template:

```json
{
  "runtime_python": "policy/TacMind-0/.venv-pretraining/bin/python",
  "dataset_path": "/path/to/gsmini_multitask_lerobot",
  "output_dir": "data/runs/taclewm_pretrain",
  "backbone_dir": "data/weights/dinov2-with-registers-small",
  "vision_backbone_dir": "data/weights/dinov3-vitb16",
  "devices": 3
}
```

`devices` must match the GPU count selected by `CUDA_VISIBLE_DEVICES`; this example uses 3 GPUs. The default trains for 20 epochs and writes to
`data/runs/taclewm_pretrain`; use a new `output_dir` for each run. In the TacMind learning terminal:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m policy.taclewm --config tmp/pretrain.json
```

Keep the generated `checkpoints/weights_epoch_N.pt`, its adjacent `tactile_input.json`, and the complete `resources/` directory together for TacMind-0 training.
With the default config, use `checkpoints/weights_epoch_20.pt`. This entry does not support optimizer resume.

<a id="tacmind0-policy-training"></a>
### 3. Train the target-task policy and Tac-FRS

The example uses the Precision 0 dataset converted above. Start in the TacMind learning
terminal (`conda activate tacmind`); the launcher starts the model's Python 3.12 process.

```bash
mkdir -p tmp
cp policy/configs/tacmind0.json tmp/tacmind0_precision0.json
```

The template uses these paths. Update them if your data or weights are elsewhere:

| Field | Precision 0 example |
| --- | --- |
| `dataset.root` / `dataset.repo_id` | `data/lerobot/gsmini_precision0` / `tacmind/gsmini_precision0` |
| `runtime_python` | `policy/TacMind-0/.venv/bin/python` |
| `pretrained_path` | `data/weights/DM05` |
| `model.world_weights` | `data/runs/taclewm_pretrain/checkpoints/weights_epoch_20.pt` |
| `model.resources` | `data/runs/taclewm_pretrain/resources` |
| `output_dir` | `data/runs/precision0_tacmind0` (new directory) |

```bash
CUDA_VISIBLE_DEVICES=0 tacmind learning train --config tmp/tacmind0_precision0.json
```

The default runs all three stages in order. Outputs are relative to `output_dir`:

| Stage | Default steps | Output |
| --- | ---: | --- |
| Adapt the pretrained Tac-LeWM to the target task | 10,000 | `tactile/checkpoint-10000/` |
| Train the VTLA action policy | 10,000 | `policy/`, with intermediate `policy/checkpoint-<step>/` |
| Train Tac-FRS | 10,000 | `frs/checkpoint-10000/` |

Read `stages.json` for the completed checkpoint paths. Use the policy or Tac-FRS checkpoint
for evaluation; tactile world-model weights alone are not an action policy.
This unified training entry uses one process and does not support optimizer resume.
The evaluation commands below use the final Tac-FRS checkpoint.

<a id="policy-evaluation"></a>
## <img src="docs/media/icon-evaluate.svg" width="28" height="28" alt=""> Evaluate policies

<a id="tacmind0-inference"></a>
### TacMind-0 inference environment

TensorRT + CUDA Graph inference uses a separate PyTorch 2.10 environment and requires
a CUDA 13.0-compatible driver. In a fresh terminal without Isaac:

```bash
conda run -n tacmind0-python python -m venv policy/TacMind-0/.venv-inference
policy/TacMind-0/.venv-inference/bin/python -m pip install -r policy/TacMind-0/requirements-tensorrt.txt
policy/TacMind-0/.venv-inference/bin/python -m pip install --no-deps -e policy/TacMind-0
```

### Nominal evaluation

Switch to the simulation terminal configured above; the evaluator starts the specified inference process.

```bash
CUDA_VISIBLE_DEVICES=0 tacmind learning evaluate \
  --checkpoint data/runs/precision0_tacmind0/frs/checkpoint-10000 \
  --policy-python "$PWD/policy/TacMind-0/.venv-inference/bin/python" \
  --episodes 20 --seed-start 10000 \
  --output-dir tmp/eval/precision0_tacmind0/nominal --headless --device cuda:0
```

For ACT/DP, omit `--policy-python`; for other separate model environments, set it to their Python executable. Change the checkpoint and output directory accordingly.

Checkpoint paths for other models are listed below.
Evaluation reads the checkpoint's task and settings; use `--policy-python` if its interpreter moved.
Choose seeds held out from training and normalization; compare models with the same data,
seeds and conditions. `--episodes` must be positive; all seeds must be in `[0, 2**32)`.
Task limits are fixed at the longest reference episode plus 100 steps, at 20 Hz.

Read `success_rate` in `results.json` when `score_complete=true`. Videos go to `videos/`
by default; disable with `--video-mode none`. The output directory must be absent or empty.
After interruption, rerun the same checkpoint and seeds in a new directory.

<a id="checkpoint-layouts"></a>
### Choose the checkpoint

| Policy | Checkpoint path |
| --- | --- |
| ACT / DP | `checkpoints/<step>/pretrained_model/` |
| OpenPI / N0-VTLA | `checkpoints/tacmind/run/<step>/` |
| OpenDM | `checkpoints/checkpoint-<step>/`, or final `checkpoints/` |
| ACT Tactile | `checkpoints/best/`, or `checkpoints/<six-digit-step>/` |
| TacMind-0 | Policy/FRS path in `stages.json`; the default final bundle is `frs/checkpoint-10000/` |

Keep weights, configuration and normalization files when copying checkpoints;
copy symlink targets. TacMind-0 uses the separate inference environment above.

<a id="robustness-evaluation"></a>
### Robustness evaluation

Run all 12 perturbations for the same TacMind-0 checkpoint in a separate output directory:

```bash
CUDA_VISIBLE_DEVICES=0 tacmind learning evaluate \
  --checkpoint data/runs/precision0_tacmind0/frs/checkpoint-10000 \
  --policy-python "$PWD/policy/TacMind-0/.venv-inference/bin/python" \
  --perturb --episodes 20 --seed-start 10000 \
  --output-dir tmp/eval/precision0_tacmind0/perturbations --headless --device cuda:0
```

`--perturb lighting`, `camera` or `scale` selects 4, 6 or 2 conditions.
Supported suites: Precision, Recog, Dynamic; Extra supports nominal evaluation only.
Each condition has a `results.json`; the root report is an index.

Aggregation requires all 13 complete conditions per task. **Avg InD** averages nominal
scores within Precision (6), Recog (11) and Dynamic (11), then averages the three categories.
**Avg OOD** averages conditions within each task's perturbation group, then tasks within
each category, then the nine category × group scores. Report percentages; missing conditions
leave the aggregate incomplete. Keep manifests, configs, source/environment versions,
complete checkpoints and evaluation commands with results.

<a id="policies"></a>
## <img src="docs/media/icon-policies.svg" width="28" height="28" alt=""> Other policies and configurations

<a id="supported-policies"></a>
### Supported policies

| Policy | Configuration |
| --- | --- |
| ACT | [act/config.json](policy/act/config.json) |
| Diffusion Policy | [diffusion/config.json](policy/diffusion/config.json) |
| OpenPI (π0.5) | [configs/openpi.json](policy/configs/openpi.json) |
| OpenDM (DM05) | [configs/opendm.json](policy/configs/opendm.json) |
| N0-VTLA | [configs/n0-vtla.json](policy/configs/n0-vtla.json) |
| ACT Tactile | [configs/act_tactile.json](policy/configs/act_tactile.json) |
| TacMind-0 | [configs/tacmind0.json](policy/configs/tacmind0.json) |

<a id="source-and-environment-setup"></a>
### Source and environment setup

ACT/DP use the [learning environment](#training-and-evaluation).
TacMind-0 uses the separate training and inference environments prepared above.
All policy sources are included under `policy/`. Use separate environments for
OpenPI, OpenDM and N0-VTLA. Follow the installation steps below in their existing
source directories; skip cloning the repositories again.

| Model environment | Installation reference |
| --- | --- |
| OpenPI | [README](policy/openpi/README.md), including PyTorch support and Transformers replacement |
| OpenDM | [README](policy/opendm/README.md) |
| N0-VTLA | [Installation guide](policy/n0-vtla/docs/INSTALL.md), including model weights, Transformers replacement and DINOv2 cache |
| ACT Tactile | Training environment with Torch, Torchvision, NumPy, Pillow and safetensors; cache the visual ResNet18 ImageNet weights or allow the initial download |

ALOHA CAD and LIBERO scene assets are optional; see [asset setup](policy/optional-assets.md)
for upstream hardware and simulator examples.

<a id="training-configuration"></a>
### Training configuration

In a copy of the selected model configuration, set these fields before running the training command at the end of this section:

| Field | Value |
| --- | --- |
| `dataset.root` / `dataset.repo_id` | Dataset directory and identifier |
| `dataset.episodes` | Optional list of training episode indices; omit to use all episodes |
| `output_dir` | New training output directory |
| `runtime_python` | Python executable in the model environment; omit for ACT/DP |
| `pretrained_path` | OpenPI/N0-VTLA PyTorch directory containing `model.safetensors`, or DM05 directory for OpenDM/TacMind-0; omit for ACT/DP and ACT Tactile |

Replace placeholder paths; relative paths resolve from the repository root.
ACT/DP normalization uses the whole dataset even when `dataset.episodes` selects a subset.

#### Other model settings

The unified training entry for OpenPI, OpenDM, N0-VTLA, ACT Tactile and TacMind-0 supports one process.
ACT Tactile and N0-VTLA also require `gsmini_left` and `gsmini_right` tactile RGB.
ACT Tactile uses an 80/20 training/validation split. Set `model.tactile_ckpt` to initialize its
tactile encoder; otherwise it trains from scratch.

#### Reuse prepared data

These models prepare episodes under `dataset.root/policy_exports/<policy_type>` by default.
Set `prepared_root` to choose another directory, or run `tacmind learning prepare --config tmp/policy.json`
to prepare separately. Reuse requires the same policy type, dataset, episode selection and prompt.
When these change, choose a new `prepared_root` and prepare again.
To reuse an export on another host, copy the entire prepared directory and set `prepared_root`;
omit `dataset` and `prompt` to use the saved settings without the original dataset.

<a id="baseline-training"></a>
### Start training and resume

Copy a model configuration from the table above, using DP here:

```bash
mkdir -p tmp
cp policy/diffusion/config.json tmp/policy.json
```

Set the dataset and output paths using the fields above, then run:

```bash
tacmind learning train --config tmp/policy.json
```

The default DP configuration trains for 50,000 steps and saves to `data/runs/gsmini_precision0_dp`.

<a id="training-and-checkpoint-resume"></a>
To resume ACT/DP, keep the complete checkpoint, including `pretrained_model/` and `training_state/`:

```bash
tacmind learning train \
  --config data/runs/gsmini_precision0_dp/checkpoints/last/pretrained_model/train_config.json \
  --resume
```

<a id="advanced"></a>
## <img src="docs/media/icon-advanced.svg" width="28" height="28" alt=""> Advanced usage

Use the training or inference environment prepared above.
Run all commands from the TacMind repository root.

<a id="individual-stages"></a>
### Individual training stages

Activate the training environment, then use the stage entry below with `--help`:

```bash
source policy/TacMind-0/.venv/bin/activate
```

| Stage | Native entry |
| --- | --- |
| Tac-LeWM fine-tuning | `python -m tacmind0.tactile.finetune --help` |
| TacMind-0 policy training | `python -m tacmind0 --task train --help` |
| Tac-FRS | `python -m tacmind0.tac_frs.training --help` |

For fine-tuning, pass pretrained weights with `--weights` and matching configurations with `--resources`.
Policy training needs the base policy directory, fine-tuned `encoder.pt` and `backbone_config.json`,
episode JSONL files and images. Tac-FRS requires the complete fine-tuned world model, not just
its encoder.

### Policy service

Activate the inference environment and provide a trained policy or Tac-FRS bundle:

```bash
source policy/TacMind-0/.venv-inference/bin/activate
python -m tacmind0 --task inference \
  --model-config.model-name-or-path /path/to/policy_or_frs_bundle
```

The default port is `7897`; change it with `--inference-config.port`.

```python
from tacmind0.policy.client import TacMind0Client

client = TacMind0Client("http://127.0.0.1:7897")
client.reset(lane=0)
# Supply both Deformation images every simulation step, with consecutive frame indices starting at 0.
client.observe(left_deformation, right_deformation, frame_index=0, lane=0)
actions = client.infer(agentview, wrist, state, prompt, lane=0)
```

Images are RGB `uint8` arrays; `state` contains 8 joint/gripper values. Returned physical actions
have shape `(H, 8)`. Call `observe()` on every step, including while executing an existing action chunk.
Use separate lanes for parallel environments. For Tac-FRS, use `tacmind0.tac_frs.client.TacFRSClient`,
call `reset()` at each episode start and `close(lane=0)` when finished using the lane.

<a id="multitask"></a>
### Multitask policy training

Use a multitask dataset in the [pretraining format](#taclewm-pretraining), pretrained Tac-LeWM
and OpenDM 0.5 weights. Create a config with local paths:

```json
{
  "dataset": "/path/to/lerobot",
  "world_weights": "/path/to/taclewm/checkpoints/weights_epoch_20.pt",
  "world_resources": "/path/to/taclewm/resources",
  "base": "/path/to/dm05",
  "output": "/path/to/new-output"
}
```

```bash
source policy/TacMind-0/.venv/bin/activate
python -m policy.tacmind0.train_multitask_lerobot --config /path/to/direct.json prepare
torchrun --standalone --nproc_per_node=4 -m policy.tacmind0.train_multitask_lerobot \
  --config /path/to/direct.json train
```

The default uses global batch 32 and 100,000 steps, saving to `output/policy/checkpoint-N`
every 10,000 steps. Supported process counts are 1, 2, 4, 8, 16 and 32.
`process count × per_device_batch_size` must divide 32 exactly.
After interruption, rerun `train`
with the same config without repeating `prepare`. These checkpoints cannot be used
directly with unified evaluation; use [staged training](#tacmind0-policy-training) for single-task evaluation.
