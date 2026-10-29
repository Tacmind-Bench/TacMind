# DM05 VLA-Arena 训练与评测指南

本文档介绍如何使用 `DM05` 完成 VLA-Arena 场景下的模型训练、推理服务启动以及 benchmark 评测流程。

## VLA-Arena 训练

### 前提条件

开始训练前，请确认已经完成以下准备：

- **已按照官方步骤完成 OpenDM 环境安装和源码初始化。**
- 训练和推理需要使用 GPU 资源，推荐使用 A100, H100, H20, 4090 等 GPU 卡。在 RTX 4090 等显存较小的 GPU 上执行 Full SFT 时，FSDP 初始化阶段的显存峰值可能导致训练失败，建议采用 LoRA 微调。

### 数据准备

VLA-Arena 数据和基础模型可从 Hugging Face 下载：

- VLA-Arena 数据集：[Dexmal/vla_arena_L0_L](https://huggingface.co/datasets/Dexmal/vla_arena_L0_L)
- DM05 模型：[Dexmal/DM05](https://huggingface.co/Dexmal/DM05)

在 OpenDM 工程根目录下准备数据和模型：

```bash
# 在 OpenDM 仓库根目录运行。
cd opendm

# 下载 VLA-Arena 数据集
huggingface-cli download Dexmal/vla_arena_L0_L --repo-type dataset \
  --local-dir ./data/vla_arena_L0_L

# 解压分片 tar 包
cd ./data/vla_arena_L0_L
cat vla_arena_L0_L.tar.gz.part-* | tar -xzf -
cd -

# 下载 DM05 基础模型
huggingface-cli download Dexmal/DM05 --local-dir ./checkpoints/DM05
```

确认 VLA-Arena 数据目录已准备完成。数据集结构如下：

```
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

准备数据集注册文件，并确认归一化参数生成方式：

```
# VLA-Arena 数据集注册文件
opendm/dataset/vla_arena.py
```

说明：

- `opendm/dataset/vla_arena.py` 用于注册 VLA-Arena 数据集，默认注册的数据集名称为 `vla_arena_eef_L0_L`。
- 训练启动时，如果对应的归一化参数文件不存在，脚本会根据当前数据集、action mode 和 action chunk 长度自动计算并保存到 `./norm_stats/`。
- checkpoint 保存时会同时把训练使用的归一化参数复制为 checkpoint 目录下的 `norm_stats.json`。推理会优先读取 checkpoint 目录下的 `norm_stats.json`；如果不存在，则根据当前数据集、action mode 和 action chunk 长度到 `./norm_stats/` 查找对应文件。
- 训练命令中的 `--data-config.dataset-name` 需要与 `opendm/dataset/vla_arena.py` 中注册的数据集名称保持一致。

### 启动训练

确认数据、模型和注册文件都准备完成后，启动 VLA-Arena 训练：

```bash
# 在 OpenDM 仓库根目录运行。
cd opendm

script/dm05_launcher.sh \
  --exp ./playground/dm05_vla_arena.py \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name vla_arena_eef_L0_L \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --trainer-config.num-train-steps 60000
```

参数说明：

- `--exp ./playground/dm05_vla_arena.py`：使用 VLA-Arena 场景的训练入口，该入口预设了 VLA-Arena 需要的 action mode 和默认 chunk size。
- `--task train`：指定当前任务为训练模式。
- `--nproc_per_node 8`：单节点使用的 GPU 数量，推荐 8 卡。
- `--data-config.dataset-name vla_arena_eef_L0_L`：指定训练使用的 VLA-Arena 数据集名称。
- `--model-config.model-name-or-path ./checkpoints/DM05`：指定基础模型 checkpoint 路径。
- `--trainer-config.num-train-steps 60000`：总训练步数。

## VLA-Arena 推理

**完成环境安装和源码初始化后**，可以利用上一步训练好的 DM05 VLA-Arena 模型 checkpoint 启动模型推理服务。

### 前提条件

推理前，请确认已经**按照官方步骤完成 OpenDM 环境安装和源码初始化。** 推理需要至少 1 张 GPU。

### 启动推理

- 推理会优先读取模型 checkpoint 目录下的 `norm_stats.json`；请确认该 checkpoint 来自与当前推理配置一致的 VLA-Arena 训练流程。

```bash
# 启动推理服务
script/dm05_launcher.sh \
  --exp playground/dm05_vla_arena.py \
  --task inference \
  --nproc_per_node 1 \
  --model-config.model-name-or-path ./checkpoints/DM05-vla-arena-checkpoint \
  --inference-config.output-action-dim 7
```

参数说明：

- `--exp playground/dm05_vla_arena.py`：使用 VLA-Arena 场景的推理入口，与训练阶段保持一致。
- `--task inference`：任务类型，推理时使用 inference。
- `--nproc_per_node 1`：单节点使用的 GPU 数量，推理使用 1 卡就可以。
- `--model-config.model-name-or-path ./checkpoints/DM05-vla-arena-checkpoint`：模型 checkpoint 路径。
- `--inference-config.output-action-dim 7`：VLA-Arena 动作输出维度。

推理服务启动后需要保持运行，评测脚本会通过 HTTP 接口请求该服务完成动作预测。

## VLA-Arena 评测

### 准备阶段

1. 建议使用至少 2 张 GPU 完成评测流程：一张 GPU 运行 OpenDM 推理服务，另一张 GPU 运行 VLA-Arena 仿真和 EGL 渲染。
2. 按照上述步骤启动 DM05 推理服务并保持运行。dexbotic-benchmark 会通过 HTTP 请求该服务，配置中使用 `base_url`，例如 `http://<SERVER_IP>:7891`，不要附加 `/process_frame`。
3. 配置 benchmark 环境。VLA-Arena 评测使用 `dexbotic-benchmark` 中的 Arena runner，不再需要把 OpenDM 的评测文件复制到 VLA-Arena 官方仓库。

```bash
# 获取评测脚本和配置文件。
git clone https://github.com/dexmal/dexbotic-benchmark.git
cd dexbotic-benchmark

# 只评测 VLA-Arena 时，仅初始化 Arena 子模块即可。
git submodule update --init --recursive arena
```

4. 创建本地 VLA-Arena 评测环境。Arena 评测当前推荐使用本地 Conda 环境运行；如需更完整的系统依赖说明，可参考 `dexbotic-benchmark/docs/local_install.md` 中的 VLA-Arena Environment 部分。

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

### 评测阶段

#### 修改评测配置

使用并修改 VLA-Arena 示例配置 `evaluation/configs/arena/example_arena.yaml`：

```yaml
# 任务选择。all 会评测非 LIBERO 的 11 个 Arena suite。
task_suite_name: all
task_level: all
parallelized: false
num_trials_per_task: 10
num_steps_wait: 10
seeds: [7, 42, 1000]

# 仿真环境与鲁棒性开关。
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

# OpenDM 推理服务。
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

# legacy multipart 请求参数。
batch_size: 1
speed: "0.5"
action_horizon: 20
request_timeout: 30

# 指令鲁棒性。
replacements_file: VLA-Arena/language_replacements
use_replacements: false
replacement_probability: 1.0
replacement_level: 1

# 输出。
save_video_mode: first_success_failure
video_fps: 30
result_json_path: default
output_dir: experiments/eval_results
log_level: INFO
```

可修改的配置内容包括：

- `base_url`：已启动的 DM05 推理服务地址，通常为 `http://<SERVER_IP>:7891`。dexbotic-benchmark 会根据 `api_style: legacy` 自动请求 `/process_frame`。
- `api_style`：OpenDM VLA-Arena 服务使用 legacy multipart 接口时保持 `legacy`。
- `task_suite_name`：评测任务集。设为 `all` 评测全部 Arena suite，也可以指定具体 suite 名称，如 `safety_static_obstacles`、`extrapolation_unseen_objects`。
- `task_level`：评测难度级别，可选 `0`、`1`、`2` 或 `all`。
- `parallelized`：设为 `true` 时会为 L0、L1、L2 各启动一个评测进程；此时 `task_level` 必须为 `all`，并且推理服务需要能承载 3 路并发请求。
- `seeds`：随机种子列表。命令行覆盖时使用逗号分隔，例如 `--set seeds 7,42,1000`。
- `num_trials_per_task`：每个任务评测的 episode 数量。
- `replan_step`：每次调用模型后执行的动作步数，默认执行 10 步后重新请求模型。
- `action_horizon`：服务每次返回的动作块长度，需要与推理服务 `--model-config.chunk-size` 保持一致。
- `use_replacements`：是否启用 VLA-Arena 的 instruction replacement 鲁棒性测试；默认 `false`，与参考配置一致。
- `save_video_mode`：视频保存模式，可选 `all`、`first_success_failure`、`none`。
- `output_dir`：评测结果保存目录。

#### 启动 VLA-Arena 评测

在 `dexbotic-benchmark` 仓库根目录下执行：

```bash
cd dexbotic-benchmark
conda activate arena

python3 evaluation/run_arena_evaluation.py \
  --config evaluation/configs/arena/example_arena.yaml \
  --set base_url http://<SERVER_IP>:7891 \
  --set output_dir experiments/eval_results
```

如需并发评测 L0、L1、L2，可以开启 parallelized：

```bash
python3 evaluation/run_arena_evaluation.py \
  --config evaluation/configs/arena/example_arena.yaml \
  --set task_level all \
  --set parallelized true \
  --set base_url http://<SERVER_IP>:7891 \
  --set output_dir experiments/eval_results_parallel
```

并发模式会共享同一个推理服务。如果服务端使用全局随机采样、默认 session 记忆或无法承载并发请求，结果可能与串行评测不完全一致；需要严格复现轨迹时使用默认串行模式。

#### 确认评测结果

评测结果保存在 `output_dir` 下，主要文件：

- `results.json`：当前 run 的详细结果和汇总结果。
- `leaderboard.json`：VLA-Arena leaderboard 格式结果。
- `config.yaml`：实际生效的评测配置。
- `logs/evaluation.log`：评测日志。
- `videos/`：按 `save_video_mode` 保存的 rollout 视频。
- `seed_<N>/`：多 seed 评测时，每个 seed 的独立结果目录。
- `batch_summary_<timestamp>.json`：多 seed 评测时生成的 SR/CC 聚合统计。
- `parallel_levels/`：并发评测时，每个 level worker 的中间输出。

使用默认 `seeds` 配置时，各 seed 的 `leaderboard.json` 和 `videos/` 位于
`seed_<N>/` 下；顶层 `results.json` 汇总各 seed 的结果。

评测结束后，优先查看 `results.json` 和 `batch_summary_<timestamp>.json`。Safety suite 的 CC 对应 cumulative cost。
