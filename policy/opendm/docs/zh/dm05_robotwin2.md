# DM05 RoboTwin 2.0 训练与评测指南

使用 `DM05` 在 RoboTwin 2.0 上训练、部署推理服务并评测。

## 参考指标

| 方法 | Clean | Randomized | Average |
| --- | ---: | ---: | ---: |
| DM0.5 | 93.6 | 93.3 | 93.5 |

## RoboTwin 2.0 训练

### 前提条件

- 按官方步骤安装并初始化 OpenDM。
- 使用 NVIDIA GPU，如 A100、H100、H20 或 RTX 4090。在 RTX 4090 等显存较小的
  GPU 上，Full SFT 可能在 FSDP 初始化时显存不足，此时建议使用 LoRA 微调。
- 若没有 `hf` 命令，先安装 Hugging Face CLI：

```bash
pip install -U huggingface_hub
```

### 数据准备

从 Hugging Face 下载数据集和基础模型：

- RoboTwin 2.0 完整数据集：[Dexmal/robotwin2-full](https://huggingface.co/datasets/Dexmal/robotwin2-full)
- DM05 模型：[Dexmal/DM05](https://huggingface.co/Dexmal/DM05)

在 OpenDM 仓库根目录准备数据和模型：

```bash
cd opendm

# 下载 RoboTwin 2.0 数据集压缩包的全部分卷。
mkdir -p data/.hf_downloads/robotwin
hf download Dexmal/robotwin2-full \
  --repo-type dataset \
  --local-dir data/.hf_downloads/robotwin

# 合并分卷并解压到 ./data/robotwin2.0。
cat data/.hf_downloads/robotwin/robotwin2.tar.part-* \
  | tar -xf - -C data

# 下载 DM05 基础模型。
hf download Dexmal/DM05 --local-dir checkpoints/DM05
```

解压后的目录结构（`<setting>` 为 `clean` 或 `randomized`）：

```text
data/robotwin2.0/
├── jsonl/<task>/<setting>/
└── video/<task>/<setting>/
```

`opendm/dataset/robotwin2.py` 将该数据包注册为 `robotwin2_generalist`，根目录为
`./data/robotwin2.0`。默认配方使用 100 个名为
`robotwin2_component_<task>_<setting>` 的加权视图，覆盖全部 50 个任务的两种设置。
各视图读取 `jsonl/<task>/<setting>/`，共用 `video/` 根目录。原有的
`robotwin2_example_<task>_<setting>` 配置仍作为兼容别名受到支持。

ALOHA RoboTwin2 配方使用：

- 头部、左腕、右腕 RGB 图像，依次对应 `images_1`、`images_2`、`images_3`。
- 未来帧 `state` 构造的绝对关节位置目标，忽略原始 `action`。
- 原生 14 维 state/action 校验和归一化，不做分位数裁剪；仅将归一化后的 action
  和 mask 补齐至 32 维。
- task/subtask prompt 采样，以及先选数据集、再选样本的加权采样。

归一化参数缺失时，训练启动后会自动计算并保存到
`<norm-stats-root>/norm_stats.json`（默认为
`./norm_stats/robotwin2_absolute_action/14d/norm_stats.json`）。可用
`--data-config.norm-stats-root` 覆盖目录，也可在训练前计算：

```bash
python playground/dm05_robotwin2.py --task compute_norm_stats
```

使用与训练相同的数据集选择和路径。归一化参数会复制到各 checkpoint 的
`norm_stats.json`，推理优先读取该文件，请与权重一起保留。

### 启动训练

RoboTwin 2.0 参考训练配置必须使用 4 机训练，每机 8 张 GPU。在每台机器的
OpenDM 仓库根目录运行以下命令，将 `<NODE_RANK>` 分别设为 `0`、`1`、`2`、`3`：

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

将 `<MASTER_ADDR>` 设为 rank 0 机器的地址，并确保所有节点均可访问其 `29500` 端口。

global batch size 必须为 `1024`，且 `gradient_accumulation_steps=1`：

```text
global batch size = 机器数 × 单机 GPU 数 × per-device train batch size × gradient accumulation steps
                  = 4 × 8 × 32 × 1
                  = 1024
```

该配置不能改成单机训练。调整 GPU 数量或单卡 batch size 时，仍须使用 4 台机器，
保持 global batch size 为 `1024`、`gradient_accumulation_steps=1`。

如需将整个数据包作为单一数据集采样，关闭默认加权视图，可同时传入
`--data-config.dataset-name robotwin2_generalist` 和
`--data-config.dataset-weights 1.0`。

默认配置：输出目录 `user_checkpoints/dm05_robotwin2`、绝对动作、chunk size 50、
最大 token 长度 1152、FSDP1、BF16 和 Muon AdamW。可通过对应命令行选项覆盖训练参数，
但须满足上述 batch size 约束。训练、推理和评测的 chunk size 必须一致，
推理使用 checkpoint 中的归一化参数。

## RoboTwin 2.0 推理

发布版 RoboTwin 2.0 checkpoint、完整服务命令、fast backend 配置和 HTTP API 使用方法统一参考 [DM05 推理指南](dm05_inference.md)。

## RoboTwin 2.0 评测

### 准备 dexbotic-benchmark

条件允许时，推理和仿真使用不同 GPU。准备官方 benchmark Docker 镜像：

```bash
git clone https://github.com/dexmal/dexbotic-benchmark.git
cd dexbotic-benchmark
git submodule update --init --recursive RoboTwin
docker pull dexmal/dexbotic_benchmark
```

评测前，按照 [RoboTwin 安装文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html#4-download-assets-robotwin-od-texture-library-and-embodiments)
下载 assets、object-data texture library 和 embodiment 文件。

### 配置评测任务

使用 `evaluation/configs/robotwin2/adjust_bottle.yaml` 作为参考配置，并将
`base_url` 修改为推理服务地址：

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

重要配置项：

- `task_name`：50 个任务之一，独立评测。
- `task_config`：`demo_clean` 或 `demo_randomized`。
- `base_url`：DM05 服务地址；远程服务填写 `http://<SERVER_IP>:7891`。
- `cameras`：保持头部、左腕、右腕顺序，对应 `images_1/2/3`。
- `action_horizon`：每次请求返回并执行的动作数，须与 chunk size 50 一致。
- `action_mode`：当前 checkpoint 使用 `absolute`（绝对关节位置）。
- `output_dir`：结果和 rollout 视频目录。

每个任务、每种设置评测 100 个 episode。复现 Clean 和 Randomized 汇总指标时，
在两种设置下评测全部 50 个任务并汇总成功率。

### 使用 Docker 启动评测

在 `dexbotic-benchmark` 仓库根目录，将 `ROBOTWIN_ASSETS` 设置为已下载的
RoboTwin assets 绝对路径，然后运行：

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

### 使用本地环境启动评测

调试仿真环境时，参考 `dexbotic-benchmark/docs/local_install.md` 安装环境后运行：

```bash
cd dexbotic-benchmark
conda activate RoboTwin

# 推荐使用官方 shell 脚本。
bash scripts/env_sh/robotwin2.sh \
  evaluation/configs/robotwin2/adjust_bottle.yaml

# 或直接运行 Python，并按需覆盖配置。
python evaluation/run_robotwin2_evaluation.py \
  --config evaluation/configs/robotwin2/adjust_bottle.yaml \
  --set base_url http://localhost:7891 \
  --set output_dir ./result_test/robotwin2_evaluation
```

### 确认评测结果

评测结果会按照以下层级保存在 `output_dir` 下：

```text
<output_dir>/<task_name>/<task_config>/<timestamp>/
├── _result.txt
└── *.mp4
```

`_result.txt` 记录该任务和设置的成功率。RoboTwin 任务配置启用视频记录时，
rollout 视频保存在同一目录。
