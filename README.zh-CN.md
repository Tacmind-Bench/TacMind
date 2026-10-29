<h1 align="center"><img src="docs/media/tacmind.svg" width="42" height="42" alt=""> TacMind</h1>

<p align="center"><b>面向机器人操作的触觉交互状态评测与建模</b></p>

<p align="center">
  Jiaxin Shi<sup>1,4</sup> · Zibin Wu<sup>2,4</sup> · Yichi Zhang<sup>3,4</sup> · Jianghua Duan<sup>4</sup> · Siyu Zhu<sup>5</sup> · Weihao Yuan<sup>4,6</sup>
</p>

<p align="center">
  <sup>1</sup> 上海交通大学 · <sup>2</sup> 香港科技大学<br>
  <sup>3</sup> 哈尔滨工业大学 · <sup>4</sup> Daimon Robotics<br>
  <sup>5</sup> 复旦大学 · <sup>6</sup> 南京大学
</p>

<p align="center">
  <a href="https://tacmind-bench.github.io/"><img src="docs/media/link-website.svg" height="40" alt="项目主页"></a>
  <a href="https://tacmind-bench.github.io/#leaderboard"><img src="docs/media/link-leaderboard.svg" height="40" alt="排行榜"></a>
  <a href="https://huggingface.co/datasets/JiaxinShi/TacMind-Bench"><img src="docs/media/link-huggingface.svg" height="40" alt="Hugging Face 数据集"></a>
  <a href="https://modelscope.cn/datasets/shijiaxin/TacMind-Bench"><img src="docs/media/link-modelscope.svg" height="40" alt="ModelScope 数据集"></a>
</p>

<p align="center"><a href="README.md">English</a> · 简体中文</p>

<p align="center">
  <a href="#benchmark">Benchmark</a> · <a href="#tacmind-0">TacMind-0</a> · <a href="#results">评测结果</a> · <a href="#usage-guide">使用指南</a>
</p>

https://github.com/user-attachments/assets/8b76b294-3eef-4678-963d-1d717e91d473

<p align="center"><b>擦白板任务。</b>同一回合中同步记录两路相机和双指触觉。<br>触觉以 RGB、标记点运动和形变展示。<a href="https://tacmind-bench.github.io/#demo">观看演示 ↗</a></p>

<a id="benchmark"></a>
## <img src="docs/media/icon-benchmark.svg" width="28" height="28" alt=""> TacMind-Bench

**TacMind-Bench 通过 128 个机器人操作任务研究触觉交互。**
已发布的仿真流程基于 Isaac Sim 和 Isaac Lab，使用搭载双 GelSight Mini 传感器的 Franka 机械臂，
同步记录外部与腕部 RGB、触觉 RGB、标记点运动、机器人状态和语言指令。

仓库提供专家示范采集、LeRobot 数据转换、七类策略的训练和统一的任务成功率评估。
**TacMind-0** 是我们用于建模触觉交互状态，并利用这些状态引导操作的策略。

### 任务集合

| Suite | 任务数 | 能力与示例 |
| --- | ---: | --- |
| **Precision** | 6 | 精细接触控制：管、钥匙、连接器、齿轮、电源插头和灯泡插入 |
| **Recog** | 11 | 通过触觉消除视觉歧义：辨别尺寸、形状与表面纹理，包含遮挡场景 |
| **Dynamic** | 11 | 操作运动中的物体或目标，例如向传送带上的插座完成插入 |
| **Extra** | 100 | 更丰富的接触行为：拉开抽屉、旋转阀门、切割物体和擦拭表面 |

**Standard = Precision + Recog + Dynamic，共 28 个任务。** 100 个 Extra 任务扩展了物体和交互技能的范围。
[交互式任务图库](https://tacmind-bench.github.io/#tasks)提供同步传感器视角，Standard 任务还展示了 Daimon 传感器的演示。

### 评估条件

Standard 任务支持普通条件评估，以及在评估时施加的 **12 种受控扰动**。
Extra 任务支持普通条件评估。

| 扰动组 | 条件数 | 变化内容 |
| --- | ---: | --- |
| **光照** | 4 | 暖光、冷光、荧光灯与夜间光照 |
| **相机** | 6 | 左右视角、高低俯仰角，以及两个更远的视角 |
| **物体尺寸** | 2 | 缩放至 0.85 倍与 0.90 倍 |

模型和条件之间使用相同的留出随机种子。结果报告成功率，以及按套件均衡汇总的
**InD**（普通条件）和 **OOD**（扰动条件）平均值；完整计分方法与命令见下方[扰动评估](#robustness-evaluation)。

<a id="tacmind-0"></a>
## <img src="docs/media/icon-model.svg" width="28" height="28" alt=""> TacMind-0

TacMind-0 将触觉交互状态建模与视觉、语言及机器人状态结合，用于生成动作。
其触觉输入为**左右手指的 Deformation 图像历史**，以箭头表示标记点位移。

- **Tac-LeWM** 学习触觉交互表征，并以动作为条件预测未来触觉特征；世界模型训练还使用视觉上下文。
- **VTLA 策略** 将触觉表征与外部／腕部图像、语言指令、机器人状态结合，生成动作序列。
- **Tac-FRS** 预测候选动作下的未来触觉特征，再用这些预测调制后续动作生成。

训练首先在多任务示范上预训练 Tac-LeWM，随后按顺序进行目标任务的触觉模型适配、VTLA 策略训练和 Tac-FRS 训练。
下方[训练指南](#train)给出了这一完整流程所需的环境、预训练依赖、命令与 checkpoint 路径。

<a id="results"></a>
## <img src="docs/media/icon-results.svg" width="28" height="28" alt=""> 评测结果

**GelSight Mini Standard 任务**上的成功率（%），越高越好；粗体标出每列最优值。
三个套件列展示普通条件下的表现。

| 模型 | Precision InD | Recog InD | Dynamic InD | Avg InD | Avg OOD |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACT+Tactile† | 30.0 | **69.1** | **15.9** | **39.8** | 16.4 |
| DP† | 23.3 | 27.7 | 13.2 | 21.1 | 8.1 |
| OpenPI0.5 | 5.0 | 30.0 | 0.5 | 13.0 | 5.5 |
| OpenDM0.5 | 20.0 | 38.6 | 8.2 | 22.7 | 11.4 |
| N0-VTLA | 25.8 | 45.9 | 5.9 | 25.9 | 7.8 |
| **TacMind-0** | **40.0** | 50.0 | 10.5 | 32.3 | **20.7** |

† **单任务训练：** ACT+Tactile 和 DP 为每个任务分别训练一个策略。
[完整排行榜](https://tacmind-bench.github.io/#leaderboard)还列出了各套件的光照、相机和物体尺寸扰动成绩。

<a id="usage-guide"></a>

## 使用指南

[安装](#installation) · [数据](#data) · [TacMind-0 训练](#train) · [评估](#policy-evaluation) · [其他模型与配置](#policies) · [高级用法](#advanced)

下面以 Precision 0 演示完整流程：安装 → 准备数据 → 训练 → 普通与扰动评估。所有命令从仓库根目录运行。

<a id="installation"></a>
## <img src="docs/media/icon-install.svg" width="28" height="28" alt=""> 安装环境与资产

需要 Linux x86_64、Conda 和 Git LFS；采集和评估还需兼容 Isaac 的 GPU。
所有命令从仓库根目录运行。
环境版本：Ubuntu 22.04、CUDA 12.8、Isaac Lab 2.3.2 和 Isaac Sim 5.1.0。

### 基础环境

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
### 训练依赖

用于数据转换、ACT/DP 训练和策略评估；其他模型还需[专用环境](#source-and-environment-setup)。
安装系统 `ffmpeg`，然后在新终端运行：

```bash
conda activate tacmind
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r benchmark/config/learning.txt
python -m pip install --no-deps lerobot==0.4.4
```

<a id="simulation-environment"></a>
### 仿真依赖

安装 NVIDIA 驱动、含 `nvcc` 的 CUDA 12.8 工具包，
以及系统工具 `cmake`、`build-essential`、`ffmpeg`、`rsync`、`git-lfs`、`unzip`。
从[官方下载页](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)下载
**Isaac Sim 5.1.0 Linux x86_64 standalone ZIP**，保存为
`data/downloads/isaac-sim-standalone-5.1.0-linux-x86_64.zip`。
解压到新目录并运行安装脚本：

```bash
export TACMIND_ISAAC_SIM="$HOME/isaacsim-5.1.0"
mkdir -p "$TACMIND_ISAAC_SIM"
unzip data/downloads/isaac-sim-standalone-5.1.0-linux-x86_64.zip -d "$TACMIND_ISAAC_SIM"
(cd "$TACMIND_ISAAC_SIM" && ./post_install.sh)
```

安装 Isaac Lab 时保持 `TACMIND_ISAAC_SIM` 指向该目录：

```bash
conda activate tacmind
mkdir -p third_party
git clone https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
git -C third_party/IsaacLab checkout b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8
ln -s "$TACMIND_ISAAC_SIM" third_party/IsaacLab/_isaac_sim
./third_party/IsaacLab/isaaclab.sh --install none
```

安装器运行后恢复 Conda 的 Torch 版本，再安装共用依赖：

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r benchmark/config/simulation.txt
```

先加载 Isaac，再编译 cuRobo，使其 Torch/CUDA 依赖匹配：

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
### 运行终端

训练和仿真使用独立终端；从仿真切回训练时另开新 shell。
Isaac 加载自带库；上方安装的 PyTorch 2.10 用于训练环境。

<a id="learning-terminal"></a>
#### 转换与训练

```bash
conda activate tacmind
```

<a id="simulation-terminal"></a>
#### 采集与评估

```bash
conda activate tacmind
tacmind_root="$PWD"
source third_party/IsaacLab/_isaac_sim/setup_conda_env.sh
cd "$tacmind_root"
export HEADLESS=1 LIVESTREAM=0
```

通过 `CUDA_VISIBLE_DEVICES` 选择分配的 GPU，评估使用逻辑设备 `cuda:0`。

<a id="assets"></a>
### 仿真资产

从 [Hugging Face](https://huggingface.co/datasets/JiaxinShi/TacMind-Bench) 或
[ModelScope](https://modelscope.cn/datasets/shijiaxin/TacMind-Bench) 下载 `assets/tacmind-assets.tar.gz`，
解压至 `data/downloads/tacmind-assets/`，确保该目录直接包含 `assets/`。在仿真终端运行：

```bash
tacmind assets import --source data/downloads/tacmind-assets
```

启动场景前先完成资产导入。已有文件会保留；需要替换时先移开再导入。

<a id="data"></a>
## <img src="docs/media/icon-data.svg" width="28" height="28" alt=""> 获取和转换数据

从仓库根目录运行。采集和评估使用[仿真终端](#simulation-terminal)，
转换和训练使用[训练终端](#learning-terminal)。以下以 Precision 0 为例。

<a id="collection"></a>
### 1. 获取数据

从 [Hugging Face](https://huggingface.co/datasets/JiaxinShi/TacMind-Bench) 或
[ModelScope](https://modelscope.cn/datasets/shijiaxin/TacMind-Bench) 下载任务包并解压，
保持 `collection.json` 与录制文件的相对位置。自行采集 100 条成功示范则运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmark.expert.collection_queue \
  --task precision 0 --gpu 0 --target 100 \
  --output data/expert/gsmini/precision0_100
```

产物为 HDF5 录制、触觉预览和 `summary.json`；清单显示 `status="complete"` 后用于转换。
`--gpu` 使用 `CUDA_VISIBLE_DEVICES` 中的物理编号或 UUID，以当前容器的 `nvidia-smi` 为准。
更换任务或采集配置时使用新目录；续采先停止采集进程及 worker，再运行同一命令。
默认最多尝试 500 次，可用 `--max-attempts` 增加；失败原因见清单和 worker 日志。

<a id="tasks"></a>

| 采集 suite | 转换 suite | ID |
| --- | --- | --- |
| `precision` | `tac_precision` | 0–5 |
| `recog` | `tac_recog` | 0–10 |
| `dynamic` | `tac_dynamic` | 0–10 |
| `extra` | `tac_extra` | 0–99 |

用 `tacmind catalog --suite precision` 查看任务名与 ID。

<a id="dataset-conversion"></a>
<a id="数据转换"></a>
### 2. 转换数据

```bash
mkdir -p tmp
cp examples/learning/convert_precision0.json tmp/convert_precision0.json
```

将 `collection_manifest` 改为下载的 `collection.json` 或自行采集的 `summary.json`：

```bash
tacmind learning convert --config tmp/convert_precision0.json
```

输出到 `data/lerobot/gsmini_precision0`，包含视觉、触觉 RGB、marker 图像与 Deformation 图像；目录须不存在或为空。
配置路径相对于仓库根目录，录制路径相对于清单目录。
迁移时复制整个采集目录。
更换数据时修改以下字段，并同步训练配置中的 `dataset.root` / `dataset.repo_id`：

| 字段 | 内容 |
| --- | --- |
| `suite` / `task_id` | 上表的转换 suite 与任务 ID |
| `episodes` | 按清单顺序选取的成功示范数量，须为正整数且不超过可用数量 |
| `dataset_root` / `repo_id` | 新数据集目录与标识 |
| `camera_names` | 模板保留 `agentview`、`wristview`、`gsmini_left`、`gsmini_right` |
| `include_markers` / `include_deformation` | 模板同时启用两项；TacMind-0 必须保留 `include_deformation: true` |

<a id="train"></a>
## <img src="docs/media/icon-train.svg" width="28" height="28" alt=""> 训练 TacMind-0

输入需要 `agentview`、`wristview` 视觉图像、关节状态、语言指令，以及
`observation.images.deformation_left/right` 中无损保存的 FOTS 位移箭头图像。
保存训练 bundle 时须一并保留生成的 `tactile_input.json`。

<a id="tacmind0-setup"></a>
### 1. 准备环境与预训练权重

先完成 [TacMind 安装](#installation)，所有命令从仓库根目录运行。
在未加载 Isaac 的新终端准备 Python 3.12 训练环境：

```bash
conda create -n tacmind0-python python=3.12 pip
conda run -n tacmind0-python python -m venv policy/TacMind-0/.venv
policy/TacMind-0/.venv/bin/python -m pip install -r policy/TacMind-0/requirements.txt
policy/TacMind-0/.venv/bin/python -m pip install --no-deps -e policy/TacMind-0
```

下载以下预训练权重，保留对应配置及 processor/tokenizer 文件：

| 资源 | 来源与保存位置 | 用途 |
| --- | --- | --- |
| OpenDM 0.5 | [Dexmal/DM05](https://huggingface.co/Dexmal/DM05) → `data/weights/DM05/` | `pretrained_path` |
| DINOv2 Small | [facebook/dinov2-with-registers-small](https://huggingface.co/facebook/dinov2-with-registers-small) → `data/weights/dinov2-with-registers-small/` | Tac-LeWM 的 `backbone_dir` |
| DINOv3 ViT-B/16 | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) → `data/weights/dinov3-vitb16/` | Tac-LeWM 的 `vision_backbone_dir` |

<a id="taclewm-pretraining"></a>
### 2. 训练 Tac-LeWM

需要含视觉、FOTS Deformation 图像和绝对关节动作的 GSmini 多任务 LeRobot 数据集。

已有使用 Deformation 图像预训练的 Tac-LeWM 权重和配套 `resources/` 时，可直接进入第 3 步。

安装预训练环境并复制配置：

```bash
conda run -n tacmind0-python python -m venv policy/TacMind-0/.venv-pretraining
policy/TacMind-0/.venv-pretraining/bin/python -m pip install -r policy/TacMind-0/requirements-pretraining.txt
mkdir -p tmp
cp policy/configs/taclewm_pretrain.json tmp/pretrain.json
```

下面的路径示例对应上述权重目录；将 `dataset_path` 改为实际的多任务数据集目录。保留模板的其他训练参数：

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

`devices` 必须等于 `CUDA_VISIBLE_DEVICES` 中所选 GPU 数量；下面示例使用 3 张 GPU。默认训练 20 轮，输出到 `data/runs/taclewm_pretrain`；
每次新训练设置新的 `output_dir`。在 TacMind 训练终端运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m policy.taclewm --config tmp/pretrain.json
```

保留训练生成的 `checkpoints/weights_epoch_N.pt`、同目录的 `tactile_input.json` 及完整的配套 `resources/` 目录，供 TacMind-0 训练使用。
默认配置对应 `checkpoints/weights_epoch_20.pt`。此入口不支持优化器断点恢复。

<a id="tacmind0-policy-training"></a>
### 3. 训练目标任务策略与 Tac-FRS

以下使用上面转换的 Precision 0 数据集。在 TacMind 训练终端中运行（`conda activate tacmind`）；
启动器会调用模型的 Python 3.12 环境。

```bash
mkdir -p tmp
cp policy/configs/tacmind0.json tmp/tacmind0_precision0.json
```

模板使用以下路径；如果数据或权重位置不同，在副本中修改：

| 字段 | Precision 0 示例 |
| --- | --- |
| `dataset.root` / `dataset.repo_id` | `data/lerobot/gsmini_precision0` / `tacmind/gsmini_precision0` |
| `runtime_python` | `policy/TacMind-0/.venv/bin/python` |
| `pretrained_path` | `data/weights/DM05` |
| `model.world_weights` | `data/runs/taclewm_pretrain/checkpoints/weights_epoch_20.pt` |
| `model.resources` | `data/runs/taclewm_pretrain/resources` |
| `output_dir` | `data/runs/precision0_tacmind0`（新目录） |

```bash
CUDA_VISIBLE_DEVICES=0 tacmind learning train --config tmp/tacmind0_precision0.json
```

默认按顺序运行以下三个阶段，产物路径相对于 `output_dir`：

| 阶段 | 默认步数 | 产物 |
| --- | ---: | --- |
| 将预训练 Tac-LeWM 适配到目标任务 | 10,000 | `tactile/checkpoint-10000/` |
| 训练 VTLA 动作策略 | 10,000 | `policy/`，以及中间的 `policy/checkpoint-<step>/` |
| 训练 Tac-FRS | 10,000 | `frs/checkpoint-10000/` |

`stages.json` 记录已完成阶段的 checkpoint 路径。评估使用 Policy 或 Tac-FRS checkpoint；
仅含触觉世界模型的权重不能作为动作策略。此统一训练入口使用单进程，不支持优化器断点恢复。
下方评估命令使用最终的 Tac-FRS checkpoint。

<a id="policy-evaluation"></a>
## <img src="docs/media/icon-evaluate.svg" width="28" height="28" alt=""> 评估策略

<a id="tacmind0-inference"></a>
### TacMind-0 推理环境

TensorRT + CUDA Graph 推理使用独立的 PyTorch 2.10 环境，需要支持 CUDA 13.0 的驱动。
在未加载 Isaac 的新终端运行：

```bash
conda run -n tacmind0-python python -m venv policy/TacMind-0/.venv-inference
policy/TacMind-0/.venv-inference/bin/python -m pip install -r policy/TacMind-0/requirements-tensorrt.txt
policy/TacMind-0/.venv-inference/bin/python -m pip install --no-deps -e policy/TacMind-0
```

### 普通条件评估

切换到上方的仿真终端；评估器会启动指定的独立推理进程。

```bash
CUDA_VISIBLE_DEVICES=0 tacmind learning evaluate \
  --checkpoint data/runs/precision0_tacmind0/frs/checkpoint-10000 \
  --policy-python "$PWD/policy/TacMind-0/.venv-inference/bin/python" \
  --episodes 20 --seed-start 10000 \
  --output-dir tmp/eval/precision0_tacmind0/nominal --headless --device cuda:0
```

ACT/DP 评估时删除 `--policy-python`，其他独立模型设置为对应环境的 Python；同时更换 checkpoint 和输出目录。

其他模型的 checkpoint 路径见下表。
评估使用 checkpoint 中的任务与配置；模型解释器迁移后用 `--policy-python` 指定。
选择未用于训练或归一化的 seed；模型比较使用相同示范、seed 与条件。
`--episodes` 须为正整数，所有 seed 须在 `[0, 2**32)` 内。
任务时长固定为最长参考轨迹加 100 个控制步，控制频率 20 Hz。

在 `results.json` 的 `score_complete=true` 时读取 `success_rate`；视频默认保存到 `videos/`，
加 `--video-mode none` 可关闭。输出目录须不存在或为空；中断后使用相同 checkpoint 与 seed 在新目录重跑。

<a id="checkpoint-layouts"></a>
### 选择 checkpoint

| 模型 | Checkpoint 路径 |
| --- | --- |
| ACT / DP | `checkpoints/<step>/pretrained_model/` |
| OpenPI / N0-VTLA | `checkpoints/tacmind/run/<step>/` |
| OpenDM | `checkpoints/checkpoint-<step>/`，或最终的 `checkpoints/` |
| ACT Tactile | `checkpoints/best/`，或 `checkpoints/<六位 step>/` |
| TacMind-0 | `stages.json` 中的 Policy/FRS 路径；默认最终产物为 `frs/checkpoint-10000/` |

复制 checkpoint 时保留权重、配置和归一化文件；符号链接需复制目标目录。
TacMind-0 使用上方的独立推理环境。

<a id="robustness-evaluation"></a>
### 扰动评估

下面对同一 TacMind-0 checkpoint 运行全部 12 种扰动，输出目录与普通评估分开：

```bash
CUDA_VISIBLE_DEVICES=0 tacmind learning evaluate \
  --checkpoint data/runs/precision0_tacmind0/frs/checkpoint-10000 \
  --policy-python "$PWD/policy/TacMind-0/.venv-inference/bin/python" \
  --perturb --episodes 20 --seed-start 10000 \
  --output-dir tmp/eval/precision0_tacmind0/perturbations --headless --device cuda:0
```

`--perturb lighting`、`camera`、`scale` 分别选择 4、6、2 种条件。
仅支持 Precision、Recog、Dynamic；Extra 仅普通评估。每个条件独立保存 `results.json`，根报告为索引。

汇总需每任务全部 13 种条件的完整成绩。**Avg InD** 先在 Precision（6）、Recog（11）、Dynamic（11）
内平均普通成绩，再平均三类。**Avg OOD** 先平均每任务的组内扰动，再平均每类任务，最后平均九个
“任务类 × 扰动组”。以百分数报告，缺失条件时不生成完整汇总。
随结果保留数据清单、配置、源码和环境版本、完整 checkpoint 与评估命令。

<a id="policies"></a>
## <img src="docs/media/icon-policies.svg" width="28" height="28" alt=""> 其他模型与配置

<a id="supported-policies"></a>
### 支持的模型

| 模型 | 配置 |
| --- | --- |
| ACT | [act/config.json](policy/act/config.json) |
| Diffusion Policy | [diffusion/config.json](policy/diffusion/config.json) |
| OpenPI（π0.5） | [configs/openpi.json](policy/configs/openpi.json) |
| OpenDM（DM05） | [configs/opendm.json](policy/configs/opendm.json) |
| N0-VTLA | [configs/n0-vtla.json](policy/configs/n0-vtla.json) |
| ACT Tactile | [configs/act_tactile.json](policy/configs/act_tactile.json) |
| TacMind-0 | [configs/tacmind0.json](policy/configs/tacmind0.json) |

<a id="source-and-environment-setup"></a>
### 环境

ACT/DP 使用[训练环境](#training-and-evaluation)。
TacMind-0 使用上方准备的独立训练和推理环境。
所有 Policy 源码均随仓库保存在 `policy/` 下。OpenPI、OpenDM、N0-VTLA
分别使用独立环境，在已有源码目录中按下表安装依赖与预训练权重，跳过重复克隆仓库的步骤。

| 模型 | 安装依据 |
| --- | --- |
| OpenPI | [README](policy/openpi/README.md)，包括 PyTorch 支持和 Transformers replacement |
| OpenDM | [README](policy/opendm/README.zh-CN.md) |
| N0-VTLA | [安装指南](policy/n0-vtla/docs/INSTALL.md)，包括模型权重、Transformers replacement 和 DINOv2 缓存 |
| ACT Tactile | 训练环境中的 Torch、Torchvision、NumPy、Pillow、safetensors；首次构建视觉 ResNet18 需下载或缓存 ImageNet 权重 |

ALOHA CAD 和 LIBERO 场景资源按需获取，运行上游硬件或仿真示例前见[资源准备](policy/optional-assets.zh-CN.md)。

<a id="training-configuration"></a>
### 训练配置

在所选模型配置的副本中填写以下字段，然后运行本节末尾的训练命令：

| 字段 | 内容 |
| --- | --- |
| `dataset.root` / `dataset.repo_id` | 数据集目录与标识 |
| `dataset.episodes` | 可选的训练 episode 索引列表；省略时使用全部 episode |
| `output_dir` | 新的训练输出目录 |
| `runtime_python` | 模型环境中的 Python 可执行文件路径；ACT/DP 不填写 |
| `pretrained_path` | OpenPI/N0-VTLA 的 PyTorch 权重目录（含 `model.safetensors`），或 OpenDM/TacMind-0 使用的 DM05 目录；ACT/DP、ACT Tactile 不填写 |

替换配置中的占位路径，相对路径以仓库根目录为基准。
即使 `dataset.episodes` 只选择部分 episode，ACT/DP 归一化仍使用整个数据集。

#### 其他模型设置

OpenPI、OpenDM、N0-VTLA、ACT Tactile 和 TacMind-0 的统一训练入口支持单进程。
ACT Tactile 和 N0-VTLA 还需 `gsmini_left`、`gsmini_right` 触觉 RGB。
ACT Tactile 使用 80/20 训练/验证划分；`model.tactile_ckpt` 可初始化触觉编码器，否则从头训练。

#### 复用准备好的数据

上述模型默认将数据准备到 `dataset.root/policy_exports/<policy_type>`。
可用 `prepared_root` 指定目录，或运行 `tacmind learning prepare --config tmp/policy.json` 单独准备。
复用时模型类型、数据集、episode 选择与 prompt 必须一致；修改这些设置时，使用新的 `prepared_root` 重新准备。
跨机器复用时复制整个准备目录，填写 `prepared_root`，省略 `dataset` 和 `prompt` 即可使用保存的设置，无需原始数据集。

<a id="baseline-training"></a>
### 启动训练与断点恢复

复制上表中的模型配置，以下以 DP 为例：

```bash
mkdir -p tmp
cp policy/diffusion/config.json tmp/policy.json
```

按上表填写数据集和输出路径，然后运行：

```bash
tacmind learning train --config tmp/policy.json
```

DP 默认配置训练 50,000 步，输出到 `data/runs/gsmini_precision0_dp`。

<a id="training-and-checkpoint-resume"></a>
ACT/DP 恢复训练需保留完整 checkpoint，包括 `pretrained_model/` 和 `training_state/`：

```bash
tacmind learning train \
  --config data/runs/gsmini_precision0_dp/checkpoints/last/pretrained_model/train_config.json \
  --resume
```

<a id="advanced"></a>
## <img src="docs/media/icon-advanced.svg" width="28" height="28" alt=""> 高级用法

使用上方准备的训练或推理环境。以下命令均从 TacMind 仓库根目录运行。

<a id="individual-stages"></a>
### 独立训练阶段

启用训练环境，再通过下表入口的 `--help` 查看各阶段参数：

```bash
source policy/TacMind-0/.venv/bin/activate
```

| 阶段 | 原生入口 |
| --- | --- |
| Tac-LeWM 微调 | `python -m tacmind0.tactile.finetune --help` |
| TacMind-0 策略训练 | `python -m tacmind0 --task train --help` |
| Tac-FRS | `python -m tacmind0.tac_frs.training --help` |

微调用 `--weights` 指定预训练权重，`--resources` 指定配套配置目录。
策略训练需要基础策略目录、微调后的 `encoder.pt` 和 `backbone_config.json`，以及 episode JSONL 和图像目录。
Tac-FRS 需要完整的微调世界模型，不能只提供 encoder。

### 策略服务

启用推理环境，指定已训练策略或 Tac-FRS bundle：

```bash
source policy/TacMind-0/.venv-inference/bin/activate
python -m tacmind0 --task inference \
  --model-config.model-name-or-path /path/to/policy_or_frs_bundle
```

默认端口为 `7897`，用 `--inference-config.port` 修改。

```python
from tacmind0.policy.client import TacMind0Client

client = TacMind0Client("http://127.0.0.1:7897")
client.reset(lane=0)
# 每个仿真步都提供左右 Deformation 图像，frame_index 从 0 连续递增。
client.observe(left_deformation, right_deformation, frame_index=0, lane=0)
actions = client.infer(agentview, wrist, state, prompt, lane=0)
```

图像是 RGB `uint8` 数组，`state` 是 8 维关节/夹爪状态；返回物理动作的形状为 `(H, 8)`。
执行已有动作块时仍须每步调用 `observe()`。并行环境使用独立 `lane`。
Tac-FRS 使用 `tacmind0.tac_frs.client.TacFRSClient`，每回合开始时 `reset()`，结束使用时 `close(lane=0)`。

<a id="multitask"></a>
### 多任务策略训练

多任务数据格式沿用[预训练要求](#taclewm-pretraining)。准备预训练 Tac-LeWM 和 OpenDM 0.5 权重，
创建配置并填写实际路径：

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

默认全局 batch 32，训练 100000 步，每 10000 步保存至 `output/policy/checkpoint-N`。
支持 1、2、4、8、16、32 个进程。`进程数 × per_device_batch_size` 必须整除 32；
中断后用同一配置重新运行 `train`，不要重复 `prepare`。
此入口的 checkpoint 不能直接用于统一评估；需要单任务评估时使用[多阶段训练](#tacmind0-policy-training)。
