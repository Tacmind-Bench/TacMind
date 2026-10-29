# 可选的 Prefix MLP FP8 推理

[English](../en/dm05_prefix_mlp_fp8.md)

Fast backend 可以使用当前 checkpoint 专属的 TensorRT FP8 engines 替换 Gemma3
prefix MLP。量化范围是 gate、up、down 三组矩阵乘，采用逐张量 E4M3 权重与激活
scale。Engine 输入和输出为 FP16，周围的 prefix 计算仍为 BF16。视觉、attention、
action expert、action mask 和去噪步数沿用原配置。
Fast 会自动使用 checkpoint 下有效的 prepared generation；找不到时继续使用原 BF16
MLP。服务启动时的自动发现不会在线校准或构建。

FP8 会改变数值精度，需用实际部署配置评测任务成功率。

## 一键准备

`prepare_fast` 为 fast 推理离线准备完整产物。当前必须显式选择
`--prefix-mlp-precision fp8`；这个精度参数只控制 prefix MLP，不表示整个模型使用 FP8：

| 组件 | 准备的产物 | 是否需要校准 |
| --- | --- | --- |
| Vision encoder 和 projector | TensorRT FP16 engine | 不需要 FP8 校准 |
| Prefix MLP 的 gate/up/down | TensorRT FP8 engines，输入输出为 FP16 | 需要有代表性的真实输入 |

周围的 prefix 计算仍为 BF16，attention 和 action expert 沿用原来的 fast 实现。
工具依次完成输入前处理、vision FP16 engine 构建、使用同一个 vision engine 收集 MLP
校准数据、MLP FP8 engines 构建，无需再单独准备 vision engine。

安装下文的 fast 依赖后，用与部署相同的 playground 前处理真实请求：

```bash
python -m opendm.infer.prepare_fast \
  --checkpoint ./checkpoints/DM05-libero \
  --playground libero \
  --prefix-mlp-precision fp8 \
  --requests-jsonl ./calibration_requests.jsonl \
  --samples 128
```

这一步只准备产物，不启动推理服务。当前 `--prefix-mlp-precision` 仅支持 `fp8`，尚不
提供仅 BF16 的准备模式；普通 BF16 fast 推理无需执行这项校准。仅在服务启动时选择
`--inference-config.backend fast`，不会自动校准 MLP 或构建 MLP FP8 engines。

旧命令 `python -m opendm.infer.prepare_fast_fp8 ...` 保留为兼容入口：省略精度参数时
默认选择 FP8，同样准备 vision FP16 + prefix MLP FP8 的整套产物。已有脚本及兼容的
prepared 产物可继续使用。新脚本请使用 `prepare_fast` 并显式指定精度；产物目录和
`prepared.json` 格式保持兼容。

原始请求 JSONL 每行包含 `text`、`states` 和按相机顺序排列的 `images` 路径列表；视频
可以写成 `{"type":"video","url":"...","frame_idx":N}`。实际数据集中的
`prompt`、`state`、`images_1`、`images_2` 等行可直接通过 `--dataset-root` 或
`--jsonl` 输入；媒体不在 JSONL 目录树时可重复传 `--media-root`。已有前处理张量继续
支持 `--inputs file.pt`。

抽样由 seed 确定，并按任务、episode 轮转覆盖。若任务数更多或状态分布更广，应提高
默认的 128 个样本。RoboTwin2 使用 `--playground robotwin2`。自定义部署可传
`--playground-module package.module`（或 `.py` 路径）、`--dataset-name` 和可重复的
`--config-override dotted.path=JSON`；自定义模块必须显式指定 `--min-seq-len` 与
`--opt-seq-len`。

一键 FP8 准备当前只支持不超过 `1024` 的 `--prefix-capacity`，且 `--bucket` 不得超过
该容量。`2048` 等值会在创建输出目录、启动前处理或构建子进程之前被拒绝。更大容量请
使用 BF16 fast 路径；本工具不会为其构建截断到 1024 token 的 FP8 profile。

各准备阶段依次在独立进程中执行，全部成功后才发布 `prepared.json`。默认目录为
`<checkpoint>/.opendm/fast_fp8`；新构建失败不会替换旧的 ready generation。
使用默认目录时，正常启动 fast 即可自动复用兼容的 vision 和 MLP engines：

```bash
python playground/dm05_libero.py --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-libero \
  --inference-config.backend fast
```

若准备时指定了 `--output-dir ./artifacts/libero-fast-fp8`，服务启动时传入相同目录：

```bash
python playground/dm05_libero.py --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-libero \
  --inference-config.backend fast \
  --inference-config.prepared-fast-artifacts-dir ./artifacts/libero-fast-fp8
```

用 `--inference-config.no-auto-prepared-fast-artifacts` 可关闭自动选择。显式旧 vision
和 FP8 engine 路径仍保持 fail-fast，并优先于自动发现。LoRA adapter 需先合并、导出为
独立 checkpoint。

准备成功只说明产物身份与部署 contract 一致，不代表任务成功率已经认证；上线前仍需
运行对应任务评测。

## 高级手工流程

### 1. 准备环境和视觉 engine

按[推理指南](dm05_inference.md)安装 OpenDM 和 `pip install -e ".[fast-infer]"`。
在 TensorRT 支持 FP8 的 GPU 上构建，环境需要 strongly typed network 和 FP8 API。
产物记录 GPU 型号/计算能力、CUDA 和 TensorRT 版本；运行环境不匹配时必须重建。

以公开 LIBERO 模型为例：

```bash
python -m opendm.infer.build_vision_trt \
  --checkpoint ./checkpoints/DM05-libero \
  --num-images 2 \
  --onnx-path ./artifacts/libero/vision.onnx \
  --engine-path ./artifacts/libero/vision.engine \
  --force-rebuild
```

每个 checkpoint 使用独立产物目录。校准和 FP8 服务必须加载相同字节的视觉 engine；
重建视觉 engine 后，需要重新校准和构建 FP8 bundle。

### 2. 保存有代表性的真实输入

使用与部署相同的 playground 配置、前处理和相机顺序，将前处理输出保存为张量字典列表：

```python
import torch

keys = ("pixel_values", "input_ids", "attention_mask", "token_type_ids")
samples = []
for observation in representative_observations:
    # inference_config 由对应 playground 实验初始化。
    observation = dict(observation)
    observation.setdefault("robot_type", None)  # Use the playground robot default.
    raw = inference_config._prepare_model_input(**observation)
    transformed = inference_config.input_transform(raw)
    samples.append({key: transformed[key].detach().cpu() for key in keys})
torch.save(samples, "calibration_inputs.pt")
```

每条 observation 提供 `text`、已解码的 RGB PIL `images`、`states`，以及部署时使用的
机器人/控制模式等可选字段。使用覆盖目标任务、视角和状态的真实观测。至少两个不同
样本只是文件格式要求，不代表已充分校准；后续验证应使用不同 episode。

字典只包含上述四个必需张量，以及可选的 `diffusion_input_noise` 和 `action_mask`。
Token 张量形状是 `[1, S]`，图像是 `[num_images, 3, H, W]`；可选噪声为
`[1, chunk_size, 32]`，mask 为 `[1, 1, 32]`。工具使用
`torch.load(..., weights_only=True)` 读取，遇到超长输入直接拒绝。生成输入时也应
确认前处理保留了完整文本、状态和所有相机 token。

### 3. 校准与构建

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

复制或挂载产物时，须同时保留 bundle 目录及其相邻的 `.<bundle-name>.lock` 文件。
已有锁文件时，推理支持只读挂载；缺少锁文件时，需要父目录可写才能创建。
推理或构建进程运行期间，不要删除或替换该锁文件。

RoboTwin2 需要单独校准：使用对应 checkpoint、**三张图、chunk size 50**，构建参数
为 `--min-seq-len 896 --opt-seq-len 896 --max-seq-len 1024`。

## 4. 启动服务

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

所有配置的 bucket 必须在 engine 实际 profile 范围内。显式开启 FP8 后，
缺失、不兼容或损坏的产物会导致启动失败；history 输入和超出 bucket 的请求直接
拒绝，不会静默退回 BF16。未选中 FP8 产物时，BF16 fast 路径可在 `fast_prefix_len`
范围内对超过最大 bucket 的 prefix 使用 eager。当前支持 BF16 静态 fast backend 和标准 Gemma3 MLP；
LoRA adapter 请先合并导出，再构建 engines。
