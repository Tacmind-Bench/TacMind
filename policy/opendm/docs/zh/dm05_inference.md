# DM05 推理指南

本文档是 DM05 推理服务启动和 HTTP API 使用的统一指南。Benchmark 与训练文档只链接
到这里，不再重复维护推理配置。

[English](../en/dm05_inference.md)

可选的 TensorRT Prefix MLP FP8 后端参见
[校准、engine 构建与部署指南](dm05_prefix_mlp_fp8.md)。

## 1. 使用前准备

所有命令都在 OpenDM 仓库根目录运行。请准备：

- 按照主 README 安装完成的 OpenDM 环境。
- 如果要使用 `--inference-config.backend fast`，还需要在同一个环境中执行
  `pip install -e ".[fast-infer]"` 安装额外依赖层。
- 与所选 playground 入口匹配的 DM05 checkpoint。
- checkpoint 目录下与训练配置匹配的 `norm_stats.json`。如果不存在，OpenDM 会回退到
  `./norm_stats/` 下的匹配文件。RoboColiseum 入口除外：它们默认要求文件位于 checkpoint
  目录中，除非显式传入 `--norm-stats`。
- 一张 NVIDIA GPU。

一个 `norm_stats.json` 可以在 `norm_stats_by_robot` 中保存多个机型 profile。
OpenDM 根据 `observation.robot_type` 精确选择；未传入时使用数据集配置的默认机型，
没有数据集默认值时再使用文件的 `default_robot_type`。历史单 profile 文件仍受支持，
未知机型不会静默回退到默认 profile。

Checkpoint、playground 入口、`chunk_size`、image keys、state/action 维度和归一化统计
必须来自同一套训练配置。

对于 fast backend，还需要满足和代码路径一致的运行前置条件：

- TensorRT 不是可选项。fast 启动时会先构建或加载 vision TensorRT engine，然后才开始
  对外提供服务。
- Triton 不是可选项。fast suffix big-kernel 和 fast prefix decoder path 都会直接导入并
  调用 Triton kernels。
- PyTorch FlexAttention 支持是必需项。fast 推理会强制把 LLM attention backend 切到
  `flex_attention`，而 static prefix fastpath 也会逐层校验 decoder layer 是否使用该 backend。
- 请使用提供 `torch.nn.attention.flex_attention` 的 PyTorch 版本，例如 `torch>=2.5`。

### Fast Backend 启动前检查

第一次启动 fast backend 前，请确认：

- 目标环境中 CUDA GPU 可见。
- 当前环境已经执行过 `pip install -e ".[fast-infer]"`。
- `python -c "import tensorrt"` 可以成功执行。
- `python -c "import triton"` 可以成功执行。
- `python -c "import torch.nn.attention.flex_attention"` 可以成功执行。
- checkpoint、playground 入口、`chunk_size`、action 维度和 `image_prompts` 来自同一轮训练。
- 上传图片的数量和顺序与 `--inference-config.image-prompts` 完全一致。
- 为第一次启动预留额外时间，因为服务会先导出 ONNX 并构建 TensorRT engine，之后 HTTP
  服务才会就绪。

## 2. 选择推理入口

| 使用场景 | 入口 | 常用 checkpoint | Chunk size | 图片数 | API state/action 维度 |
| --- | --- | --- | ---: | ---: | --- |
| 基础预训练模型 | `opendm/exp/dm05_exp.py` | `Dexmal/DM05` | 50 | 3 | 14 / 14 |
| LIBERO | `playground/dm05_libero.py` | `Dexmal/DM05-libero` | 10 | 2 | 8 / 7 |
| RoboTwin 2.0 | `playground/dm05_robotwin2.py` | `Dexmal/DM05-robotwin2` | 50 | 3 | 14 / 14 |
| RoboColiseum G2 | `playground/dm05_robocoliseum.py` | 匹配 `Instruction Following`、`Spatial Reasoning` 和 `Robustness` 套件的 RoboColiseum checkpoint | 50 | 3 | 16 / 16（内部 22 维） |
| RoboColiseum manipulation | `playground/dm05_robocoliseum_manipulation.py` | 匹配 `General Manipulation` 套件的 RoboColiseum checkpoint | 30 | 3 | 21 / 21（内部 22 维） |
| DM05-MEM | `playground/dm05_mem_sft_demo.py` | `Dexmal/DM05-MEM` | 50 | 3 | 14 / 14 |
| Demo 或自定义 SFT | `playground/dm05_sft_demo.py` 或自定义入口 | SFT checkpoint | 训练值 | 训练值 | 训练值 |
| LIBERO LoRA | `playground/dm05_libero_lora.py` | LIBERO LoRA step checkpoint | 10 | 2 | 8 / 7 |

不要混用不同 benchmark 的 checkpoint、入口或推理维度。

## 3. 启动 Default Backend

Default backend 的 VLM prefix 使用标准 PyTorch 路径。在 CUDA eval 模式下，默认的
`sdpa` action attention 会自动对 action suffix 使用 CUDA Graph replay；不满足条件时
安全回退 eager，不需要安装额外的 `fast-infer` 依赖。

Suffix Graph profile 按执行 shape 延迟创建：新 profile 第一次出现时使用 eager，第二次
出现时 capture，后续请求复用。Prefix 长度按 16 tokens 向上分桶，最多保留 8 个
profiles，因此不同长度的 prompt 不需要固定的 Graph prefix 上限；capture 或 replay
失败也会回退 eager。

推理模式下 launcher 会直接启动一个 Python 进程，不需要传 `--nproc_per_node`。

### DM05 基础预训练模型

下载基础预训练模型：

```bash
hf download Dexmal/DM05 \
  --local-dir ./checkpoints/DM05
```

启动服务：

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

基础预训练模型使用三路图像输入和 14 维 state/action。将
`observation.robot_type` 设为 `DOS W1` 或 `Aloha`，可以选择对应的归一化
profile；省略该字段时使用 checkpoint 中配置的默认 profile `DOS W1`。

直接使用基础预训练模型时，应显式提供 `observation.control_mode` 和
`observation.speed`，使输入与预训练条件保持一致。省略
`observation.speed` 时，服务默认使用 `"0.5"`。

### LIBERO

下载发布的 checkpoint：

```bash
hf download Dexmal/DM05-libero \
  --local-dir ./checkpoints/DM05-libero
```

启动服务：

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

LIBERO 入口使用两张图片、8 维 state、7 维 action 和 `Franka` robot type。

### RoboTwin 2.0

下载发布的 checkpoint：

```bash
hf download Dexmal/DM05-robotwin2 \
  --local-dir ./checkpoints/DM05-robotwin2-bf16
```

启动服务：

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

RoboTwin 2.0 入口使用三张图片、14 维 state/action 和 `Aloha RoboTwin2` robot type。

### RoboColiseum G2

请使用与所选 RoboColiseum 布局匹配的 checkpoint。基础 G2 服务启动命令为：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_robocoliseum.py \
  --task inference \
  --model-config.model-name-or-path /path/to/robocoliseum-g2-checkpoint \
  --inference-config.port 7891
```

API 接受 RoboColiseum 原生顺序的 16 维 state：

```text
[左臂关节 (7), 左夹爪, 右臂关节 (7), 右夹爪]
```

服务返回相同 16 维顺序的 50 个 action。入口会在内部完成原生布局与 checkpoint 共享
22 维布局之间的映射，因此 `--inference-config.output-action-dim` 应保留入口默认值 `22`。
该入口要求按 `Head`、`Left wrist`、`Right wrist` 顺序提供三张当前图像，不接受历史图像。

30 步全身 manipulation checkpoint 使用以下入口：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_robocoliseum_manipulation.py \
  --task inference \
  --model-config.model-name-or-path /path/to/robocoliseum-manipulation-checkpoint \
  --inference-config.port 7891
```

该变体接受并返回原生 21 维布局：

```text
[左臂关节 (7), 左夹爪, 右臂关节 (7), 右夹爪, 躯干 (5)]
```

响应包含 30 个 action。模型只预测最后一个躯干维度；前四个躯干值会从请求 state 复制
到每个 action。入口在内部完成共享布局和夹爪符号转换。该变体同样使用三张当前图像、
不接受历史图像，并保持内部 `output_action_dim=22`。

两个入口默认都从 checkpoint 目录下的 `norm_stats.json` 读取归一化统计。如果文件位于
其他位置，请传入 `--norm-stats /path/to/norm_stats.json`。

### DM05-MEM

`playground/dm05_mem_sft_demo.py` 是 DM05-MEM 的训练和推理入口。完整训练流程参考 [DM05-MEM SFT 与验证指南](dm05_mem_sft.md)。

下载发布的 checkpoint：

```bash
hf download Dexmal/DM05-MEM \
  --local-dir ./checkpoints/DM05-MEM
```

启动服务：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_mem_sft_demo.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-MEM \
  --inference-config.port 7891
```

DM05-MEM 入口使用三张图片、14 维 state/action 和 `DOS W1`
robot type。默认 `observation.control_mode` 为 `joint`；需要末端位姿控制时传
`eef`。默认 `speed` 为 `"0.1"`。历史帧默认 32；FastInfer 需要匹配的 35 图 TensorRT engine。
验证本入口刚训出的 checkpoint 时，将 `model-name-or-path` 换成对应 step 目录，并用带
`history_images` 的请求；完整例子见 [DM05-MEM SFT 与验证指南](dm05_mem_sft.md) 第 6 节。

### Demo 或自定义 SFT

使用训练产出的 checkpoint，并保持所有数据相关配置与训练一致。内置 demo 的命令为：

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

使用自定义数据时，将入口、dataset name、checkpoint、chunk size、image keys 和 action
维度替换为训练时使用的值。

### LIBERO LoRA

将 LoRA step checkpoint 作为 `model-name-or-path`。Loader 会读取其中的
`adapter_config.json`，加载记录的 base model，并合并 adapter 用于推理。

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

## 4. 启动 Fast Backend

Fast backend 使用 TensorRT vision encoder、优化后的 attention/MLP kernels 和启动时
预捕获的 CUDA Graph profiles 来降低推理延迟。

安装 fast backend 必需的依赖层：

```bash
pip install -e ".[fast-infer]"
```

`fast-infer` 会安装 `onnx`、`triton==3.6.0` 和 `tensorrt`。fast backend 缺少这些组件时
不会自动回退：TensorRT 用于准备 vision engine，Triton 是 prefix/suffix fast kernels 的
硬依赖，而 PyTorch 中也必须提供 `flex_attention`。

若需要可选的 prefix MLP FP8 优化，先离线同时准备 **vision FP16 engine 和 prefix MLP
FP8 engines**：

```bash
python -m opendm.infer.prepare_fast \
  --checkpoint ./checkpoints/DM05-libero \
  --playground libero \
  --prefix-mlp-precision fp8 \
  --requests-jsonl ./calibration_requests.jsonl \
  --samples 128
```

精度参数只控制 MLP。Vision 保持 FP16，无需 FP8 校准；MLP 校准使用刚构建的 vision
engine 和真实部署输入。该命令只准备产物，不启动服务，当前必须显式传入
`--prefix-mlp-precision fp8`，尚不提供仅 BF16 的准备模式。普通 BF16 fast 推理无需执行
这一步。旧命令 `prepare_fast_fp8` 保留为默认选择 FP8 的兼容入口。
校准数据、自定义输出目录和当前 1024 token 限制，见
[一键准备指南](dm05_prefix_mlp_fp8.md#一键准备)。

在对应的 default backend 命令中加入 `--inference-config.backend fast`。未指定 vision/FP8
路径和外部 prepared 目录时，启动流程会优先选择 checkpoint 下兼容的 prepared FP8
generation；没有选中时，公共 BF16 fast 路径才使用默认 vision engine
`checkpoints/trt_engines/dm05_vision.engine`。LIBERO 完整示例：

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

在 BF16 fast 路径中，如果 vision engine 不存在，launcher 不会立刻对外提供服务，而是会先
导出 ONNX、构建 TensorRT engine，再完成配置好的 CUDA Graph profiles capture，最后
HTTP 服务才会就绪。每个 checkpoint 和图片布局应使用独立的 engine 路径。现有 engine 目前只按图片
数量决定是否复用，因此更换 checkpoint 时必须换新路径，或者传
`--inference-config.force-rebuild`。

为 BF16 fast 路径手动构建 vision FP16 engine：

```bash
python -m opendm.infer.build_vision_trt \
  --checkpoint ./checkpoints/DM05-libero \
  --onnx-path checkpoints/trt_engines/dm05_vision.onnx \
  --engine-path checkpoints/trt_engines/dm05_vision.engine \
  --num-images 2
```

`--num-images` 默认等于 `--inference-config.image-prompts` 的数量。若启动时加了
`--data-config.is-history`，则为 `len(image_prompts) + max_history_images`。父类
默认 5 个 history 槽（三路当前图：`--num-images 8`）。MEM demo playground 为 32
槽（`--num-images 35`）；RoboDojo `cover_blocks` 为 20（`--num-images 23`）。

### Fast Backend 约束

- 服务就绪时间包含 TensorRT engine 准备和 CUDA Graph profile capture；第一次启动通常会
  比 default backend 明显更慢。
- 请求使用 batch size 1，服务串行处理请求。
- `diffusion_steps` 在服务启动并 capture profiles 后固定。
- 预处理后的 multimodal prefix 上限由 `--inference-config.fast-prefix-len` 决定（父类
  默认 `1024`）。MEM demo 和 RoboDojo `cover_blocks` 设为 `2048`。
- 父类默认 prefix buckets 为 `576 704 768 896 1024`；无法容纳当前图片数量的过小默认值
  会被自动跳过。MEM demo 和 `cover_blocks` 使用单个 bucket `2048`。Default backend
  不使用 prefix buckets。
- RoboColiseum manipulation 在两个 backend 上都使用 `4096` token 上限；其 fast backend
  默认 buckets 为 `896 1024 2048 4096`。
- 自定义 bucket 列表必须非空、严格递增，并且不能超过 `fast_prefix_len`。
- Bucket 越多，服务启动时间和显存占用越高。
- BF16 fast 路径中，请求超过最大自定义 bucket、但仍不超过 `fast_prefix_len` 时，会使用
  更慢的 eager fallback。Prepared 或显式 FP8 要求请求落入已配置 bucket，超出时直接拒绝。

只有 workload 需要不同 prefix shape 时才覆盖默认值：

```bash
--inference-config.prefix-seq-len-buckets 576 704 768 896 1024
```

所有配置的 profiles 都会在 HTTP 服务就绪前完成 capture。

## 5. 调用 HTTP API

服务提供 `POST /v1/infer`，请求格式为 JSON。图片通过 base64 字符串传入，并使用连续的
1-based 槽位键名，这样 benchmark checkpoint、demo checkpoint 和自定义 SFT 服务都可
以共用同一套接口。

`POST /process_frame` 作为 legacy multipart 接口仍然保留，用于兼容已有客户端。新的
接入方应优先使用 `/v1/infer`。这个 legacy 接口会逐步被替换。

一个包含两张图片的 LIBERO 请求如下：

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

请求字段：

- `observation.prompt`：任务指令，默认是空字符串。
- `observation.state`：必填的一维 JSON array，长度和顺序必须与 checkpoint 的归一化统计一致。
- `observation.images`：必填 JSON 对象，值为 base64 编码图片。键名必须是连续的 1-based
  字符串（`"1"`、`"2"`、…），并与 `--inference-config.image-prompts` **按顺序一一对应**：
  例如 `"1"` → 第 1 个 prompt（如 `Head`），`"2"` → 第 2 个（如 `Left wrist`）。
- `observation.history_images`：可选的 base64 历史帧 JSON 数组，**从旧到新**。父类默认上限 5 帧。
  Default 与 Fast 都遵循 `--inference-config.max-history-images`：MEM demo 为 `32`（同时设置
  `--trainer-config.model-max-length 2048` 和 `fast_prefix_len=2048`）；RoboDojo
  `cover_blocks` 为 `20`。仅当服务启动时加了 `--data-config.is-history` 才可使用；无历史时
  省略或传 `[]`。
- `observation.robot_type`：用于说明 state/action 语义的可选机器人类型。Benchmark 入口会
  继承数据集默认值，例如 `Franka` 和 `Aloha RoboTwin2`。多机型 checkpoint 会按字段值
  精确选择 profile，例如 `Aloha` 或 `DOS W1`；自定义 relative-action 入口可能要求显式传入。
- `observation.control_mode` 和 `observation.speed`：文本条件。直接使用 `Dexmal/DM05`
  基础预训练模型时应显式传入；SFT checkpoint 仅在训练数据包含这些字段时需要传入。
  服务默认 `speed` 为 `"0.5"`。
- `sampling`：可选 JSON 对象。`num_steps` 必须与服务固定的 diffusion steps 一致，`seed`
  可用于固定采样随机性。

正常响应包含 action chunk 和端到端 API 延迟（毫秒）：

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

### VLM Chat Completions 接口

`POST /v1/chat/completions` 为多模态 VLM 评测实现了 OpenAI Chat Completions
响应 schema。与 `/v1/infer` 不同，该接口返回文本，不需要机器人 state，支持任意数量的
图片与文本交错输入，也不会添加动作推理使用的 `Robot:`、`Control mode:`、
`Overall speed:`、`Task:` 或 `States:` 前缀。两个接口复用同一个已加载的 checkpoint
和 processor。

请求可以使用纯字符串 content，也可以使用 OpenAI 风格的 content parts：

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

图片必须是 data URL 或裸 base64 字符串。接口拒绝外部 HTTP(S) 图片 URL，也不支持
video/audio part。`max_tokens` 和 `max_completion_tokens` 都可使用，默认值为 1024。
`temperature` 省略或为 0 时使用 greedy decoding；大于 0 时启用采样。`stop` 可以是
字符串或字符串数组。为了兼容 lmms-eval，未知顶层字段会被忽略；但服务不实现 SSE，
所以 `stream: true` 会被拒绝。

成功响应遵循 OpenAI schema：

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

VLM chat 配置：

- `--inference-config.enable-vlm-chat`：是否启用该接口，默认 `true`。
- `--inference-config.chat-max-new-tokens`：服务端输出上限，默认 2048。请求值超过
  上限时会被 clamp，并记录 warning。
- `--inference-config.chat-max-input-tokens`：tokenize 后的最大 prompt 长度，默认 8192；
  超过限制时返回 HTTP 400。

使用 `--inference-config.backend fast` 时，动作推理仍走 fast runtime，但 chat 生成会
特意走标准 PyTorch VLM 路径。如果该配置下 chat 生成失败，请改用
`--inference-config.backend default`。

通过 lmms-eval 的 OpenAI backend 跑图片 benchmark：

```bash
OPENAI_API_KEY=EMPTY python3 -m lmms_eval \
  --model openai \
  --model_args model=dm05,base_url=http://127.0.0.1:7891/v1,api_key=EMPTY,num_concurrent=1 \
  --tasks mme \
  --batch_size 1 \
  --log_samples \
  --output_path ./lmms_results/dm05
```

该服务推荐使用 `num_concurrent=1` 到 `4`。对于 `max_tokens` 上千的长输出任务，务必
使用 `num_concurrent=1`：Flask 以 `threaded=False` 单线程串行处理请求。lmms-eval
默认的 32 路并发会让队尾请求撞到 OpenAI SDK 默认的 600 秒超时，随后产生的重试还会
进一步倒灌并压满队列。

### Legacy `/process_frame` 接口

兼容接口接收 `multipart/form-data`，并通过重复的 `image` 文件字段上传图片。
新接入请优先用上面的 `/v1/infer` JSON；legacy 用重复表单字段，而不是 `"1"`/`"2"` 键。

```bash
curl -X POST http://127.0.0.1:7891/process_frame \
  -F 'text=pick up the black bowl and place it on the plate' \
  -F 'states=[0,0,0,0,0,0,0,0]' \
  -F 'robot_type=Franka' \
  -F image=@/path/to/agentview.jpg \
  -F image=@/path/to/wrist.jpg
```

Legacy 请求字段：

- `text`：任务指令，默认是空字符串。
- `states`：必填的一维 JSON array，长度和顺序必须与 checkpoint 的归一化统计一致。
- `image`：可重复的图片字段，数量和顺序必须与 `image_prompts` 一致（如 Head、Left wrist、Right wrist）。
- `history_images`：可选的历史帧（重复 multipart 字段，从旧到新）。父类默认上限 5 帧；
  两个 backend 都遵循 `--inference-config.max-history-images`（MEM demo `32`，
  `cover_blocks` `20`）。仅当服务启动时加了 `--data-config.is-history` 才可使用。
- `robot_type`：用于说明 state/action 语义的可选机器人类型。Benchmark 入口会继承数据集默认值，
  例如 `Franka` 和 `Aloha RoboTwin2`。多机型 checkpoint 会按字段值精确选择 profile，例如
  `Aloha` 或 `DOS W1`；自定义 relative-action 入口可能要求显式传入。
- `control_mode` 和 `speed`：文本条件。直接使用 `Dexmal/DM05` 基础预训练模型时应显式
  传入；SFT checkpoint 仅在训练数据包含这些字段时需要传入。服务默认 `speed` 为
  `"0.5"`。

Legacy 成功响应保持历史格式：

```json
{
  "response": [
    [0.012, -0.034, 0.18, 0.0, 0.0, 0.0, -1.0],
    [0.015, -0.031, 0.17, 0.0, 0.0, 0.0, -1.0]
  ],
  "model_latency_ms": 71.884
}
```

## 6. 常用参数

| 参数 | 说明 |
| --- | --- |
| `--exp` | 与 checkpoint 和任务匹配的 playground 入口。 |
| `--model-config.model-name-or-path` | Full model 或 LoRA checkpoint 目录。 |
| `--model-config.chunk-size` | Action horizon，必须与训练和客户端一致。 |
| `--trainer-config.model-max-length` | Tokenized multimodal prefix 最大长度。 |
| `--inference-config.diffusion-steps` | Action diffusion steps，默认值为 `10`。 |
| `--inference-config.output-action-dim` | 内部归一化 action 维度，通常与 API action 维度相同；两个 RoboColiseum 入口均保持 `22`，再将响应映射为 16 维或 21 维。 |
| `--inference-config.image-prompts` | 有序相机标签，与 `observation.images` 的 `"1"`、`"2"`、… 一一对应。 |
| `--inference-config.max-history-images` | 两个 backend 的历史帧请求上限。父类默认 `5`；MEM demo `32`；RoboDojo `cover_blocks` `20`。 |
| `--inference-config.fast-prefix-len` | Fast backend 的 prefix token 容量。父类默认 `1024`；MEM demo 和 `cover_blocks` 为 `2048`；RoboColiseum manipulation 为 `4096`。 |
| `--data-config.is-history` | 开启后才接受 `history_images`；普通 `curl_demo.sh` 不需要。 |
| `--inference-config.backend` | `default` 或 `fast`。 |
| `--inference-config.vision-trt-engine-path` | 显式指定当前 checkpoint 专用的 TensorRT vision engine，并跳过 prepared 自动选择；BF16 fallback 默认路径为 `checkpoints/trt_engines/dm05_vision.engine`。 |
| `--inference-config.force-rebuild` | Fast 推理前重新构建 vision engine。 |
| `--inference-config.prefix-seq-len-buckets` | 可选的非空 fast-backend 自定义 buckets，每个值必须 `<= fast_prefix_len`。 |
| `--inference-config.no-auto-prepared-fast-artifacts` | 关闭默认启用的 checkpoint 内兼容离线 FP8 generation 自动选择。 |
| `--inference-config.prepared-fast-artifacts-dir` | `prepare_fast --prefix-mlp-precision fp8`（旧入口：`prepare_fast_fp8`）生成的外部 prepared generation 根目录，包含 vision FP16 和 prefix MLP FP8 engines。 |
| `--inference-config.port` | HTTP 服务端口，默认值为 `7891`。 |

## 7. 常见问题

| 错误或现象 | 检查项 |
| --- | --- |
| 找不到归一化统计或统计不匹配 | 使用 checkpoint 的 `norm_stats.json`，并保持 dataset/action/chunk 配置与训练一致。 |
| State 或 action 维度错误 | 让 `observation.state` 和 `output_action_dim` 与归一化向量维度一致。 |
| 上传图片数量错误 | 让 `observation.images` 的数量和顺序与 `image_prompts` 一致。 |
| Fast 启动阶段出现 import 错误 | 在当前环境重新执行 `pip install -e ".[fast-infer]"`，并检查 `import tensorrt`、`import triton`、`import torch.nn.attention.flex_attention` 是否成功。 |
| TensorRT 图片数量不匹配 | 用与 `image_prompts` 数量相同的 `--num-images` 重建；开启 history 时用 `len(image_prompts) + max_history_images`。 |
| 更换 checkpoint 后复用同一 engine 导致结果异常 | 使用 checkpoint 专用 engine 路径，或者传 `--inference-config.force-rebuild`。 |
| 预期使用 BF16，但启动时启用了 prepared FP8 | 传 `--inference-config.no-auto-prepared-fast-artifacts`；显式 engine 路径仍有更高优先级。 |
| Prefix buckets 为空、乱序或过大 | 传入非空递增列表，且所有值不超过 `fast_prefix_len`。 |
| Fast prefix 超过最大 bucket | BF16 在 `fast_prefix_len` 内可走 eager；FP8 会拒绝，应使用覆盖该请求的 prepared profile 或关闭 FP8 自动选择。 |
| Default backend 偶尔有单次请求较慢 | 新执行 profile 第一次走 eager，第二次出现时 capture；后续匹配请求会复用 Graph。 |
| Fast 服务长时间未就绪 | 等待 engine 准备和所有 CUDA Graph profiles 在启动阶段完成 capture。 |

## 8. 相关指南

- [DM05 LIBERO 训练与评测](dm05_libero.md)
- [DM05 RoboTwin 2.0 训练与评测](dm05_robotwin2.md)
- [DM05 RoboDojo-Sim 训练与评测](dm05_robodojo.md)
- [DM05 SFT 与验证](dm05_finetuning.md)
- [DM05-MEM SFT 与验证](dm05_mem_sft.md)
- [DM05 LIBERO LoRA 训练](dm05_libero_lora_training.md)
