# 可选上游资源

[English](optional-assets.md) · [Policy 指南](../README.zh-CN.md#policies)

TacMind 策略训练与评估使用 TacMind 资源。ALOHA 硬件 CAD 和 LIBERO 仿真资源为可选项，
不随 Policy 源码分发。

运行上游 OpenPI LIBERO 示例前，先下载并解压配套资源。从 TacMind 仓库根目录执行：

```bash
mkdir -p data/downloads policy/openpi/third_party/libero/libero/libero/assets
curl -fL https://github.com/Lifelong-Robot-Learning/LIBERO/archive/f78abd68ee283de9f9be3c8f7e2a9ad60246e95c.tar.gz \
  -o data/downloads/libero-assets.tar.gz
tar -xzf data/downloads/libero-assets.tar.gz --strip-components=4 \
  -C policy/openpi/third_party/libero/libero/libero/assets \
  LIBERO-f78abd68ee283de9f9be3c8f7e2a9ad60246e95c/libero/libero/assets
```

然后按 [OpenPI LIBERO 示例](openpi/examples/libero/README.md)运行。
制作 ALOHA 硬件时，使用[对应版本的上游 CAD 文件](https://github.com/Physical-Intelligence/aloha/tree/d1dc83afd89ded4379851257fe5d85632d31d5ec/aloha2)。
