# Franka + GelSight Mini

[English](README.md)

来源记录：[provenance.json](provenance.json)。
模型来自 [Tabero](https://github.com/NathanWu7/Tabero/tree/56a3b9055bfe7dc157583cc10e121c7a7781cf8f)、
[触觉资产数据集](https://huggingface.co/datasets/china-sae-robotics/Tactile_Manipulation_Dataset/tree/9929b1a9e16b51d2d9b93c6733bc8efe91c83c05)
和 [gsrobotics](https://github.com/gelsightinc/gsrobotics)。
代码许可：[Tabero](../../../benchmark/backends/gsmini/tacex/LICENSE.Tabero)、
[Taxim](../../../benchmark/backends/gsmini/tacex/simulation_approaches/gpu_taxim/sim/LICENSE)。

PhysX 使用刚性胶垫，Taxim 根据深度变化渲染光学效果。
Taxim 和 FOTS 输出不代表经过标定的胶垫形变或力测量。

使用方法见[数据采集](../../../README.zh-CN.md#collection)。
视频导出运行 `python -m benchmark.backends.gsmini.export --help`。
