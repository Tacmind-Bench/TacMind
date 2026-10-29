# Franka + GelSight Mini

[中文](README.zh-CN.md)

Sources: [provenance.json](provenance.json).
Models derive from [Tabero](https://github.com/NathanWu7/Tabero/tree/56a3b9055bfe7dc157583cc10e121c7a7781cf8f),
the [tactile asset dataset](https://huggingface.co/datasets/china-sae-robotics/Tactile_Manipulation_Dataset/tree/9929b1a9e16b51d2d9b93c6733bc8efe91c83c05)
and [gsrobotics](https://github.com/gelsightinc/gsrobotics).
Code licenses: [Tabero](../../../benchmark/backends/gsmini/tacex/LICENSE.Tabero),
[Taxim](../../../benchmark/backends/gsmini/tacex/simulation_approaches/gpu_taxim/sim/LICENSE).

PhysX uses rigid gelpads; Taxim renders depth-derived optical changes.
Taxim and FOTS outputs are not calibrated gel deformation or force measurements.

Usage: [data collection](../../../README.md#collection).
For video exports, run `python -m benchmark.backends.gsmini.export --help`.
