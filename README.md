# physical_ai_av

This repository contains a python developer kit and documentation (in the form of a [wiki](https://github.com/NVlabs/physical_ai_av/wiki) and interactive [notebooks](notebooks/)) for working with the [NVIDIA Physical AI Autonomous Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles), one of the largest, most geographically diverse collections of multi-sensor data empowering AV researchers to build the next generation of Physical AI based end-to-end driving systems.

## Support

📣 **Usage questions and discussion about Physical AI AV dataset & devkit**: please join us on the [Alpamayo NV Developer Forum](https://forums.developer.nvidia.com/c/autonomous-vehicles/alpamayo/766).

🐛 **Code-level bugs, documentation issues, and feature requests**: file a [GitHub issue](../../issues/new/choose) using the appropriate template (Bug report, Documentation request, or Feature request). The relevant NVIDIA responder is auto-assigned via the `assignees:` field on the template.

🚨 **Security vulnerabilities**: please use [NVIDIA's Vulnerability Disclosure Program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). Do not file security issues publicly here.

## Installation & Setup

```
pip install physical_ai_av
```

To use this package to access the data hosted on Hugging Face, you'll need to:

- [Create a Hugging Face account](https://huggingface.co/join) (if you don't have one already).
- Login and agree to the NVIDIA Autonomous Vehicle Dataset License Agreement visible at the top of the [PhysicalAI AV dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
- Create a [User Access Token](https://huggingface.co/docs/hub/en/security-tokens) (if you don't have one already) and choose a method for [authentication](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).

## Splits

Split labels live in `splits/<name>.parquet` on the [dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles#splits) and are attached to `clip_index` through `split_name`:

```python
from physical_ai_av import PhysicalAIAVDatasetInterface

dataset = PhysicalAIAVDatasetInterface(split_name="alpamayo_aligned")        # train / test / alpasim
dataset.clip_index["split"]

dataset = PhysicalAIAVDatasetInterface(split_name="challenging_scenarios")   # Alpamayo open-loop evaluation set
dataset.clip_index                                                           # only its 1,156 clips, with `t0` anchors
dataset.clip_index["t0"].explode()                                           # 1,434 (clip_id, t0) pairs

dataset = PhysicalAIAVDatasetInterface(split_name=None)                      # no split column
```

A split file that lists only part of the dataset restricts `clip_index`, `feature_presence`, and `data_collection` to the clips it lists. The default, `split_name="default"`, is deprecated and logs a warning: that split is not aligned with the training data of the released Alpamayo models, so use `alpamayo_aligned` for leakage-free evaluation.
