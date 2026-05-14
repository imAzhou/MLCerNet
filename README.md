# MLCerNet

Official implementation of **Multi-Label Learning for Reliable Cervical Cytology
Screening**. (ICIC2026)


## Overview

This repository contains PyTorch code for multi label cervical lesion classification and localization on patch-level whole-slide image data. The main model is implemented as `MLCerNet` in `mlcernet/nets/MLCerNet.py`, with baseline backbones under `mlcernet/nets/baseline/`.

Main entry points:

- `main4MLCerNet.py`: train MLCerNet.
- `test_mlcernet.py`: evaluate a trained MLCerNet checkpoint.
- `test_baseline.py`: evaluate baseline models.
- `scripts/analyze/heatmap.py`: generate attention heatmap visualizations.
- `scripts/convert/`: convert CDetector annotations into the formats used by training and evaluation.

## Installation

```bash
conda create --name mlcernet python=3.10
conda activate mlcernet
pip install -r requirements.txt
pip install -e .
```

## Pre-trained Weights

Download pre-trained backbone weights and place them under `checkpoints/`.

- ViT-L and DINOv2: download from [MMPreTrain](https://github.com/open-mmlab/mmpretrain/tree/main).
- ResNet-50: download from [timm](https://huggingface.co/timm).
- UNI: download from the [UNI repository](https://github.com/mahmoodlab/UNI/tree/main).
- SmartCCS: download from the [Smart-CCS repository](https://github.com/hjiangaz/Smart-CCS).

The default checkpoint paths are configured in:

- `configs/train_stratefy.py`
- `configs/baseline.py`

`MLCerNet` uses `smartccs` as the default backbone in `configs/train_stratefy.py`. The default SmartCCS checkpoint path is `checkpoints/CCS_vitl_100M.pth`. 

## Dataset Preparation

The provided configs target the public CDetector dataset.

1. Download CDetector from [ComparisonDetector](https://github.com/kuku-sichuan/ComparisonDetector).
2. Set `data_root` in the dataset config, for example:

   ```python
   data_root = 'CervicalDatasets/ComparisonDetectorDataset'
   ```

3. Convert the raw annotations to COCO-style annotations:

   ```bash
   python scripts/convert/cdetector2coco.py
   ```

4. Convert the COCO-style annotations to the multilabel patch format used by this project:

   ```bash
   python scripts/convert/cdetector2multilabel.py
   ```

5. Optionally extract the mini CDetector split with small lesion areas:

   ```bash
   python scripts/convert/cdetector_mini.py
   ```

Available dataset configs:

- `configs/dataset/cdetector_dataset.py`
- `configs/dataset/mini_cdetector_dataset.py`
- `configs/dataset/l_cerscan_dataset.py`

The classifier dataset loader expects this directory layout:

```text
{data_root}/
  images/{prefix}/{filename}
  annofiles/train_patches.json
  annofiles/val_annojson.json
```

Each annotation JSON file is a list of image items:

```json
[
  {
    "prefix": "Pos",
    "filename": "sample.png",
    "diagnose": 1,
    "gtmap_14": [[0, 0, 1], [3, 4, 2]]
  }
]
```

`diagnose` is the binary image label. `gtmap_14` is a list of positive token labels in `[row, col, class_id]` format, where `class_id` starts from 1. Use an empty list for negative samples.

## Training

Configure the dataset and training strategy before running:

- Dataset: `configs/dataset/cdetector_dataset.py`
- Training strategy: `configs/train_stratefy.py`

Example distributed training command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=8 \
  --master_port=12342 \
  main4MLCerNet.py \
  configs/dataset/cdetector_dataset.py \
  configs/train_stratefy.py \
  --record_save_dir log/cdetector
```

Training logs, copied configs, and checkpoints are saved under `--record_save_dir`.

## Evaluation

Evaluate a trained MLCerNet checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
  --nproc_per_node=3 \
  --master_port=12340 \
  test_mlcernet.py \
  log/cdetector/path_to_your_run/config.py \
  log/cdetector/path_to_your_run/checkpoints/best.pth \
  log/cdetector/path_to_your_run
```

Evaluate a baseline checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master_port=12345 \
  test_baseline.py \
  log/baseline/path_to_your_run/config.py \
  log/baseline/path_to_your_run/checkpoints/best.pth \
  log/baseline/path_to_your_run
```

Baseline backbone selection is controlled by `baseline_backbone` in `configs/baseline.py`. Supported values are `resnet50`, `vit`, `dinov2`, and `uni`.

## Heatmap Visualization

Generate attention heatmaps from a trained checkpoint:

```bash
python scripts/analyze/heatmap.py \
  log/cdetector/path_to_your_run/config.py \
  log/cdetector/path_to_your_run/checkpoints/best.pth \
  statistic_results/WSI_heatmap/cdetector
```

## Contact

If you have any questions, please contact us via email: zhoulyaxx@zju.edu.cn
