# Official Implementation of 《Weakly Supervised Multitype Lesion Locating for Reliable Cervical Screening》

## Installation

1. `conda create --name wscernet python=2.10`
2. `conda activate wscernet`
3. `pip install -r requirements.txt`
4. `pip install -e .`

## Download Pre-trained model weights
Please download weights and save in `checkpoints/`.  
1. For Vit-L and DINOV2, download from open source code [mmpretrain](https://github.com/open-mmlab/mmpretrain/tree/main)
2. For Resnet50, download from open source code [timm](https://huggingface.co/timm)
3. For UNI, download from [UNI github url](https://github.com/mahmoodlab/UNI/tree/main)

## Prepare Dataset
1. Download Dataset  
For public dataset CDetector, you can download from [here](https://github.com/kuku-sichuan/ComparisonDetector).
2. Format Dataset  
After downloading, configure the storage path in the file `scripts/convert/cdetector2coco.py` and execute the following command:  
`python scripts/convert/cdetector2coco.py`.  
This script formats CDetector to COCO2017.
3. Convert Dataset  
   Configure the storage path and execute the following command:  
   `python scripts/convert/cdetector2multilabel.py`
4. (optional) Extract mini CDetector (with small lesion area)
   Configure the storage path and execute the following command:  
   `scripts/convert/cdetector_mini.py`

## Train WSCerNet
1. Set configure in `configs/dataset/cdetector_dataset.py` and `configs/train_strategy.py`
2. Execute the following command:  
   ```
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --nproc_per_node=8 --master_port=12342 main4WSCerNet.py \
    configs/dataset/cdetector_dataset.py \
    configs/train_strategy.py \
    --record_save_dir log/cdetector
    ```

## Test WSCerNet
1. Execute the following command:  
   ```
   CUDA_VISIBLE_DEVICES=0,1,2 torchrun  --nproc_per_node=3 --master_port=12340 test_wscernet.py \
    log/cdetector/path_to_your_dir/config.py \
    log/cdetector/path_to_your_dir/checkpoints/best.pth \
    log/cdetector/path_to_your_dir
   ```

## Heatmap Visualization

Execute the following command:  
   ```
   python scripts/analyze/heatmap_v4.py \
    log/cdetector_ours/path_to_your_dir/config.py \
    log/cdetector_ours/path_to_your_dir/checkpoints/best.pth \
    statistic_results/WSI_heatmap/cdetector
   ```