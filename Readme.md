

<h2 align="center">
  ContourFormer:Real-Time Contour-Based End-to-End Instance Segmentation Transformer
</h2>


<p align="center">
    <a href="https://github.com/talebolano/Contourformer/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <a href="https://arxiv.org/abs/2501.17688">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2501.17688-red">
    </a>
    <!-- <a href="https://github.com/talebolano/Contourformer">
        <img alt="stars" src="https://img.shields.io/github/stars/talebolano/Contourformer">
    </a> -->
</p>

<p align="center">
    📄 This is the official implementation of the paper:
    <br>
    <a href="https://arxiv.org/abs/2501.17688">ContourFormer:Real-Time Contour-Based End-to-End Instance Segmentation Transformer</a>
</p>

<!-- ## 🚀 Updates
- [x] **\[2025.02.04\]** Release ContourFormer series. -->

## Weights
Download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1w3RYsNJD8v5ax_4ymDYxPBk5hdeUGdpH?usp=drive_link).

## Quick start

### Setup

```shell
conda create -n contourformer python=3.11.9
conda activate contourformer
pip install -r requirements.txt
```


### Data Preparation

<details>
<summary> COCO2017 Dataset </summary>

1. Download COCO2017 from [OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017) or [COCO](https://cocodataset.org/#download).
2. Modify paths in [coco_poly_detection.yml](./configs/dataset/coco_poly_detection.yml)

    ```yaml
    train_dataloader:
        img_folder: /data/COCO2017/train2017/
        ann_file: /data/COCO2017/annotations/instances_train2017.json
    val_dataloader:
        img_folder: /data/COCO2017/val2017/
        ann_file: /data/COCO2017/annotations/instances_val2017.json
    ```

</details>

<details>
<summary> SBD Dataset </summary>

1. Download COCO format SBD Dataset from [here](https://drive.google.com/file/d/12EW4frUd9wL95gjUQek9U1_BRDrGsy2Y/view?usp=sharing).

2. Modify paths in [sbd_poly_detection.yml](./configs/dataset/sbd_poly_detection.yml)

    ```yaml
    train_dataloader:
        img_folder: /data/sbd/img/
        ann_file: /data/sbd/annotations/sbd_train_instance.json
    val_dataloader:
        img_folder: /data/sbd/img/
        ann_file: /data/sbd/annotations/sbd_trainval_instance.json
    ```

</details>

<details>
<summary>KINS dataset</summary>

1. Download the Kitti dataset from the official [website](http://www.cvlibs.net/download.php?file=data_object_image_2.zip).

2. Download the annotation file instances_train.json and instances_val.json from [KINS](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset).

3. Organize the dataset as the following structure:
    ```
    ├── /path/to/kitti
    │   ├── testing
    │   │   ├── image_2
    │   │   ├── instances_val.json
    │   ├── training
    │   │   ├── image_2
    │   │   ├── instances_train.json
    ```

4. Modify paths in [kins_poly_detection.yml](./configs/dataset/kins_poly_detection.yml)
    ```yaml
    train_dataloader:
        img_folder: /data/kins_dataset/training/image_2/
        ann_file: /data/kins_dataset/training/instances_train.json
    val_dataloader:
        img_folder: /data/kins_dataset/testing/image_2/
        ann_file: /data/kins_dataset/testing/instances_val.json
    ```


</details>

## Demo

```
 python draw.py -c configs/contourformer/contourformer_hgnetv2_b3_sbd.yml -r weight/contourformer_hgnetv2_b3_sbd.pth -i your_image.jpg
```

## Usage
<details open>
<summary> COCO2017 </summary>

<!-- <summary>1. Training </summary> -->
1. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/contourformer/contourformer_hgnetv2_b2_coco.yml --seed=0
```

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/contourformer/contourformer_hgnetv2_b3_coco.yml --seed=0
```

<!-- <summary>2. Testing </summary> -->
2. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/contourformer/contourformer_hgnetv2_b2_coco.yml --test-only -r model.pth
```

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/contourformer/contourformer_hgnetv2_b3_coco.yml --test-only -r model.pth
```
</details>

<details open>
<summary> SBD </summary>

<!-- <summary>1. Training </summary> -->
1. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/contourformer/contourformer_hgnetv2_b2_sbd.yml --seed=0
```

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/contourformer/contourformer_hgnetv2_b3_sbd.yml --seed=0
```

<!-- <summary>2. Testing </summary> -->
2. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/contourformer/contourformer_hgnetv2_b2_sbd.yml --test-only -r model.pth
```

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/contourformer/contourformer_hgnetv2_b3_sbd.yml --test-only -r model.pth
```
</details>


<details open>
<summary> KINS </summary>

<!-- <summary>1. Training </summary> -->
1. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=7777 --nproc_per_node=8 train.py -c configs/contourformer/contourformer_hgnetv2_b2_kins.yml --seed=0
```

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=7777 --nproc_per_node=8 train.py -c configs/contourformer/contourformer_hgnetv2_b3_kins.yml --seed=0
```

<!-- <summary>2. Testing </summary> -->
2. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=7777 --nproc_per_node=8 train.py -c configs/contourformer/contourformer_hgnetv2_b2_kins.yml --test-only -r model.pth
```

```shell
CUDA_VISIBLE_DEVICES=0,1,2,34,5,6,7 torchrun --master_port=7777 --nproc_per_node=8 train.py -c configs/contourformer/contourformer_hgnetv2_b3_kins.yml --test-only -r model.pth
```
</details>



## Citation
If you use `ContourFormer` or its methods in your work, please cite the following BibTeX entries:
<details open>
<summary> bibtex </summary>

```latex
@misc{yao2025contourformerrealtimecontourbasedendtoendinstance,
      title={ContourFormer:Real-Time Contour-Based End-to-End Instance Segmentation Transformer}, 
      author={Weiwei Yao and Chen Li and Minjun Xiong and Wenbo Dong and Hao Chen and Xiong Xiao},
      year={2025},
      eprint={2501.17688},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.17688}, 
}
```
</details>

## Acknowledgement
Our work is built upon [D-FINE](https://github.com/Peterande/D-FINE).
Thanks to the inspirations from [D-FINE](https://github.com/Peterande/D-FINE) and [PolySnake](https://github.com/fh2019ustc/PolySnake).

✨ Feel free to contribute and reach out if you have any questions! ✨
