# PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Spatio-Temporal State Space Model

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![arXiv](https://img.shields.io/badge/arXiv-2408.03540-b31b1b.svg)](https://arxiv.org/abs/2408.03540)

This is the official PyTorch implementation of our AAAI 2025 paper *"[PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional
Spatio-Temporal State Space Model](https://arxiv.org/pdf/2408.03540v2)"*.

## Environment

The project is developed under the following environment:

- Python 3.8.5
- PyTorch 1.13.1+cu117
- torchvision 0.14.1+cu117
- torchaudio 0.13.1+cu117
- CUDA 11.7

For installation of the project dependencies, please run:

```
conda create -n posemamba python=3.8.5
conda activate posemamba
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
cd kernels/selective_scan && pip install -e .
```

## Dataset

### Human3.6M

#### Preprocessing

1. Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d', or direct download our processed data [here](https://drive.google.com/file/d/1WWoVAae7YKKKZpa1goO_7YcwVFNR528S/view?usp=sharing) and unzip it.
2. Slice the motion clips by running the following python code in `tools/convert_h36m.py`:

```text
python convert_h36m.py
```

### MPI-INF-3DHP

#### Preprocessing

Please refer to - [MotionAGFormer](https://github.com/taatiteam/motionagformer) for dataset setup. 

## Training

After dataset preparation, you can train the model as follows:

### Human3.6M

You can train Human3.6M with the following command:

```
CUDA_VISIBLE_DEVICES=0 python train.py --config <PATH-TO-CONFIG> --checkpoint <PATH-TO-CHECKPOINT>
```

where config files are located at `configs/h36m`. 

### MPI-INF-3DHP

Please refer to - [MotionAGFormer](https://github.com/taatiteam/motionagformer) for training. 

## Evaluation

We provide [checkpoint](https://drive.google.com/file/d/1WFRAeal8W6ntrTPNrf-SNywdgupj0-S8/view?usp=sharing). You can download and unzip it to get pretrained weight. 

| Method     | frames          | Params | MACs | Human3.6M weights                                            |      
| ----------- | --------------- | ------ | ---- | ------------------------------------------------------------ | 
| PoseMamba-S | 243 | 0.9M   | 3.6G | [PoseMamba-S](https://drive.google.com/file/d/1LZtEjeiAIx6LXFmjoyKKzbaCPV3R1-P7/view?usp=sharing)  
| PoseMamba-B  | 243 | 3.4M  | 13.9G  |    [PoseMamba-B](https://drive.google.com/file/d/1aP6WAq5fKNIqyYcI_ZnYbuagR3_zVik2/view?usp=sharing) 
| PoseMamba-L  | 243 |  6.7M  | 27.9G  |    [PoseMamba-L](https://drive.google.com/file/d/16_Tg0Aqzgih243_dflyFv0UB79gU9u8q/view?usp=sharing)    

After downloading the weight, you can evaluate Human3.6M models by:

```
python train.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```

For example if PoseMamba-L of H.36M is downloaded and put in `checkpoint` directory, then we can run:

```
python train.py --eval-only --checkpoint checkpoint --checkpoint-file PoseMamba-l-h36m.pth.tr --config configs/h36m/PoseMamba-large.yaml
```

## Demo

Our demo is a modified version of the one provided by [MotionAGFormer](https://github.com/taatiteam/MotionAGFormer) repository. First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in the './demo/lib/checkpoint' directory. Next, download our base model checkpoint from [here](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view) and put it in the './checkpoint' directory. Then, you need to put your in-the-wild videos in the './demo/video' directory.
We provide [demo](https://drive.google.com/file/d/1hbK1HDz1nMTGYcczOC5r33Mk8nAtLZCr/view?usp=sharing). You can download and unzip it to get demo file.
Run the command below:

```
python vis.py --video sample_video.mp4 --gpu 0
```

Sample demo output:

<p align="center">
<img src='sample_video.gif' width="60%" alt="no img" />
</p>



## Acknowledgement

Our code refers to the following repositories:

- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [MotionAGFormer](https://github.com/taatiteam/MotionAGFormer)
- [VMamba](https://github.com/mzeromiko/vmamba)

We thank the authors for releasing their codes.

## Citation

If you find our work useful for your project, please consider citing the paper:

```
@article{huang2024posemamba,
  title={PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model},
  author={Huang, Yunlong and Liu, Junshuo and Xian, Ke and Qiu, Robert Caiming},
  journal={arXiv preprint arXiv:2408.03540},
  year={2024}
}
```
