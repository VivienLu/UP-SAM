# UP-SAM
Official Implementation for "UP-SAM: Uncertainty-Informed Adaptation of Segment Anything Model for Semi-Supervised Medical Image Segmentation"

# UP-SAM: Uncertainty-Informed Adaptation of Segment Anything Model for Semi-Supervised Medical Image Segmentation"

[![arXiv](https://img.shields.io/badge/BIBM-2024-blue)](https://ieeexplore.ieee.org/abstract/document/10822398)

Official PyTorch implementation of UP-SAM from the paper: "UP-SAM: Uncertainty-Informed Adaptation of Segment Anything Model for Semi-Supervised Medical Image Segmentation" Accepted by BIBM, 2024.

## Contents

- [Abstract](##Abstract)
- [Datasets](##Datasets)
- [Usage](##Usage)
- [Acknowledgment](##Acknowledgment)

## Abstract

![avatar](./fig/UP-SAM.pdf)

Semi-supervised segmentation is extensively employed in medical image analysis due to its ability to leverage a small amount of labeled data alongside abundant unlabeled data. However, its performance is hindered by the inadequate knowledge of the data domain learned from limited labeled data and the absence of effective strategies for exploiting unlabeled regions, especially when annotations are extremely scarce. To address these challenges, the Segment Anything Model (SAM) has emerged as a promising solution. As a foundation model enriched by extensive and diverse domain knowledge, SAM has been leveraged to mitigate the epistemic uncertainty (EU) of semi-supervised segmentation models, while aleatoric uncertainty (AU) is often ignored. In this paper, we propose a novel semi-supervised medical image segmentation framework called UP-SAM, which adapts SAM for dual uncertainty assessments. The framework achieves effective collaboration between large foundation models and domain-specific models, leading to a simultaneous reduction in the impact of EU and AU. The experiments on the left atrium and pancreas datasets demonstrate the superior efficacy of UP-SAM against baseline methods. Particularly, UP-SAM exhibits substantial advantages over other semi-supervised learning models when dealing with exceedingly scarce labeled data. 

## Datasets

**Dataset licensing term**:

* Left atrium dataset: http://atriaseg2018.cardiacatlas.org
* Pancreas dataset: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT

## Usage

### 1. Clone the repo.;

   ```
   git clone
   ```

### 2. Download the [SAM-Med3D-turbo checkpoint](https://github.com/uni-medical/SAM-Med3D#checkpoint);

Its path is passed to `train_test.sh` as the final argument.

### 3. All training and inference commands are provided in `train_test.sh`. Run the script from the repository root:


```bash
bash train_test.sh \
  METHOD \
  DATA_DIR \
  SAVE_PATH \
  DATASET \
  LABEL_NUM \
  BATCH_SIZE \
  GPU \
  SAM_CHECKPOINT
```

Arguments:

| Argument | Description |
| --- | --- |
| `METHOD` | Experiment name, for example `UPSAM` |
| `DATA_DIR` | Root of the preprocessed LA or Pancreas-CT dataset |
| `SAVE_PATH` | Directory used for checkpoints, logs, and predictions |
| `DATASET` | `LA` or `Pancreas` |
| `LABEL_NUM` | Number of labeled training volumes |
| `BATCH_SIZE` | Total batch size; half of each stage-2 batch is labeled |
| `GPU` | CUDA device index, for example `0` |
| `SAM_CHECKPOINT` | Path to `sam_med3d_turbo.pth` |

Example for the Left Atrium dataset with two labeled volumes:

```bash
bash train_test.sh \
  UPSAM \
  /path/to/LA_dataset \
  ./results \
  LA \
  2 \
  4 \
  0 \
  /path/to/sam_med3d_turbo.pth
```

For the one-labeled-volume setting, use a total batch size of 2:

```bash
bash train_test.sh \
  UPSAM \
  /path/to/LA_dataset \
  ./results \
  LA \
  1 \
  2 \
  0 \
  /path/to/sam_med3d_turbo.pth
```

Example for Pancreas-CT:

```bash
bash train_test.sh \
  UPSAM \
  /path/to/Pancreas-processed \
  ./results \
  Pancreas \
  2 \
  4 \
  0 \
  /path/to/sam_med3d_turbo.pth
```

The script sequentially performs:

1. VNet pre-training for 7,500 iterations with SGD and a learning rate of
   `1e-2`.
2. UP-SAM semi-supervised training for 7,500 iterations with AdamW and a
   learning rate of `1e-5`.
3. Sliding-window inference using the best domain-specific checkpoint.

Generated files follow this structure:

```text
SAVE_PATH/
└── DATASET_lab-LABEL_NUM/
    └── METHOD/
        ├── pretrain/
        │   └── best_model.pth
        └── semi_train/
            ├── best_model.pth
            ├── best_sam_model.pth
            └── test_prediction/
                ├── record_log.txt
                └── *_UPSAM_results.npy
```

Testing reports Dice, Jaccard index, 95% Hausdorff distance, and average
symmetric surface distance.

## Repository structure

```text
.
├── dataloaders/       # LA and Pancreas-CT dataset readers
├── networks/          # VNet and four-head VNet
├── SAM_Med3D/         # SAM-Med3D model and prompting components
├── utils/             # Losses, stochastic alignment, and evaluation
├── train.py           # Stage-1 and stage-2 training
├── test.py            # Sliding-window inference
└── train_test.sh      # End-to-end training and inference
```

## Citation

```bibtex
@inproceedings{lu2024upsam,
  title     = {UP-SAM: Uncertainty-Informed Adaptation of Segment Anything Model
               for Semi-Supervised Medical Image Segmentation},
  author    = {Lu, Wenjing and Hong, Yi and Yang, Yang},
  booktitle = {2024 IEEE International Conference on Bioinformatics and
               Biomedicine (BIBM)},
  pages     = {2256--2261},
  year      = {2024},
  doi       = {10.1109/BIBM62325.2024.10822398}
}
```

## Acknowledgment

Part of the code is adapted from the open-source codebase and original implementations of algorithms, we thank these authors for their fantastic and efficient codebase:

*  SSL4MIS: https://github.com/HiLab-git/SSL4MIS
*  UPCoL: https://github.com/VivienLu/UPCoL
*  FUSSNet: https://github.com/grant-jpg/FUSSNet
*  SAM-Med3D: https://github.com/uni-medical/SAM-Med3D
