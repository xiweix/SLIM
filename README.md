<div align="left">

# SLIM
Xiwei Xuan, Ziquan Deng, Hsuan-Tien Lin, and Kwan-Liu Ma
##### [ECCV'24] SLIM : Spuriousness Mitigation with Minimal Human Annotations

</div>

<div align="center">
  <span style="display: inline-block;">
    <a href="https://arxiv.org/abs/2407.05594">
      <img src="https://img.shields.io/badge/arXiv-2407.05594-b31b1b" alt="arXiv">
    </a>
  </span>
  <span style="display: inline-block;">
    <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06273.pdf">
      <img src="https://img.shields.io/badge/ECCV%202024-PDF-FACE27" alt="ECCV 2024 PDF">
    </a>
  </span>
  <span style="display: inline-block;">
    <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06273-supp.pdf">
      <img src="https://img.shields.io/badge/ECCV%202024-Supp-7DCBFF" alt="ECCV 2024 Supp">
    </a>
  </span>
  <span style="display: inline-block;">
    <a href="#citation">
      <img src="https://img.shields.io/badge/ECCV%202024-Bibtex-CB8CEA" alt="ECCV 2024 Bibtex">
    </a>
  </span>
</div>

<a name="intro"></a>

## Introduction

Recent studies highlight that deep learning models often learn spurious features mistakenly linked to labels, compromising their reliability in real-world scenarios where such correlations do not hold. Despite the increasing research effort, existing solutions often face two main challenges: they either demand substantial annotations of spurious attributes, or they yield less competitive outcomes with expensive training when additional annotations are absent. In this paper, we introduce SLIM, a cost-effective and performance-targeted approach to reducing spurious correlations in deep learning. Our method leverages a human-in-the-loop protocol featuring a novel attention labeling mechanism with a constructed attention representation space. SLIM significantly reduces the need for exhaustive additional labeling, requiring human input for fewer than 3% of instances. By prioritizing data quality over complicated training strategies, SLIM curates a smaller yet more feature-balanced data subset, fostering the development of spuriousness-robust models. Experimental validations across key benchmarks demonstrate that SLIM competes with or exceeds the performance of leading methods while significantly reducing costs. The SLIM framework thus presents a promising path for developing reliable models more efficiently. 

#### Video
[SLIM-YouTube](https://www.youtube.com/watch?v=Gzo_r3e49B0)

#### Poster
<!-- Show SLIM's poster here -->
<img src="./SLIM_Poster.png" alt="SLIM's Poster" style="width: 100%; border: 1px solid gray;">

<a name="requirements"></a>

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.1 [(PyTorch Official - Get Started)](https://pytorch.org/get-started/locally/)

Other dependencies can then be installed using the following command:
```yaml
pip install -r requirements.txt
```

Alternatively, if you are using conda, a conda environment named ```slim``` with packages installed can be created by:
```yaml
conda env create -f environment.yml -n slim
```

<a name="data"></a>

## Datasets

Please follow the below link to download and organize datasets.
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [Waterbirds](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz)

The code and datasets are organized as:

```
- datasets
    - celeba
        - img_align_celeba
        - metadata.csv
    - waterbirds
        - waterbird_complete95_forest2water2
        - metadata.csv
    - ...
- slim_code (this repository)
    - ...
```

<a name="train"></a>
### 1. Train a model

To train a reference model:

```sh
python model_training.py --dataset waterbirds --data_dir [path-to-waterbirds-data-dir] --exp_info reference_model_training
```

This command trains a ResNet50 on the waterbirds dataset. You can modify the command to run different experiments with different hyperparameters or on different datasets.

<a name="gradcam"></a>
### 2. Generate GradCAM Results and evaluate the model attention

To generate GradCAM visual explanation results and produce the attention evaluation score (AIOU):

```sh
python get_gradcam.py
```

Note that the data folder, model architecture, and trained model path, etc., can be modified to obtain GradCAM on other datasets and models.

### 3. Obtain feature vectors

```sh
python get_feature_vectors.py
```

Similarly, the data folder, model architecture, and trained model path, etc., can be modified to get feature vectors from other datasets and models.

### 4. Curate data subset

This involves a human-in-the-loop step of annotating sampled data. The details are provided in the notebook `slim_data_sampling.ipynb`.

### 5. Re-train and evaluate the model

After data curation, we can re-train the model and evaluate its performance with [`1. Train a model`](#train) and [`2. Generate GradCAM Results and evaluate the model attention`](#gradcam).

## Citation

If you find our work useful, please cite it using the following BibTeX entry:

```bibtex
@InProceedings{Xuan_2024_ECCV,
author = {Xuan, Xiwei and Deng, Ziquan and Lin, Hsuan-Tien and Ma, Kwan-Liu},
title = {SLIM: Spuriousness Mitigation with Minimal Human Annotations},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {September},
year = {2024}
}
```

<a name="acknowledgement"></a>

## Acknowledgement

This GradCAM implementation is based on [PyTorch-Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam).
We also refer to the wonderful repositories of our related works, including [GroupDRO](https://github.com/kohpangwei/group_DRO), [Correct-N-Contrast](https://github.com/HazyResearch/correct-n-contrast), and [DFR](https://github.com/PolinaKirichenko/deep_feature_reweighting).
