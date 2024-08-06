# Code for Multistain Pretraining for Slide Representation Learning in Pathology (ECCV'24)
[arXiv]() | [Proceedings]()

Welcome to the official GitHub repository of our ECCV 2024 paper, "Multistain Pretraining for Slide Representation Learning in Pathology". This project was developed at the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Brigham and Women's Hospital.

## Abstract
<img src="support/madeleine.jpeg" width="300px" align="right" />
Developing self-supervised learning (SSL) models that can learn universal and transferable representations of H\&E gigapixel whole-slide images (WSIs) is becoming increasingly valuable in computational pathology. These models hold the potential to advance critical tasks such as few-shot classification, slide retrieval, and patient stratification. Existing approaches for slide representation learning extend the principles of SSL from small images (e.g., 224x224 patches) to entire slides, usually by aligning two different augmentations (or \emph{views}) of the slide. Yet the resulting representation remains constrained by the limited clinical and biological diversity of the \emph{views}. Instead, we postulate that slides stained with multiple markers, such as immunohistochemistry, can be used as different \emph{views} to form a rich task-agnostic training signal. To this end, we introduce MADELEINE, a multimodal pretraining strategy for slide representation learning. MADELEINE is trained with a dual global-local cross-stain alignment objective on large cohorts of breast cancer samples (N=4,211 WSIs across five stains) and kidney transplant samples (N=12,070 WSIs across four stains). We demonstrate the quality of slide representations learned by MADELEINE on various downstream evaluations, ranging from morphological and molecular classification to prognostic prediction, comprising 21 tasks using 7,299 WSIs from multiple medical centers.

## Overview
<img src="support/overview.png" width="1024px" align="center" />


## Installation 

```
# Clone repo
git clone https://github.com/mahmoodlab/MADELEINE
cd MADELEINE

# Create conda env
conda create --name madeleine python=3.9
conda activate madeleine
pip install -r requirements.txt
```

## Preprocessing 

TODO.

## How to use pretrained MADELEINE slide embedder
To download the public MADELEINE checkpoint (trained on 10X breast samples) from HuggingFace and extract slide embeddings on Breast Core Needle Biopsy dataset [1], run the following command:

```
cd ./bin
python extract_slide_embeddings_from_checkpoint.py --overwrite --local_dir ../results_brca/MADELEINE
```

This command downloads the model config and weights to `./results_brca/MADELEINE` and extracts slide embeddings using MADELEINE for the BCNB dataset, saving them at `./results_brca/MADELEINE`.

## Linear probe for molecular status prediction
To run linear probe using MADELEINE on BCNB molecular status prediction, run:

```
cd ./bin
python run_linear_probing.py
```

The command performs linear probing for `k=1,10,25`, testing the data efficiency of the slide emebddings.

## How do MADELEINE slide emebddings compare with state of the art

MADELEINE slide emebddings are able to outperform various baselines, including GigaPath (Xu et al. *Nature*, 2024), on molecular status prediction:

|            | |   k=1   |      |  |   k=10  |      |  |   k=25  |      |
|------------|-----|-----|------|------|-----|------|------|-----|------|
|            | ER  | PR  | HER2 | ER   | PR  | HER2 | ER   | PR  | HER2 |
| **Mean (CONCH)** | 0.575   | 0.528   | 0.509   | 0.759    | 0.678   | 0.603   | 0.785    | 0.724   | 0.647   |
| **Mean (GigaPath)**  | 0.568   | 0.523   | 0.501   | 0.718    | 0.657   | 0.588   | 0.762    | 0.71   | 0.637   |
| **GigaPath (linear probe)**  | 0.555   | 0.514   | 0.498   | 0.691    | 0.636   | 0.577   | 0.741    | 0.689   | 0.618   |
| **MADELEINE (BRCA)** | **0.664** | **0.537**   | **0.545**   | **0.818**    | **0.756**   | **0.662**   | **0.838**    | **0.791**   | **0.706**   |

MADELEINE-SE is the model trained with stain encodings

# How to train your version of MADELEINE

## Train MADELEINE on Breast tissue using ACROBAT
```
cd ./bin

# launch pretraining without stain encodings
bash ../scripts/launch_pretrain_withoutStainEncodings.sh

# launch pretraining with stain encodings
bash ../scripts/launch_pretrain_withStainEncodings.sh

# launch both experiments
bash ../scripts/master.sh
```
NOTE: The pretrain script by default extracts the slide emebddings of the BCNB dataset used for downstream evaluation.

TIP: place the data directory on SSD for faster I/O and training. We use 3x24GB 3090Ti for training and it takes ~1 h to train MADELEINE.

## Issues 

- The preferred mode of communication is via GitHub issues.
- If GitHub issues are inappropriate, email avaidya@mit.edu (and cc gjaume@bwh.harvard.edu).
- Immediate response to minor issues may not be available.
- We cannot provide access to CONCH weights. Please refer to instructions on [CONCH GitHub page](https://github.com/mahmoodlab/CONCH).

## Cite
If you find our work useful in your research, please consider citing:

```
@inproceedings{jaume2024multistain,
  title={Multistain Pretraining for Slide Representation Learning in Pathology},
  author={Jaume, Guillaume and Vaidya, Anurag Jayant and Zhang, Andrew and Song, Andrew H and Chen, Richard J. and Sahai, Sharifa and Mo, Dandan and Madrigal, Emilio and Le, Long Phi and Mahmood Faisal},
  booktitle={European Conference on Computer Vision},
  pages={TODO},
  year={2024},
  organization={Springer}
}
```

<img src="support/joint_logo.png" width="2048px" align="center" />



