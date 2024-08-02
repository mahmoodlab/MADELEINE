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
conda create -n madeleine
conda activate madeleine
pip install -r requirements.txt

# install torch with cuda
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

## Code 

The workflow is as follows:

0. Tissue segmentation and patch feature extraction of pre-training and downstream datasets
1. Train MADELEINE slide embedder model (HE-multi-stain alignment)
2. Use MADELEINE checkpoint to extract slide embeddings of a downstream dataset
3. Perform linear probe evaluation using the MADELEINE extracted slide embeddings

We now explain how to run each step.


## Preprocessing 

TODO.

## Train MADELEINE on Breast tissue using ACROBAT
```
cd ./bin

# launch pretraining without stain encodings
bash ../scripts/launch_pretrain_withoutStainEncodings.sh

# launch pretraining with stain encodings
bash ../scripts/launch_pretrain_withStainEncodings.sh
```
NOTE: The pretrain script by default extracts the slide emebddings of the BCNB dataset used for downstream evaluation.

TIP: place the data directory on SSD for faster I/O and training. We use 3x24GB 3090Ti for training.

## Evaluate MADELEINE on BCNB molecular status prediction

To evaluate a checkpoint, run:

```
cd ./bin
# update file with the checkpoint you want to evaluate
python run_linear_probing.py
```

If you want to load a checkpoint and only extract the slide embeddings, run:
```
cd ./bin
# update file with the checkpoint to extract slide emebddings
python extract_slide_embeddings_from_checkpoint.py
```

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



