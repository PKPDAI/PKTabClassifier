
# PKTabClassifier

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/v-smith/PKTabClassifier/blob/master/LICENSE) ![version](https://img.shields.io/badge/version-0.1.0-blue) 


[**About the Project**](#about-the-project) | [**Dataset**](#dataset) | [**Getting Started**](#getting-started-) | [**Usage**](#usage) | [**Licence**](#lincence) | [**Citation**](#citation)

## About the Project

This repository contains custom pipes and models to classify tables contained in scientific publications in PubMed Open Access, depending on whether they contain pharmacokinetic (PK) parameter estimates from _in vivo_ studies or associated study population characteristic information.


#### Project Structure

- The main code is found in the root of the repository (see Usage below for more information).

```
├── annotation guidelines # used by annotators for annotating data in this project
├── configs # config files for training and inference arguments. 
├── pk_tableclass # code for data preprocessing, post-processing, and prompt templates.
├── scripts  # scripts for model training and inference.
├── .gitignore
├── LICENCE
├── README.md
├── requirements.txt
└── setup.py
```

#### Built With

[![Python v3.9](https://img.shields.io/badge/python-v3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)



## Dataset

The annotated PKTableClassification (PKTC) corpus can be downloaded from [zenodo](https://zenodo.org/records/13884895). The data is available under an MIT licence. The code assumes data is located in the `data` folder. 

## Getting Started 

#### Installation

To clone the repo:

`git clone https://github.com/PKPDAI/PKTabClassifier`
    
To create a suitable environment:
- ```conda create --name PKTabClassifier python==3.9```
- `conda activate PKTabClassifier`
- `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
- `pip install -e .`

#### GPU Support

Using GPU is recommended. Single-GPU training has been tested with:
- `NVIDIA® GeForce RTX 30 series`
- `cuda 12.2`

## Usage

#### Train the supervised classifier pipeline:

````bash
python scripts/train_xgb_classifier.py \
--path-to-config configs/config.json \
--train-data-path data/train.pkl \
--val-data-path data/validation.pkl \
--model-save-dir trained_models/
````

#### Evaluate the supervised classifier: 

```python
python scripts/evaluate_xgb_classifier.py \
--path-to-config configs/config.json \
--test-data-path data/test.pkl \
--path-to-trained-model trained_models/best_classifier.pkl

```

#### Evaluate zero-shot classification. Please note to run this script you will need aan OpenAI API key and organization key which you will need to add to the config file.
```python
python scripts/evaluate_zero_shot_classifier.py \
--path-to-config configs/config.json \
--test-data-path data/test.pkl

```

#### Evaluate combined supervised & zero-shot classification. Please note to run this script you will need aan OpenAI API key and organization key which you will need to add to the config file.
```python
python scripts/evaluate_combined_approach.py \
--path-to-config configs/config.json \
--test-data-path data/test.pkl \
--path-to-trained-model trained_models/best_classifier.pkl \
--confidence-threshold 0.9
```

#### Inference with the supervised classifier: 

```python
python scripts/inference.py \
--path-to-config configs/config.json \
--path-to-trained-model trained_models/best_classifier.pkl \
--inference-data-path data/inference.pkl \
--confidence-threshold 0.9 \
--batch-size 500

```

## License

The codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

[mit]: LICENCE

## Citation

```bibtex
@article{Smith2025,
  author       = {Smith, Victoria C. and Gonzalez Hernandez, Ferran and Wattanakul, Thanaporn and Chotsiri, Palang and Cordero, José Antonio and Ballester, Maria Rosa and Duran, Màrius and Fanlo Escudero, Olga and Lilaonitkul, Watjana and Standing, Joseph F. and Kloprogge, Frank},
  title        = {An automated classification pipeline for tables in pharmacokinetic literature},
  journal      = {Scientific Reports},
  year         = {2025},
  volume       = {15},
  number       = {1},
  pages        = {10071},
  doi          = {10.1038/s41598-025-94778-5},
  url          = {https://doi.org/10.1038/s41598-025-94778-5},
  issn         = {2045-2322}
}
```

