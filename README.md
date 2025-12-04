# PreSeCoLM (Predicting Sensitive Concepts in Language Models)

This repository includes the implementation of some experiments in the scope of predicting sensitive concepts (protected attributes such as ethnicity or gender) in language models to enhance the models interpretability.
It includes the code to reproduce the papers:

- Sarah Schröder, Alexander Schulz and Barbara Hammer. "Evaluating Concept Discovery Methods for Sensitive Attributes in Language Models". Accepted at ESANN 2025.
- Sarah Schröder, Valerie Vaquet and Barbara Hammer. "Linearity of Sensitive Concepts in Language Models". Submitted to ESANN 2026.


## Installation

Create and activate conda environment:
```commandline
conda env create -f env.yml
conda activate presecolm
```

Install our Wrapper for Huggingface Embeddings:
```commandline
git clone https://github.com/UBI-AGML-NLP/Embeddings.git
cd Embeddings/
pip install .
```


## Experiment Details

### Experiment Setup
To run the experiments, a config file (json), such as `experiments/config/esann25/experiment_config.json`, must be passed. It specifies the models, locations of other relevant configs, where to save embeddings, CAVs, checkpoints of CBMs, plots and results.  
The setup (i.e. which datasets, protected groups and defining terms are used) is specified in several .yaml files, referenced in the config. The config must refers to four .yaml files:
- `cav_train_config`: specifies the training for CAV (training one model per protected attribute, dataset does not need class labels)
- `cbm_train_config`: specifies the training for CBM (training a CBM on all protected groups of one dataset; dataset requires class labels in addition to group labels)
- `eval_config`: specifies the datasets and protected groups for evaluation of CAV and CBM (sorted by protected attribute; can include datasets without training split and/or class labels)
- `bias_space_eval_config`: specifies the evaluation setup for bias subspaces (sorted by protected attribute; includes both defining terms for bias subpsaces and a list of datasets/ protected groups for eval)

### ESANN 2025 Experiments
See the [esann25 branch](https://github.com/HammerLabML/PreSeCoLM/tree/esann25).

### ESANN 2026 Experiments
See the [esann26 branch](https://github.com/HammerLabML/PreSeCoLM/blob/esann26)

