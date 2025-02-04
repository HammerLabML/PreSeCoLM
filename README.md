# PreSeCoLM (Predicting Sensitive Concepts in Language Models)

This repository includes the implementation of some experiments in the scope of predicting sensitive concepts (protected attributes such as ethnicity or gender) in language models to enhance the models interpretability.
It includes the code to reproduce the papers:

- Sarah Schröder, Alexander Schulz and Barbara Hammer. "Evaluating Concept Discovery Methods for Sensitive Attributes in Language Models". Accepted at ESANN 2025.


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

### Currently Used Datasets
- BIOS
- TwitterAAE
- Jigsaw Unintended Bias
- CrowSPairs

### Currently Supported Language Models
- Huggingface Models (using this [Wrapper](https://github.com/UBI-AGML-NLP/Embeddings))
- OpenAI Embedding Models

### Concept Prediction Methods
- Concept Activation Vectors (CAV)
- Concept Bottleneck Models (CBM)
- Bias Subspaces (refering to semantic bias scores [1][2], our implementation is based on [1])

### Experiment Setup
To run the experiments, a config file (json), such as `experiments/config/esann25/experiment_config.json`, must be passed. It specifies the models, locations of other relevant configs, where to save embeddings, CAVs, checkpoints of CBMs, plots and results.  
The setup (i.e. which datasets, protected groups and defining terms are used) is specified in several .yaml files, referenced in the config. The config must refers to four .yaml files:
- `cav_train_config`: specifies the training for CAV (training one model per protected attribute, dataset does not need class labels)
- `cbm_train_config`: specifies the training for CBM (training a CBM on all protected groups of one dataset; dataset requires class labels in addition to group labels)
- `eval_config`: specifies the datasets and protected groups for evaluation of CAV and CBM (sorted by protected attribute; can include datasets without training split and/or class labels)
- `bias_space_eval_config`: specifies the evaluation setup for bias subspaces (sorted by protected attribute; includes both defining terms for bias subpsaces and a list of datasets/ protected groups for eval)


### ESANN 2025 Experiments
The configs for the experiments of the ESANN 2025 paper can be found at `experiments/config/esann25/`.

Before running the experiments, the config should be adapted (directories) and the embeddings of OpenAI models need to be computed via `get_openai_embeddings.ipynb`. If you do not have an API Key, you need to exclude the models from the config.  

To reproduce results, run the following python scripts. Computing the embeddings for all language models and training CAV and CBM models may take a while. The embeddings, CBM checkpoints and CAV predictions will be saved to the directories specified in `experiments/configs/esann25/experiment_config.json`. Make sure to adapt the paths and that sufficient space is available.  
The scripts are not intended to be run in parallel. This might lead to unnecessary computation of embeddings.

```commandline
python3 experiments/bias_space_eval.py -c experiments/configs/esann25/experiment_config.json
python3 experiments/cav_train_eval.py -c experiments/configs/esann25/experiment_config.json
python3 experiments/cbm_train_eval.py -c experiments/configs/esann25/experiment_config.json
```
   
Then use the jupyter-notebook `experiments/protected_feature_eval.ipynb` to process the results and create the plots from the paper.

## Cite this
TODO



## References

[1] "The SAME score: Improved cosine based bias score for word embeddings", [Arxiv](https://arxiv.org/abs/2203.14603) Paper, [IEEE IJCNN](https://ieeexplore.ieee.org/abstract/document/10651275/) Paper  
[2] "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings", [Arxiv](https://arxiv.org/abs/1607.06520) Paper, [NIPS](https://proceedings.neurips.cc/paper_files/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf) Paper