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

### Experiment Setup
To run the experiments, a config file (json), such as `experiments/config/esann25/experiment_config.json`, must be passed. It specifies the models, locations of other relevant configs, where to save embeddings, CAVs, checkpoints of CBMs, plots and results.  
The setup (i.e. which datasets, protected groups and defining terms are used) is specified in several .yaml files, referenced in the config. The config must refers to four .yaml files:
- `cav_train_config`: specifies the training for CAV (training one model per protected attribute, dataset does not need class labels)
- `cbm_train_config`: specifies the training for CBM (training a CBM on all protected groups of one dataset; dataset requires class labels in addition to group labels)
- `eval_config`: specifies the datasets and protected groups for evaluation of CAV and CBM (sorted by protected attribute; can include datasets without training split and/or class labels)
- `bias_space_eval_config`: specifies the evaluation setup for bias subspaces (sorted by protected attribute; includes both defining terms for bias subpsaces and a list of datasets/ protected groups for eval)

### Reproducing ESANN 2025 Experiments
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


### Concept Prediction Methods
- Concept Activation Vectors (CAV)
- Concept Bottleneck Models (CBM)
- Bias Subspaces (referring to semantic bias scores [1][2], our implementation is based on [1])


### Datasets
The experiments cover the datasets given in the table below. 

| Dataset     | Samples | Task                             | Data Splits (for training / eval) | Label type           |
|-------------|---------|----------------------------------|-----------------------------------|----------------------|
| BIOS        | 10,563  | job classification (10 classes)  | train / test + dev                | binary, single label |
| Jigsaw      | 375,895 | toxicity classification (binary) | train / test_public_leaderboard   | binary, multi label  |
| CrowS-Pairs | 3,016   | Masked Language Modeling         | - / test                          | binary, multi label  |
| TwitterAAE  | 100,000 | /                                | - / test                          | binary, single label |

All datasets are downloaded from Huggingface Datasets and we use the respective data splits, but due to pre-processing some datasets are reduced to a smaller subset. The number of samples in the table refers to the total number of samples (train+test) used in the experiment.

#### BIOS
For the BIOS dataset, we reduce it to a supervised subset from [3] using [this Repo](https://github.com/HammerLabML/MeasuringFairnessWithBiasedData).  
[Huggingface Link](https://huggingface.co/datasets/LabHC/bias_in_bios)

#### Jigsaw
We filter for samples where a 2/3 majority of annotators agrees on all the identity labels (0 or 1). In the experiments we further use a subset of protected groups (see next table) that had a sufficiently large amount of samples.  
[Huggingface Link](https://huggingface.co/datasets/google/jigsaw_unintended_bias)

#### CrowS-Pairs
We focus on the most frequent identity labels and protected attributes that are available in the other datasets for cross-dataset transfer.  
[Huggingface Link](https://huggingface.co/datasets/nyu-mll/crows_pairs)

#### TwitterAAE
No pre-processing done.  
[Huggingface Link](https://huggingface.co/datasets/lighteval/TwitterAAE)

### Distribution of Protected Groups

The next table shows the protected groups considered in our experiment per dataset. We tested cross-dataset transfer for any combination of datasets that shared the respective protected attributes (and omitted groups that were only present in one). For the distribution of protected groups in the datasets, see [this notebook](https://github.com/HammerLabML/PreSeCoLM/blob/main/experiments/data_stats.ipynb).

|             | Gender       | Ethnicity                   | Religion                                   | Disability                 |
|-------------|--------------|-----------------------------|--------------------------------------------|----------------------------|
| BIOS        | male, female | /                           | /                                          | /                          |
| Jigsaw      | male, female | white, black, asian, latino | christian, muslim, jewish, buddhist, hindu | mental illness             |
| CrowS-Pairs | male, female | white, black, asian         | christian, muslim, jewish                  | mental illness/ disability |
| TwitterAAE  | /            | african american english    | /                                          | /                          |




### Language Models
- Huggingface Models (using this [Wrapper](https://github.com/UBI-AGML-NLP/Embeddings))
- OpenAI Embedding Models

| Source      | Model name                        | Embedding Size | Model Type (paper) |
|-------------|-----------------------------------|----------------|--------------------|
| OpenAI      | text-embedding-3-small            | 1536           | Embedder           |
| Huggingface | bert-base-uncased                 | 768            | Encoder            |
| Huggingface | bert-large-uncased                | 1024           | Encoder            |
| Huggingface | distilbert-base-uncased           | 768            | Encoder            |
| Huggingface | roberta-base                      | 768            | Encoder            |
| Huggingface | roberta-large                     | 1024           | Encoder            |
| Huggingface | gpt2                              | 768            | Decoder            |
| Huggingface | gpt2-large                        | 1280           | Decoder            |
| Huggingface | google/electra-base-generator     | 256            | Encoder            |
| Huggingface | google/electra-base-discriminator | 768            | Encoder            |
| Huggingface | albert-base-v2                    | 768            | Encoder            |


## Cite this
TODO



## References

[1] "The SAME score: Improved cosine based bias score for word embeddings", [Arxiv Paper](https://arxiv.org/abs/2203.14603), [Conference Paper](https://ieeexplore.ieee.org/abstract/document/10651275/)    
[2] "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings", [Arxiv Paper](https://arxiv.org/abs/1607.06520), [Conference Paper](https://proceedings.neurips.cc/paper_files/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)    
[3] "Measuring fairness with biased data: A case study on the effects of unsupervised data in fairness evaluation.", [Conference Paper](https://link.springer.com/chapter/10.1007/978-3-031-43085-5_11)  