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

### ESANN 2025 Experiments
TODO refer to branch/ other readme


## Cite this
TODO



## References

[1] "The SAME score: Improved cosine based bias score for word embeddings", [Arxiv](https://arxiv.org/abs/2203.14603) Paper, [IEEE IJCNN](https://ieeexplore.ieee.org/abstract/document/10651275/) Paper  
[2] "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings", [Arxiv](https://arxiv.org/abs/1607.06520) Paper, [NIPS](https://proceedings.neurips.cc/paper_files/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf) Paper