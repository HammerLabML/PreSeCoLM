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

### ESANN 2025 Experiments
See the [esann25 branch](https://github.com/HammerLabML/PreSeCoLM/tree/esann25).

### ESANN 2026 Experiments
See the [esann26 branch](https://github.com/HammerLabML/PreSeCoLM/blob/esann26)

