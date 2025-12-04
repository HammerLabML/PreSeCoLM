# PreSeCoLM (Predicting Sensitive Concepts in Language Models)

This branch includes the implementation of our experiments for the paper:

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

### Reproducing ESANN 2026 Experiments
The configs for the experiments of the ESANN 2026 paper can be found at `experiments/config/concept_eval/`.

Before running the experiments, the config should be adapted (directories) and the embeddings of OpenAI models need to be computed via `get_openai_embeddings.ipynb`. If you do not have an API Key, you can either exclude the models from the config or contact the first author.  

To reproduce results, run the following python scripts. Running the python scripts might take a while, especially if you need to compute the embeddings first. The scripts will save embeddings to avoid unnecessary computation. Make sure you have enough disk space or run the experiments with fewer/ smaller models first.

To train and evaluate the classifiers for our first experiment run:
```commandline
python3 experiments/concept_eval/evaluate_concepts.py -c experiments/configs/concept_eval/exp1.json
python3 experiments/concept_eval/evaluate_concepts.py -c experiments/configs/concept_eval/exp2.json
```

To train and evaluate the classifiers for our second experiment (transfer) run:
```commandline
python3 experiments/concept_eval/eval_concept_transfer.py -c experiments/configs/concept_eval/transfer_lin.json
python3 experiments/concept_eval/eval_concept_transfer.py -c experiments/configs/concept_eval/transfer_mlp.json
```
   
Then use the jupyter-notebook `experiments/concept_eval/prelim_experiment.ipynb` to process the results. Finally you can re-create the plots and results from our paper (and a few more) with by using the notebooks `experiments/concept_eval/experiment1.ipynb` and `experiments/concept_eval/experiment2.ipynb`.


## Cite this
TODO


