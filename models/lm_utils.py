import os
import pickle
import numpy as np

import torch
from embedding import BertHuggingface
from sklearn.metrics import f1_score, precision_score, recall_score


# for huggingface models 

def load_or_compute_embeddings(texts, lm, dataset, split, emb_dir):
    model_name = lm.model.config._name_or_path
    pooling = lm.pooling

    if '/' in model_name:
        model_name = model_name.replace('/', '_')
    save_file = (emb_dir+'%s_%s_%s_%s.pickle' % (dataset, split, model_name, pooling))
    if os.path.exists(save_file):
        print("load precomputed embeddings for %s set" % split)
        with open(save_file, 'rb') as handle:
            embeddings = pickle.load(handle)
        assert len(embeddings) == len(texts)

    else:
        print("could not find %s" % save_file)
        print("embed %s set..." % split)
        embeddings = lm.embed(texts)
        with open(save_file, 'wb') as handle:
            pickle.dump(embeddings, handle)
    return embeddings


def get_pretrained_model(model_name, n_classes, batch_size=1, pooling='mean', multi_label=False):    
    if multi_label: 
        lm = BertHuggingface(n_classes, model_name=model_name, batch_size=batch_size, pooling=pooling, loss_function=torch.nn.BCEWithLogitsLoss)
    else:
        lm = BertHuggingface(n_classes, model_name=model_name, batch_size=batch_size, pooling=pooling)
    return lm


def get_pretrained_model_with_batch_size_lookup(model_name, n_classes, batch_size_lookup, pooling='mean', multi_label=False):
    if not model_name in batch_size_lookup.keys():
        print("batch size for model", model_name, "not specified, use 1")
        batch_size = 1
    else:
        batch_size = batch_size_lookup[model_name]
    
    lm = get_pretrained_model(model_name, n_classes, batch_size=batch_size, pooling=pooling, multi_label=multi_label)
    return lm


def get_finetuned_model(model_name, dataset_name, batch_size_lookup, n_classes, multi_label, X_train, y_train, X_test=None, y_test=None, pooling='mean', epochs=5, run_eval=False):
    if run_eval:
        assert (X_test is not None and y_test is not None)
        assert len(X_test) == len(y_test)
    assert len(X_train) == len(y_train)

    lm = get_pretrained_model_with_batch_size_lookup(model_name, n_classes, batch_size_lookup, pooling, multi_label)
    checkpoint_dir = ('models/feature_eval/%s/%i_%s' % (dataset_name, epochs, model_name))

    if not os.path.isdir(checkpoint_dir):
        lm.retrain(X_train, y_train, epochs=epochs)
        lm.save(checkpoint_dir)
    else:
        lm.load(checkpoint_dir)

    if run_eval:
        pred = lm.predict(X_test)
        if multi_label:
            # (!) huggingface classification heads output logits -> threshold is 0
            y_pred = (np.array(pred) >= 0.0).astype(int)
        else:  # single-label
            y_pred = np.argmax(pred, axis=1)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        print("finetuned LM (%s) achieved F1=%.2f, Recall=%.2f, Precision=%.2f on the test set" % (model_name, f1, recall, precision))

    return lm


# openai models

def get_embeddings(texts, dataset_name, split, embedding_model, emb_dir):
    save_file = (emb_dir+'%s_%s_%s.pickle' % (dataset_name, split, embedding_model))
    if os.path.exists(save_file):
        with open(save_file, 'rb') as handle:
            emb_dict = pickle.load(handle)
            assert emb_dict['model'] == embedding_model, ("trying to load embeddings of %s, but savefile contains embeddings of %s" % (embedding_model, emb_dict['model']))
            embeddings = emb_dict['embeddings']
            assert len(embeddings) == len(texts), ("excepted embeddings for %i samples, but got %i instead" % (len(texts), len(embeddings)))
    else:
        print("tried to load embeddings from %s, but the file doesn't exist" % save_file)
        return None
    return embeddings.astype(np.float32)


def get_defining_term_embeddings(defining_term_dict: dict, embedding_model: str, emb_dir: str) -> dict:
    dict_path = (emb_dir+'word_phrase_dict_%s.pickle' % embedding_model)
    assert os.path.exists(dict_path), ("dictionary for defining terms does not exist: %s" % dict_path)
    with open(dict_path, 'rb') as handle:
        loaded_dict = pickle.load(handle)
        prev_model = loaded_dict['model']
        assert prev_model == embedding_model, ("trying to load embeddings of %s, but savefile contains embeddings of %s" % (embedding_model, prev_model))
        word_phrase_emb_dict = loaded_dict['emb_dict']

    emb_defining_attr = {attr: {} for attr in defining_term_dict.keys()}
    for attr, dterms_dict in defining_term_dict.items():
        for group, dterms in dterms_dict.items():
            embs = []
            for term in dterms:
                assert term in word_phrase_emb_dict.keys(), ("term \"%s\" missing in embedding lookup" % term)
                embs.append(word_phrase_emb_dict[term])
            emb_defining_attr[attr][group] = np.asarray(embs, dtype=np.float32)
    return emb_defining_attr

