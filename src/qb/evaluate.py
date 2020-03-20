from typing import Text
import os
import tqdm
import comet_ml
import numpy as np
from allennlp.models.archival import load_archive
from qb.predictor import QbPredictor
from qb.data import QantaReader, QANTA_GUESSDEV


def compute_accuracy(predictor, instances):
    vocab = predictor._model.vocab
    batch_size = 32
    idx = 0
    preds = []
    progress = tqdm.tqdm(total=len(instances))
    while idx < len(instances):
        batch = instances[idx : idx + batch_size]
        preds.extend(predictor.predict_batch_instance(batch))
        idx += batch_size
        progress.update(idx)
    pred_pages = []
    correct = []
    for example, p in zip(instances, preds):
        predicted_page = vocab.get_token_from_index(p["preds"], namespace="page_labels")
        correct.append(example["metadata"]["page"] == predicted_page)
        pred_pages.append(predicted_page)

    return np.mean(correct)


def score_model(serialization_dir: Text, log_to_comet=False):
    archive = load_archive(os.path.join(serialization_dir, "model.tar.gz"), cuda_device=0)
    predictor = QbPredictor.from_archive(archive, predictor_name="qb.predictor.QbPredictor",)
    dataset_reader = predictor._dataset_reader  # pylint: disable=protected-access
    dataset_reader.include_label = False

    dataset_reader.first_sentence_only = True
    dataset_reader.full_question_only = False
    dev_first_sentence = dataset_reader.read("guessdev")
    accuracy_start_dev = compute_accuracy(predictor, dev_first_sentence)
    print("first", "dev", accuracy_start_dev)

    dataset_reader.first_sentence_only = False
    dataset_reader.full_question_only = True
    dev_full_question = dataset_reader.read("guessdev")
    accuracy_full_dev = compute_accuracy(predictor, dev_full_question)
    print("full", "dev", accuracy_full_dev)

    if log_to_comet:
        experiment = comet_ml.get_global_experiment()
        experiment.log_metric(f"dev_first_accuracy", accuracy_start_dev)
        experiment.log_metric(f"dev_full_accuracy", accuracy_full_dev)
