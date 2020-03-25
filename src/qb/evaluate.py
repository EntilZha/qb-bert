from typing import Text
import pickle
from collections import Counter
import json
import logging
import os

import comet_ml
from allennlp.models.archival import load_archive
import toml
import tqdm
import numpy as np
import pandas as pd

from qb import constants
from qb.predictor import QbPredictor


log = logging.getLogger(__name__)


def compute_accuracy(predictor: QbPredictor, instances):
    vocab = predictor._model.vocab  # pylint: disable=protected-access
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
    log.info("First dev accuracy: %s", accuracy_start_dev)

    dataset_reader.first_sentence_only = False
    dataset_reader.full_question_only = True
    dev_full_question = dataset_reader.read("guessdev")
    accuracy_full_dev = compute_accuracy(predictor, dev_full_question)
    log.info("Full dev accuracy: %s", accuracy_full_dev)

    log.info("log_to_comet: %s", log_to_comet)
    if log_to_comet:
        experiment = comet_ml.get_global_experiment()
        experiment.log_metric("dev_first_accuracy", accuracy_start_dev)
        experiment.log_metric("dev_full_accuracy", accuracy_full_dev)


def create_guesser_report(mapped_qanta_path: str, config_path: str):
    log.info("reading qb questions")
    with open(mapped_qanta_path) as f:
        all_questions = [q for q in json.load(f)["questions"] if q["page"] is not None]

    guesstrain_questions = [q for q in all_questions if q["fold"] == "guesstrain"]
    guesstrain_pages = {q["page"] for q in guesstrain_questions}
    log.info("N guess train questions: %s", len(guesstrain_questions))
    log.info("N guess train pages: %s", len(guesstrain_pages))

    train_example_counts = Counter()
    for q in guesstrain_questions:
        train_example_counts[q["page"]] += 1

    with open(config_path) as f:
        conf = toml.load(f)
    trial = conf.get("trial", 0)
    generated_id = conf["generated_id"]
    model_dir = conf["serialization_dir"]
    fold = "guessdev"
    for fold in constants.REPORT_FOLDS:
        log.info("starting report generation for fold=%s", fold)
        questions = [q for q in all_questions if q["fold"] == fold]
        pages = {q["page"] for q in questions}
        log.info("N Questions: %s", len(questions))
        log.info("N Pages: %s", len(pages))
        unanswerable_page_percent = len(pages - guesstrain_pages) / len(pages)

        answerable_questions = 0
        for q in questions:
            if q["page"] in guesstrain_pages:
                answerable_questions += 1
        unanswerable_question_percent = 1 - answerable_questions / len(questions)

        question_df = pd.DataFrame(
            [
                {
                    "page": q["page"],
                    "qanta_id": q["qanta_id"],
                    "text_length": len(q["text"]),
                    "n_train": train_example_counts[q["page"]],
                    "category": q["category"],
                }
                for q in questions
            ]
        )
        log.info("Loading char df")
        char_guess_df = pd.read_pickle(os.path.join(model_dir, guess_df_path("char", fold)))
        char_df = char_guess_df.merge(question_df, on="qanta_id")
        char_df["correct"] = (char_df.guess == char_df.page).astype("int")
        char_df["char_percent"] = (char_df["char_index"] / char_df["text_length"]).clip(upper=1.0)

        log.info("Loading first df")
        first_guess_df = pd.read_pickle(os.path.join(model_dir, guess_df_path("first", fold)))
        first_df = first_guess_df.merge(question_df, on="qanta_id").sort_values(
            "score", ascending=False
        )
        first_df["correct"] = (first_df.guess == first_df.page).astype("int")
        grouped_first_df = first_df.groupby("qanta_id")
        first_accuracy = grouped_first_df.nth(0).correct.mean()
        first_recall = grouped_first_df.agg({"correct": "max"}).correct.mean()

        log.info("Loading full df")
        full_guess_df = pd.read_pickle(os.path.join(model_dir, guess_df_path("full", fold)))
        full_df = full_guess_df.merge(question_df, on="qanta_id").sort_values(
            "score", ascending=False
        )
        full_df["correct"] = (full_df.guess == full_df.page).astype("int")
        grouped_full_df = full_df.groupby("qanta_id")
        full_accuracy = grouped_full_df.nth(0).correct.mean()
        full_recall = grouped_full_df.agg({"correct": "max"}).correct.mean()

        out_path = os.path.join(model_dir, f"guesser_report_{fold}.pickle")
        log.info("Saving report to: %s", out_path)
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "first_accuracy": first_accuracy,
                    "first_recall": first_recall,
                    "full_accuracy": full_accuracy,
                    "full_recall": full_recall,
                    "char_df": char_df,
                    "first_df": first_df,
                    "full_df": full_df,
                    "unanswerable_answer_percent": unanswerable_page_percent,
                    "unanswerable_question_percent": unanswerable_question_percent,
                    "guesser_name": conf["model"],
                    "guesser_params": conf["params"],
                    "trial": trial,
                    "generated_id": generated_id,
                },
                f,
            )


def guess_df_path(granularity: str, fold: str) -> str:
    return f"guesses_{granularity}_{fold}.pickle"
